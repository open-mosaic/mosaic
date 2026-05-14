// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "aggregation.h"
#include "param.h"
#include "profiler_otel.h"
#include "scale_up_inference.h"

/**
 * @file aggregation_scale_up.cc
 * @brief Turns inferred scale-up transfers into rank/channel aggregates.
 *
 * The inference layer produces total bytes, transfer count and a representative transfer size, but
 * Grafana also needs per-channel transfer time. The key detail is that scale-up channels run in
 * parallel: dividing wall-clock network time by the total transfer count across all channels would
 * understate per-transfer time by roughly the channel overlap factor.
 *
 * This file therefore distributes time in two different ways:
 * - Channel transfer intervals use channel-local span.
 *   - If KernelCh timestamps exist, each channel uses its own `[startTs, endTs]` span.
 *   - Otherwise every active channel falls back to the inferred network window
 *     `[op.startTs, op.startTs + collectiveTime * networkTimeFraction]`.
 *   - Per-channel transfer time is `channelSpan / transfersThisChannel`.
 * - Collective / P2P cached transfer time uses the sum of per-channel spans.
 *   - Average transfer time exported on the parent op is
 *     `sum(channelSpanUs) / totalTransfers`, not `wallClock / totalTransfers`.
 *   - This matches the semantics of the proxy-op path, where transfer times are summed per transfer
 *     even when channels overlap in wall clock.
 *
 * CUDA Graph mode has no stable per-channel timestamps, so the fallback assumes each active channel
 * is busy for the inferred network window and uses
 * `activeChannels * networkTime / totalTransfers` for the parent transfer-time cache while only
 * exporting byte/count volume for rank/channel aggregates.
 */

/**
 * @brief Finalize scale-up operations that completed without ProxyOp children.
 *
 * This path handles both collective and P2P operations that rely on inferred
 * transfer characteristics instead of proxy-thread transfer timing.
 *
 * @param[in] handleToOp Map of operation handles to in-progress operations.
 * @param[in] isColl True for collective operations, false for P2P operations.
 */
void WindowAggregator::finalizeScaleUpOperations(std::map<const void*, InProgressOperation>& handleToOp, bool isColl)
{
    double networkPct = (double)OTEL_GET_PARAM(ScaleUpNetworkPct);

    CommunicatorState* commState = nullptr;
    if (!handleToOp.empty())
    {
        const otelEventHandle_t* eventPtr = static_cast<const otelEventHandle_t*>(handleToOp.begin()->first);
        commState                         = const_cast<CommunicatorState*>(eventPtr ? eventPtr->commState : nullptr);
    }

    const bool isCudaGraphDriven = commState && commState->isScaleUpCudaGraphDriven();

    auto computeCollectiveTimeUs = [&](const void* opHandle, const InProgressOperation& op, bool& hasKernelEvents,
                                       const std::vector<otelEventHandle_t>*& kernelEvents) -> double
    {
        auto kernelIt   = kernelChByParent.find(opHandle);
        hasKernelEvents = (kernelIt != kernelChByParent.end() && !kernelIt->second.empty());
        kernelEvents    = hasKernelEvents ? &kernelIt->second : nullptr;

        double lastKernelEndTs = 0.0;
        if (hasKernelEvents)
        {
            for (const otelEventHandle_t& kch : kernelIt->second)
                if (kch.endTs > lastKernelEndTs) lastKernelEndTs = kch.endTs;
        }
        return hasKernelEvents ? (lastKernelEndTs - op.startTs) : (op.endTs - op.startTs);
    };

    auto inferTransfers = [&](const InProgressOperation& op) -> InferredTransfers
    {
        if (isColl) return inferCollectiveTransfers(op.func, op.algo, op.bytes, op.nRanks, op.nChannels, networkPct);
        return inferP2PTransfers(op.bytes, op.nChannels, networkPct);
    };

    auto recordCollectiveCountTime = [&](const InProgressOperation& op, double collectiveTimeUs)
    {
        if (isColl)
            collectives[op.key].addCollective(op.bytes, collectiveTimeUs);
        else
            p2ps[op.key].addP2P(op.bytes, collectiveTimeUs);
    };

    auto recordTransferCacheBatch =
        [&](const InProgressOperation& op, int numTransfers, size_t totalTransferBytes, double averageTransferTimeUs)
    {
        const double totalTime = (double)numTransfers * averageTransferTimeUs;
        if (isColl)
            collectives[op.key].addTransferBatch(numTransfers, totalTransferBytes, totalTime);
        else
            p2ps[op.key].addTransferBatch(numTransfers, totalTransferBytes, totalTime);
    };

    auto addRankChannelVolumeOnly =
        [&](const void* opHandle, const InProgressOperation& op, const InferredTransfers& inf)
    {
        if (inf.numTransfers <= 0 || inf.totalRankBytes == 0) return;

        const otelEventHandle_t* eventPtr = static_cast<const otelEventHandle_t*>(opHandle);
        const CommunicatorState* cs       = eventPtr ? eventPtr->commState : nullptr;
        const int peer                    = op.peer;

        auto bytesInTransferRange = [&](int startTransferIdx, int transferCount) -> size_t
        {
            const size_t baseBytes  = inf.totalRankBytes / (size_t)inf.numTransfers;
            const size_t remainder  = inf.totalRankBytes % (size_t)inf.numTransfers;
            const int boundedStart  = startTransferIdx < 0 ? 0 : startTransferIdx;
            const int boundedEnd    = std::min(inf.numTransfers, boundedStart + transferCount);
            const size_t extraBytes = boundedEnd > boundedStart ? std::min<size_t>(remainder, (size_t)boundedEnd) -
                                                                      std::min<size_t>(remainder, (size_t)boundedStart)
                                                                : 0;
            return (size_t)(boundedEnd - boundedStart) * baseBytes + extraBytes;
        };

        std::string rankKey              = getScaleUpRankTransferKey(cs, peer, !isColl);
        AggregatedTransfer& rankTransfer = rankTransfers[rankKey];
        rankTransfer.totalBytes += inf.totalRankBytes;
        rankTransfer.count += inf.numTransfers;

        int nCh           = inf.numChannels > 0 ? inf.numChannels : 1;
        int base          = inf.numTransfers / nCh;
        int rem           = inf.numTransfers % nCh;
        int transferIndex = 0;
        for (int ch = 0; ch < nCh; ch++)
        {
            int transfersThisCh = base + (ch < rem ? 1 : 0);
            if (transfersThisCh <= 0) continue;
            std::string channelKey              = getScaleUpChannelTransferKey(cs, peer, (uint8_t)ch, !isColl);
            AggregatedTransfer& channelTransfer = channelTransfers[channelKey];
            channelTransfer.totalBytes += bytesInTransferRange(transferIndex, transfersThisCh);
            channelTransfer.count += transfersThisCh;
            transferIndex += transfersThisCh;
        }
    };

    auto bytesForTransferIndex = [&](const InferredTransfers& inf, int transferIndex) -> size_t
    {
        if (inf.numTransfers <= 0 || transferIndex < 0 || transferIndex >= inf.numTransfers)
        {
            return 0;
        }

        const size_t baseBytes = inf.totalRankBytes / (size_t)inf.numTransfers;
        const size_t remainder = inf.totalRankBytes % (size_t)inf.numTransfers;
        return baseBytes + ((size_t)transferIndex < remainder ? 1U : 0U);
    };

    for (auto& pair : handleToOp)
    {
        const void* opHandle    = pair.first;
        InProgressOperation& op = pair.second;

        if (!isColl && alltoAllP2PHandles.count(opHandle)) continue;
        if (op.seenProxyOps > 0) continue;

        bool hasKernelEvents                               = false;
        const std::vector<otelEventHandle_t>* kernelEvents = nullptr;
        double collectiveTimeUs = computeCollectiveTimeUs(opHandle, op, hasKernelEvents, kernelEvents);

        if (collectiveTimeUs <= 0)
        {
            OTEL_WARN(NCCL_INIT, "Skipping scale-up %s with invalid duration=%.2f us: %s, bytes=%zu",
                      isColl ? "Coll" : "P2P", collectiveTimeUs, op.key.c_str(), op.bytes);
            continue;
        }

        recordCollectiveCountTime(op, collectiveTimeUs);

        InferredTransfers inferred = inferTransfers(op);
        if (inferred.numTransfers <= 0 || inferred.totalRankBytes == 0)
        {
            OTEL_TRACE(NCCL_INIT, "Scale-up %s (no inferred transfers): %s, bytes=%zu, duration=%.2f us",
                       isColl ? "Coll" : "P2P", op.key.c_str(), op.bytes, collectiveTimeUs);
            continue;
        }

        if (isCudaGraphDriven)
        {
            double averageTransferTime = 0.0;
            if (inferred.networkTimeFraction > 0.0 && inferred.numTransfers > 0)
            {
                const double networkTime = collectiveTimeUs * inferred.networkTimeFraction;
                const int activeChannels =
                    std::min(inferred.numTransfers, inferred.numChannels > 0 ? inferred.numChannels : 1);
                averageTransferTime = (networkTime * activeChannels) / inferred.numTransfers;
            }

            recordTransferCacheBatch(op, inferred.numTransfers, inferred.totalRankBytes, averageTransferTime);
            addRankChannelVolumeOnly(opHandle, op, inferred);

            OTEL_TRACE(NCCL_INIT,
                       "Finalized scale-up %s (CUDA Graph): %s, bytes=%zu, collectiveTime=%.2f us, transfers=%d, "
                       "perTransferBytes=%zu",
                       isColl ? "Coll" : "P2P", op.key.c_str(), op.bytes, collectiveTimeUs, inferred.numTransfers,
                       inferred.perTransferBytes);
            continue;
        }

        const double networkTime           = collectiveTimeUs * inferred.networkTimeFraction;
        const otelEventHandle_t* eventPtr  = static_cast<const otelEventHandle_t*>(opHandle);
        const CommunicatorState* eventComm = eventPtr ? eventPtr->commState : nullptr;
        int peer                           = op.peer;
        std::string rankKey                = getScaleUpRankTransferKey(eventComm, peer, !isColl);
        AggregatedTransfer& rankTransfer   = rankTransfers[rankKey];

        int nCh                    = inferred.numChannels > 0 ? inferred.numChannels : 1;
        int base                   = inferred.numTransfers / nCh;
        int rem                    = inferred.numTransfers % nCh;
        int transferIndex          = 0;
        double totalTransferTimeUs = 0.0;

        for (int ch = 0; ch < nCh; ch++)
        {
            int transfersThisCh = base + (ch < rem ? 1 : 0);
            if (transfersThisCh <= 0) continue;

            double channelStartTs = op.startTs;
            double channelEndTs   = op.startTs + networkTime;
            if (kernelEvents && !kernelEvents->empty())
            {
                for (const auto& kch : *kernelEvents)
                {
                    if ((int)kch.kernelCh.channelId == ch && kch.endTs > kch.startTs)
                    {
                        channelStartTs = kch.startTs;
                        channelEndTs   = kch.endTs;
                        break;
                    }
                }
            }

            double channelSpan = channelEndTs - channelStartTs;
            if (channelSpan <= 0.0) continue;

            totalTransferTimeUs += channelSpan;
            const double channelTransferTime    = channelSpan / transfersThisCh;
            std::string channelKey              = getScaleUpChannelTransferKey(eventComm, peer, (uint8_t)ch, !isColl);
            AggregatedTransfer& channelTransfer = channelTransfers[channelKey];

            for (int i = 0; i < transfersThisCh; i++)
            {
                const size_t transferBytes = bytesForTransferIndex(inferred, transferIndex++);
                double intervalStart       = channelStartTs + (channelSpan * i) / transfersThisCh;
                double intervalEnd         = intervalStart + channelTransferTime;
                if (intervalEnd > channelEndTs) intervalEnd = channelEndTs;

                rankTransfer.addTransferWithTimestamps(transferBytes, channelTransferTime, intervalStart, intervalEnd);
                channelTransfer.addTransferWithTimestamps(transferBytes, channelTransferTime, intervalStart,
                                                          intervalEnd);
            }
        }

        const double averageTransferTime = (inferred.numTransfers > 0 && totalTransferTimeUs > 0.0)
                                               ? (totalTransferTimeUs / inferred.numTransfers)
                                               : 0.0;
        recordTransferCacheBatch(op, inferred.numTransfers, inferred.totalRankBytes, averageTransferTime);

        OTEL_TRACE(NCCL_INIT,
                   "Finalized scale-up %s: %s, bytes=%zu, collectiveTime=%.2f us, networkTime=%.2f us, transfers=%d, "
                   "perTransferBytes=%zu, perTransferTime=%.2f us, mode=%s",
                   isColl ? "Coll" : "P2P", op.key.c_str(), op.bytes, collectiveTimeUs, networkTime,
                   inferred.numTransfers, inferred.perTransferBytes, averageTransferTime,
                   eventComm ? eventComm->getScaleUpExecModeString() : "unknown");
    }
}