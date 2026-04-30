// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "aggregation.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "param.h"
#include "profiler_otel.h"  // For OTEL_TRACE
#include "scale_up_inference.h"

/**
 * @brief Construct a WindowAggregator for a specific rank.
 *
 * @param[in] rank Global rank of the process (used for key generation).
 */
WindowAggregator::WindowAggregator(int rank) : rank(rank) {}

/**
 * @brief Generate aggregation key for a collective event.
 *
 * @param[in] event Collective event handle.
 *
 * @return Key string in format: Comm<hash>_Func_Algo_Proto_nChannels
 */
std::string WindowAggregator::getCollectiveKey(const otelEventHandle_t& event) const
{
    std::stringstream ss;
    uint64_t commHash = event.commState ? event.commState->comm_hash : 0;
    ss << "Comm" << commHash << "_" << (event.coll.func ? event.coll.func : "NULL") << "_"
       << (event.coll.algo ? event.coll.algo : "NULL") << "_" << (event.coll.proto ? event.coll.proto : "NULL") << "_"
       << (int)event.coll.nChannels << "Chnl";
    return ss.str();
}

/**
 * @brief Generate aggregation key for a P2P event.
 *
 * For P2P communicators (nranks=2), this generates a key that identifies:
 * - The communicator hash (for correlation with the peer side)
 * - The source (hostname, global_rank, local_rank)
 * - The function (Send/Recv)
 * - The peer rank within the communicator (0 or 1)
 * - Number of channels
 *
 * Note: We only know our own identity (hostname, ranks). The peer's identity
 * can be correlated in Grafana using the shared comm_hash (both sides have the same hash).
 *
 * @param[in] event P2P event handle.
 *
 * @return Key string in format: Comm<hash>_<hostname>_<global>_<local>_<func>_ToPeer<peer>_<nChannels>Chnl
 */
std::string WindowAggregator::getP2PKey(const otelEventHandle_t& event) const
{
    std::stringstream ss;

    uint64_t commHash    = event.commState ? event.commState->comm_hash : 0;
    std::string hostname = event.commState ? event.commState->hostname : "unknown";
    // For P2P: rank within the P2P comm (0 or 1) represents the pipeline number
    int src_pipeline = event.commState ? event.commState->rank : rank;
    const char* func = event.p2p.func ? event.p2p.func : "NULL";

    // Key format: Comm<hash>_(<hostname>)_<func>_Pipeline<src>ToPipeline<dst>_<nChannels>Chnl
    // For P2P comms (nranks=2), both src and peer (0 or 1) represent pipeline numbers
    // Example: Comm123456_(romeo)_Send_Pipeline0ToPipeline1_2Chnl
    ss << "Comm" << commHash << "_(" << hostname << ")_" << func << "_Pipeline" << src_pipeline << "ToPipeline"
       << event.p2p.peer << "_" << (int)event.p2p.nChannels << "Chnl";

    return ss.str();
}

/**
 * @brief Generate aggregation key for rank-to-rank transfers.
 *
 * For COLLECTIVE comms: Uses rank within communicator (no hostname needed)
 * For P2P comms: Uses pipeline numbers (rank within P2P comm represents pipeline)
 *
 * @param[in] commHash Communicator hash.
 * @param[in] peer Destination peer rank within the communicator.
 * @param[in] commState Communicator state for hostname, rank, and comm_type.
 *
 * @return Key string:
 *   - COLLECTIVE: Comm<hash>_Rank<X>_ToPeer<peer>
 *   - P2P: Comm<hash>_<hostname>_Pipeline<src>_ToPipeline<peer>
 */
std::string WindowAggregator::getRankTransferKey(uint64_t commHash, int peer, const CommunicatorState* commState) const
{
    std::stringstream ss;

    bool isP2P = commState && commState->comm_type == CommunicatorState::CommType::P2P;

    if (isP2P)
    {
        // P2P: Show hostname and pipeline numbers
        // rank within P2P comm (0 or 1) represents the pipeline number
        std::string hostname = commState ? commState->hostname : "unknown";
        int src_pipeline     = commState ? commState->rank : rank;
        ss << "Comm" << commHash << "_" << hostname << "_Pipeline" << src_pipeline << "_ToPipeline" << peer;
    }
    else
    {
        // COLLECTIVE: Show rank within communicator (no hostname)
        int comm_rank = commState ? commState->rank : rank;
        ss << "Comm" << commHash << "_Rank" << comm_rank << "_ToPeer" << peer;
    }
    return ss.str();
}

/**
 * @brief Generate aggregation key for per-channel transfers.
 *
 * For COLLECTIVE comms: Uses rank within communicator
 * For P2P comms: Uses pipeline numbers
 *
 * @param[in] event ProxyOp event handle containing channel and peer info.
 *
 * @return Key string:
 *   - COLLECTIVE: Comm<hash>_Rank<X>_ToPeer<peer>_Chnl<id>
 *   - P2P: Comm<hash>_<hostname>_Pipeline<src>_ToPipeline<peer>_Chnl<id>
 */
std::string WindowAggregator::getChannelTransferKey(const otelEventHandle_t& event) const
{
    std::stringstream ss;
    uint64_t commHash = event.commState ? event.commState->comm_hash : 0;

    bool isP2P = event.commState && event.commState->comm_type == CommunicatorState::CommType::P2P;

    if (isP2P)
    {
        std::string hostname = event.commState ? event.commState->hostname : "unknown";
        int src_pipeline     = event.commState ? event.commState->rank : event.rank;
        ss << "Comm" << commHash << "_" << hostname << "_Pipeline" << src_pipeline << "_ToPipeline"
           << event.proxyOp.peer << "_Chnl" << (int)event.proxyOp.channelId;
    }
    else
    {
        int comm_rank = event.commState ? event.commState->rank : event.rank;
        ss << "Comm" << commHash << "_Rank" << comm_rank << "_ToPeer" << event.proxyOp.peer << "_Chnl"
           << (int)event.proxyOp.channelId;
    }
    return ss.str();
}

/**
 * @brief Generate key for transfer channel grouping.
 *
 * @param[in] channelId Channel ID.
 *
 * @return Key string in format: Chnl<id>
 */
std::string WindowAggregator::getTransferChannelKey(uint8_t channelId) const
{
    std::stringstream ss;
    ss << "Chnl" << (int)channelId;
    return ss.str();
}

/**
 * @brief Add an event to the aggregator.
 *
 * Processes events based on type:
 * - Coll/P2P: Tracked for later linking with ProxyOps
 * - ProxyStep: Aggregated to parent ProxyOp
 * - ProxyOp: Stored for linking in finalize()
 *
 * @param[in] event Event handle to process.
 */
void WindowAggregator::addEvent(const otelEventHandle_t& event)
{
    // Phase 1: Track Coll/P2P - store by eHandle for later ProxyOp correlation
    if (event.type == ncclProfileColl)
    {
        InProgressOperation op;
        op.key            = getCollectiveKey(event);
        op.startTs        = event.startTs;
        op.endTs          = event.endTs;
        op.bytes          = event.coll.bytes;
        op.seenProxyOps   = 0;
        op.lastProxyOpEnd = event.startTs;
        op.func           = event.coll.func;
        op.algo           = event.coll.algo;
        op.nChannels      = event.coll.nChannels;
        op.nRanks         = event.commState ? event.commState->nranks : 1;
        // For ring collectives, the send peer is the next rank in the ring
        int comm_rank = event.commState ? event.commState->rank : rank;
        op.peer       = op.nRanks > 1 ? (comm_rank + 1) % op.nRanks : 0;

        collHandleToOp[&event] = op;

        OTEL_TRACE(NCCL_INIT, "Tracked Coll: %s, eHandle=%p, endTs=%.2f", op.key.c_str(), &event, op.endTs);
    }
    else if (event.type == ncclProfileP2p)
    {
        InProgressOperation op;
        op.key            = getP2PKey(event);
        op.startTs        = event.startTs;
        op.endTs          = event.endTs;
        op.bytes          = event.p2p.bytes;
        op.seenProxyOps   = 0;
        op.lastProxyOpEnd = event.startTs;
        op.func           = event.p2p.func;
        op.nChannels      = event.p2p.nChannels;
        op.nRanks         = event.commState ? event.commState->nranks : 1;
        op.peer           = event.p2p.peer;

        p2pHandleToOp[&event] = op;

        // If this P2P Send has a P2pApi parent, register it for AlltoAll collective grouping.
        // All P2P tasks from a single AlltoAll share the same P2pApi event handle as parentObj.
        if (event.parentObj && p2pApiHandleToFunc.count(event.parentObj))
        {
            p2pHandlesByApiHandle[event.parentObj].push_back(&event);
            OTEL_TRACE(NCCL_INIT, "Linked P2P Send (peer=%d) to P2pApi parent %p (%s)", event.p2p.peer, event.parentObj,
                       p2pApiHandleToFunc.at(event.parentObj).c_str());
        }

        OTEL_TRACE(NCCL_INIT, "Tracked P2P: %s, eHandle=%p, endTs=%.2f", op.key.c_str(), &event, op.endTs);
    }
    // Phase 1b: Track P2pApi group markers (AlltoAll collective anchors)
    // P2pApi events carry the original collective function name (e.g., "AlltoAll").
    // They are stored before P2p events in the buffer, so p2pApiHandleToFunc is populated
    // before P2p events reference them via parentObj.
    else if (event.type == ncclProfileP2pApi)
    {
        if (event.p2pApi.func)
        {
            p2pApiHandleToFunc[&event] = std::string(event.p2pApi.func);
            OTEL_TRACE(NCCL_INIT, "Tracked P2pApi marker: func=%s, eHandle=%p", event.p2pApi.func, &event);
        }
        return;  // No further processing; just a grouping anchor
    }
    // Phase 2: Aggregate ProxyStep transfers to their parent ProxyOp
    else if (event.type == ncclProfileProxyStep)
    {
        // Only process ProxySteps with SendWait (actual transfers)
        if (event.proxyStep.hasSendWait)
        {
            double transferTime = event.endTs - event.proxyStep.sendWaitTs;

            // Skip transfers with transferTime <= 0 (0 means infinite bandwidth, negative is invalid)
            // Do not clamp to 0 as that would incorrectly represent infinite bandwidth
            if (transferTime <= 0)
            {
                OTEL_WARN(NCCL_INIT, "Skipping ProxyStep with invalid transferTime=%.2f us (size=%zu)", transferTime,
                          event.proxyStep.transSize);
                return;  // Skip this transfer
            }

            // Find parent ProxyOp via parentObj
            const void* proxyOpHandle = event.parentObj;
            if (proxyOpHandle)
            {
                // Aggregate to ProxyOp with timestamps for interval-based rate calculation
                // sendWaitTs is the absolute start time, endTs is the absolute end time
                proxyOpTransfers[proxyOpHandle].addTransferWithTimestamps(event.proxyStep.transSize, transferTime,
                                                                          event.proxyStep.sendWaitTs, event.endTs);

                OTEL_TRACE(
                    NCCL_INIT, "Aggregated ProxyStep to ProxyOp %p: size=%zu, time=%.2f us, interval=[%.2f, %.2f]",
                    proxyOpHandle, event.proxyStep.transSize, transferTime, event.proxyStep.sendWaitTs, event.endTs);
            }
            else
            {
                // ProxyStep with SendWait should have a parent ProxyOp
                // This is unexpected and indicates a potential data consistency issue
                OTEL_WARN(NCCL_INIT, "ProxyStep with SendWait has NULL parentObj (size=%zu, transferTime=%.2f us)",
                          event.proxyStep.transSize, transferTime);
            }
        }
    }
    // Phase 3: Just store ProxyOp for later linking in finalize() (after all ProxySteps are aggregated)
    else if (event.type == ncclProfileProxyOp)
    {
        // Store a copy of the ProxyOp event for processing in finalize()
        proxyOps[&event] = event;

        // Update timing in parent Coll/P2P (needed for duration calculation)
        const void* parentHandle = getRootCollectiveHandle(event.parentObj);
        if (parentHandle)
        {
            auto collIt = collHandleToOp.find(parentHandle);
            if (collIt != collHandleToOp.end())
            {
                collIt->second.lastProxyOpEnd = std::max(collIt->second.lastProxyOpEnd, event.endTs);
                collIt->second.seenProxyOps++;
            }

            auto p2pIt = p2pHandleToOp.find(parentHandle);
            if (p2pIt != p2pHandleToOp.end())
            {
                p2pIt->second.lastProxyOpEnd = std::max(p2pIt->second.lastProxyOpEnd, event.endTs);
                p2pIt->second.seenProxyOps++;
            }
        }

        OTEL_TRACE(NCCL_INIT, "Stored ProxyOp %p for finalization: parentObj=%p", &event, event.parentObj);
    }
    // =========================================================================
    // Kernel event handling (scale-up analysis)
    // =========================================================================
    else if (event.type == ncclProfileKernelCh)
    {
        // Group KernelCh events by their parent Coll/P2P handle.
        // parentObj for KernelCh points directly to the Coll/P2P task event handle.
        if (event.parentObj)
        {
            kernelChByParent[event.parentObj].push_back(event);
            OTEL_TRACE(NCCL_INIT, "Tracked KernelCh: parent=%p, channelId=%d, hasStop=%d", event.parentObj,
                       event.kernelCh.channelId, event.kernelCh.hasStop);
        }
    }
    else if (event.type == ncclProfileKernelLaunch)
    {
        kernelLaunches.push_back(event);
        OTEL_TRACE(NCCL_INIT, "Tracked KernelLaunch: parent=%p", event.parentObj);
    }
}

/**
 * @brief Finalize aggregation and calculate metrics.
 *
 * Links ProxyOps to their parent Collectives/P2Ps, calculates correct durations
 * (Coll/P2P START → Last ProxyOp STOP), and prepares aggregated data for export.
 * Must be called after all events are added.
 */
void WindowAggregator::finalize()
{
    OTEL_TRACE(NCCL_INIT, "Finalizing: %zu ProxyOps, %zu ProxyOp transfers", proxyOps.size(), proxyOpTransfers.size());

#ifdef PROFILER_OTEL_ENABLE_TRACE
    int proxyOpsWithTransfers    = 0;
    int proxyOpsWithoutTransfers = 0;
#endif

    // =========================================================================
    // Classify communicator execution mode (CUDA Graph vs non-CUDA Graph)
    //
    // We assume a communicator is either CUDA-Graph-driven or not.
    //
    // Metric source for classification:
    // - KernelCh GPU timer start tick (pTimerStart).
    //
    // Detection heuristic:
    // - In non-CUDA-Graph execution, each collective/p2p op has its own pTimerStart.
    // - In CUDA Graph replay, multiple distinct parent ops can share the same pTimerStart.
    //
    // Implementation:
    // - Scan KernelCh events and detect any pTimerStart used by >= 2 distinct parent
    //   operation handles.
    //
    // Perf:
    // - One linear scan over KernelCh events with an unordered_map. Runs once per
    //   window per communicator.
    // =========================================================================
    CommunicatorState* commState = nullptr;
    if (!collHandleToOp.empty())
        commState = const_cast<CommunicatorState*>(
            static_cast<const otelEventHandle_t*>(collHandleToOp.begin()->first)->commState);
    else if (!p2pHandleToOp.empty())
        commState = const_cast<CommunicatorState*>(
            static_cast<const otelEventHandle_t*>(p2pHandleToOp.begin()->first)->commState);
    else if (!proxyOps.empty())
        commState = const_cast<CommunicatorState*>(proxyOps.begin()->second.commState);

    if (commState)
    {
        auto execMode =
            static_cast<CommunicatorState::ScaleUpExecMode>(commState->scaleUpExecMode.load(std::memory_order_acquire));

        // We allow a one-way upgrade to CUDA_GRAPH if we ever observe CUDA-graph evidence
        // in a later window (e.g., warm-up windows before graph replay may look non-graph).
        if (execMode != CommunicatorState::ScaleUpExecMode::CUDA_GRAPH)
        {
            bool cudaGraphDetected = false;
            bool sawAnyPTimerStart = false;

            std::unordered_map<uint64_t, const void*> firstHandleByPTimerStart;
            firstHandleByPTimerStart.reserve(kernelChByParent.size() * 2);

            for (const auto& parentPair : kernelChByParent)
            {
                const void* parentHandle = parentPair.first;
                for (const auto& kch : parentPair.second)
                {
                    // pTimerStart is recorded at KernelCh start_event; hasStop indicates that
                    // KernelChStop was recorded (pTimerStop captured). For CUDA Graph detection
                    // we only require a non-zero pTimerStart.
                    if (kch.kernelCh.pTimerStart == 0) continue;
                    sawAnyPTimerStart = true;

                    uint64_t pTimerStart = kch.kernelCh.pTimerStart;
                    auto it              = firstHandleByPTimerStart.find(pTimerStart);
                    if (it == firstHandleByPTimerStart.end())
                    {
                        firstHandleByPTimerStart.emplace(pTimerStart, parentHandle);
                    }
                    else if (it->second != parentHandle)
                    {
                        cudaGraphDetected = true;
                        break;
                    }
                }
                if (cudaGraphDetected) break;
            }

            if (cudaGraphDetected)
            {
                commState->scaleUpExecMode.store(static_cast<uint8_t>(CommunicatorState::ScaleUpExecMode::CUDA_GRAPH),
                                                 std::memory_order_release);
                OTEL_TRACE(NCCL_INIT, "Scale-up communicator classified: commHash=%lu mode=%s",
                           (unsigned long)commState->comm_hash, commState->getScaleUpExecModeString());
            }
            else if (execMode == CommunicatorState::ScaleUpExecMode::UNKNOWN && sawAnyPTimerStart)
            {
                commState->scaleUpExecMode.store(
                    static_cast<uint8_t>(CommunicatorState::ScaleUpExecMode::NON_CUDA_GRAPH),
                    std::memory_order_release);
                OTEL_TRACE(NCCL_INIT, "Scale-up communicator classified: commHash=%lu mode=%s",
                           (unsigned long)commState->comm_hash, commState->getScaleUpExecModeString());
            }
        }
    }

    // Phase 1: Link ProxyOps to their parent Coll/P2P and aggregate transfers
    for (const auto& proxyPair : proxyOps)
    {
        const otelEventHandle_t& proxyOp = proxyPair.second;
        const void* proxyOpHandle        = proxyPair.first;

        // Get aggregated transfers for this ProxyOp
        auto proxyIt = proxyOpTransfers.find(proxyOpHandle);

        if (proxyIt != proxyOpTransfers.end())
        {
#ifdef PROFILER_OTEL_ENABLE_TRACE
            proxyOpsWithTransfers++;
#endif
            const AggregatedTransfer& transfers = proxyIt->second;

            // Find parent Coll or P2P (if any)
            const void* parentHandle = getRootCollectiveHandle(proxyOp.parentObj);

            if (parentHandle)
            {
                // Link to Coll
                auto collIt = collHandleToOp.find(parentHandle);
                if (collIt != collHandleToOp.end())
                {
                    collectives[collIt->second.key].addTransferBatch(transfers.count, transfers.totalBytes,
                                                                     transfers.totalTimeUs);

                    OTEL_TRACE(NCCL_INIT, "Linked ProxyOp %p to Coll %s: bytes=%zu, time=%.2f us, count=%d",
                               proxyOpHandle, collIt->second.key.c_str(), transfers.totalBytes, transfers.totalTimeUs,
                               transfers.count);
                }

                // Link to P2P
                auto p2pIt = p2pHandleToOp.find(parentHandle);
                if (p2pIt != p2pHandleToOp.end())
                {
                    p2ps[p2pIt->second.key].addTransferBatch(transfers.count, transfers.totalBytes,
                                                             transfers.totalTimeUs);

                    OTEL_TRACE(NCCL_INIT, "Linked ProxyOp %p to P2P %s: bytes=%zu, time=%.2f us", proxyOpHandle,
                               p2pIt->second.key.c_str(), transfers.totalBytes, transfers.totalTimeUs);
                }
            }

            // Aggregate for rank/channel metrics (for ALL ProxyOps, with or without parent)
            uint64_t commHash           = proxyOp.commState ? proxyOp.commState->comm_hash : 0;
            std::string rankTransferKey = getRankTransferKey(commHash, proxyOp.proxyOp.peer, proxyOp.commState);
            rankTransfers[rankTransferKey].totalBytes += transfers.totalBytes;
            rankTransfers[rankTransferKey].totalTimeUs += transfers.totalTimeUs;
            rankTransfers[rankTransferKey].count += transfers.count;
            // Merge the individual ProxyStep data points from this ProxyOp
            rankTransfers[rankTransferKey].lr.merge(transfers.lr);
            // Merge transfer intervals for bandwidth calculation based on active transfer time
            rankTransfers[rankTransferKey].mergeIntervals(transfers);

            std::string channelTransferKey = getChannelTransferKey(proxyOp);
            channelTransfers[channelTransferKey].totalBytes += transfers.totalBytes;
            channelTransfers[channelTransferKey].totalTimeUs += transfers.totalTimeUs;
            channelTransfers[channelTransferKey].count += transfers.count;
            // Merge the individual ProxyStep data points from this ProxyOp
            channelTransfers[channelTransferKey].lr.merge(transfers.lr);
            // Merge transfer intervals for bandwidth calculation based on active transfer time
            channelTransfers[channelTransferKey].mergeIntervals(transfers);
        }
        else
        {
#ifdef PROFILER_OTEL_ENABLE_TRACE
            proxyOpsWithoutTransfers++;
#endif
            // This is expected for ProxyOps that span window boundaries
            // (ProxyOp in one window, ProxySteps in another)
        }
    }

#ifdef PROFILER_OTEL_ENABLE_TRACE
    // Log summary of ProxyOp linking (only if there are issues)
    if (proxyOpsWithoutTransfers > 0)
    {
        OTEL_TRACE(NCCL_INIT,
                   "Finalized ProxyOps: %d with transfers, %d without transfers (likely window boundary issue)",
                   proxyOpsWithTransfers, proxyOpsWithoutTransfers);
    }
    else
    {
        OTEL_TRACE(NCCL_INIT, "Finalized ProxyOps: %d with transfers", proxyOpsWithTransfers);
    }
#endif

    // Phase 2: Calculate correct Coll durations
    // Normal case (scale-out): startTs -> lastProxyOpEnd (when ProxyOps exist)
    // Scale-up (no ProxyOps): handled separately via finalizeScaleUpOperations()
    for (auto& pair : collHandleToOp)
    {
        InProgressOperation& op = pair.second;

        if (op.seenProxyOps > 0)
        {
            double realDuration = op.lastProxyOpEnd - op.startTs;
            OTEL_TRACE(NCCL_INIT,
                       "Finalized Coll: %s, bytes=%zu, duration=%.2f us (start=%.2f, lastProxyOpEnd=%.2f, proxyOps=%d)",
                       op.key.c_str(), op.bytes, realDuration, op.startTs, op.lastProxyOpEnd, op.seenProxyOps);

            if (realDuration <= 0)
            {
                OTEL_WARN(NCCL_INIT, "Skipping Coll with invalid duration=%.2f us: %s, bytes=%zu", realDuration,
                          op.key.c_str(), op.bytes);
                continue;
            }
            collectives[op.key].addCollective(op.bytes, realDuration);
        }
        // seenProxyOps == 0 is handled by finalizeScaleUpOperations() below
    }

    // Same for P2P
    for (auto& pair : p2pHandleToOp)
    {
        InProgressOperation& op = pair.second;

        if (op.seenProxyOps > 0)
        {
            double realDuration = op.lastProxyOpEnd - op.startTs;
            OTEL_TRACE(NCCL_INIT, "Finalized P2P: %s, bytes=%zu, duration=%.2f us (proxyOps=%d)", op.key.c_str(),
                       op.bytes, realDuration, op.seenProxyOps);

            if (realDuration <= 0)
            {
                OTEL_WARN(NCCL_INIT, "Skipping P2P with invalid duration=%.2f us: %s, bytes=%zu", realDuration,
                          op.key.c_str(), op.bytes);
                continue;
            }
            p2ps[op.key].addP2P(op.bytes, realDuration);
        }
        // seenProxyOps == 0 is handled by finalizeScaleUpOperations() below
    }

    // =========================================================================
    // Phase 3: Scale-up inference for operations without ProxyOps
    // When no proxy operations exist (scale-up path), use KernelCh events
    // and collective metadata to infer transfer metrics.
    // =========================================================================
    finalizeScaleUpOperations(collHandleToOp, true);
    finalizeScaleUpOperations(p2pHandleToOp, false);

    // =========================================================================
    // Phase 4: Synthesize Collective metrics for AlltoAll-style operations.
    //
    // NCCL decomposes AlltoAll (and similar collectives) into individual P2P
    // Send tasks.  Each task gets its own P2P metric, but the high-level
    // AlltoAll operation is invisible in the Collective section of the dashboard.
    //
    // When ncclProfileP2pApi events are tracked (one per AlltoAll call), all
    // corresponding P2P Send events share the same P2pApi handle as parentObj.
    // This phase groups those sends back into a single Collective entry so
    // AlltoAll operations appear in the Collective dashboard section.
    //
    // - Total bytes  = sum of bytes across all P2P Sends in the group.
    // - Start time   = earliest P2P Send start among the group.
    // - End time     = latest ProxyOp end (or P2P stop if no ProxyOps) in the group.
    // - Transfer data is accumulated from the per-peer p2ps entries (already finalized
    //   in Phase 2 above).
    // =========================================================================
    for (const auto& apiPair : p2pHandlesByApiHandle)
    {
        const void* apiHandle                      = apiPair.first;
        const std::vector<const void*>& p2pHandles = apiPair.second;

        auto funcIt = p2pApiHandleToFunc.find(apiHandle);
        if (funcIt == p2pApiHandleToFunc.end()) continue;
        const std::string& funcName = funcIt->second;

        // Compute AlltoAll collective timing and bytes across all grouped P2P Sends.
        size_t totalBytes          = 0;
        double startTs             = std::numeric_limits<double>::max();
        double endTs               = 0.0;
        int totalTransferCount     = 0;
        size_t totalTransferBytes  = 0;
        double totalTransferTimeUs = 0.0;

        const otelEventHandle_t* firstEvent = nullptr;
        for (const void* p2pHandle : p2pHandles)
        {
            auto opIt = p2pHandleToOp.find(p2pHandle);
            if (opIt == p2pHandleToOp.end()) continue;
            const InProgressOperation& op = opIt->second;
            if (!firstEvent) firstEvent = static_cast<const otelEventHandle_t*>(p2pHandle);

            totalBytes += op.bytes;
            startTs = std::min(startTs, op.startTs);

            double opEnd = (op.seenProxyOps > 0) ? op.lastProxyOpEnd : op.endTs;
            endTs        = std::max(endTs, opEnd);

            // Accumulate transfer statistics from the per-peer p2ps entries that
            // were finalized in Phase 2.  The transfer data is already present there
            // regardless of whether we also export it per-peer via P2P metrics.
            auto p2pIt = p2ps.find(op.key);
            if (p2pIt != p2ps.end())
            {
                totalTransferCount += p2pIt->second.cachedTotalTransferCount;
                totalTransferBytes += p2pIt->second.cachedTotalTransferBytes;
                totalTransferTimeUs += p2pIt->second.cachedTotalTransferTimeUs;
            }
        }

        if (!firstEvent || !firstEvent->commState) continue;

        // Only synthesize Collective metrics for communicators that are not pure P2P
        // (nranks == 2, CommType::P2P).  Explicit ncclSend/ncclRecv on a 2-rank
        // pipeline-parallel communicator also produce P2pApi events (the same code
        // path in NCCL is shared), so those must remain in the P2P section only.
        if (firstEvent->commState->comm_type == CommunicatorState::CommType::P2P)
        {
            OTEL_TRACE(NCCL_INIT,
                       "Skipping AlltoAll collective synthesis for P2P communicator (nranks=2, pipeline-parallel): "
                       "func=%s, comm_hash=%lu",
                       funcName.c_str(), firstEvent->commState->comm_hash);
            continue;
        }

        double duration = endTs - startTs;
        if (startTs >= endTs || duration <= 0)
        {
            OTEL_WARN(NCCL_INIT, "Skipping AlltoAll collective synthesis with invalid duration=%.2f us (func=%s)",
                      duration, funcName.c_str());
            continue;
        }

        // Build collective key: Comm<hash>_<func>_<nranks>Ranks
        // Using nranks instead of algo/proto (unavailable for AlltoAll-as-P2P).
        std::stringstream ss;
        ss << "Comm" << firstEvent->commState->comm_hash << "_" << funcName << "_" << firstEvent->commState->nranks
           << "Ranks";
        std::string collKey = ss.str();

        collectives[collKey].addCollective(totalBytes, duration);
        if (totalTransferCount > 0)
        {
            collectives[collKey].addTransferBatch(totalTransferCount, totalTransferBytes, totalTransferTimeUs);
        }

        OTEL_TRACE(NCCL_INIT, "Synthesized AlltoAll collective: key=%s, bytes=%zu, duration=%.2f us, transfers=%d",
                   collKey.c_str(), totalBytes, duration, totalTransferCount);
    }
}

/**
 * @brief Get the root Coll/P2P handle from parentObj chain.
 *
 * Traverses the parentObj chain to find the root Collective or P2P operation.
 * Used to link ProxyOps to their parent operations.
 *
 * @param[in] parentObj Parent object pointer (may be nullptr).
 *
 * @return Pointer to root Coll/P2P handle, or nullptr if not found.
 */
const void* WindowAggregator::getRootCollectiveHandle(const void* parentObj) const
{
    if (!parentObj) return nullptr;

    // ProxyOp's parentObj should point directly to Coll or P2P eHandle
    if (collHandleToOp.count(parentObj))
    {
        return parentObj;
    }

    if (p2pHandleToOp.count(parentObj))
    {
        return parentObj;
    }

    return nullptr;  // Not found or not a tracked operation
}

std::string WindowAggregator::getScaleUpRankTransferKey(const CommunicatorState* commState, int peer) const
{
    uint64_t commHash = commState ? commState->comm_hash : 0;
    return getRankTransferKey(commHash, peer, commState);
}

std::string WindowAggregator::getScaleUpChannelTransferKey(const CommunicatorState* commState, int peer,
                                                           uint8_t channelId) const
{
    std::stringstream ss;
    uint64_t commHash = commState ? commState->comm_hash : 0;

    bool isP2P = commState && commState->comm_type == CommunicatorState::CommType::P2P;

    if (isP2P)
    {
        std::string hostname = commState ? commState->hostname : "unknown";
        int src_pipeline     = commState ? commState->rank : rank;
        ss << "Comm" << commHash << "_" << hostname << "_Pipeline" << src_pipeline << "_ToPipeline" << peer << "_Chnl"
           << (int)channelId;
    }
    else
    {
        int comm_rank = commState ? commState->rank : rank;
        ss << "Comm" << commHash << "_Rank" << comm_rank << "_ToPeer" << peer << "_Chnl" << (int)channelId;
    }
    return ss.str();
}

void WindowAggregator::finalizeScaleUpOperations(std::map<const void*, InProgressOperation>& handleToOp, bool isColl)
{
    double networkPct = (double)OTEL_GET_PARAM(ScaleUpNetworkPct);

    // =========================================================================
    // Select execution mode (CUDA Graph vs non-CUDA Graph)
    //
    // This uses the communicator-level classification performed in finalize().
    // =========================================================================
    CommunicatorState* commState = nullptr;
    if (!handleToOp.empty())
    {
        const auto* eventPtr = static_cast<const otelEventHandle_t*>(handleToOp.begin()->first);
        commState            = const_cast<CommunicatorState*>(eventPtr ? eventPtr->commState : nullptr);
    }

    const bool isCudaGraphDriven = commState && commState->isScaleUpCudaGraphDriven();

    // =========================================================================
    // Shared scale-up logic (both modes)
    //
    // Metric sources:
    // - collectiveTimeUs:
    //     start = Coll/P2P START timestamp (op.startTs)
    //     end   = max KernelCh CPU endTs for that parent op (if present)
    //     fallback = Coll/P2P STOP timestamp (op.endTs)
    //
    // - transfer sizes/counts (inferred):
    //     inferCollectiveTransfers / inferP2PTransfers based on (func/algo/bytes/nRanks/nChannels, ScaleUpNetworkPct)
    //
    // Mode-specific behavior:
    // - non-CUDA-Graph:
    //     emits timing-derived intervals → bandwidth + latency + transfer time
    // - CUDA-Graph-driven:
    //     emits size/volume only (bytes + counts), suppresses timing-derived metrics
    // =========================================================================
    auto computeCollectiveTimeUs = [&](const void* opHandle, const InProgressOperation& op, bool& hasKernelEvents,
                                       const std::vector<otelEventHandle_t>*& kernelEvents) -> double
    {
        auto kernelIt   = kernelChByParent.find(opHandle);
        hasKernelEvents = (kernelIt != kernelChByParent.end() && !kernelIt->second.empty());
        kernelEvents    = hasKernelEvents ? &kernelIt->second : nullptr;

        double lastKernelEndTs = 0.0;
        if (hasKernelEvents)
        {
            for (const auto& kch : kernelIt->second)
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
        [&](const InProgressOperation& op, int numTransfers, size_t perTransferBytes, double perTransferTimeUs)
    {
        const size_t totalBytes = (size_t)numTransfers * perTransferBytes;
        const double totalTime  = (double)numTransfers * perTransferTimeUs;
        if (isColl)
            collectives[op.key].addTransferBatch(numTransfers, totalBytes, totalTime);
        else
            p2ps[op.key].addTransferBatch(numTransfers, totalBytes, totalTime);
    };

    auto addRankChannelVolumeOnly =
        [&](const void* opHandle, const InProgressOperation& op, const InferredTransfers& inf)
    {
        if (inf.numTransfers <= 0 || inf.perTransferBytes == 0) return;

        const auto* eventPtr        = static_cast<const otelEventHandle_t*>(opHandle);
        const CommunicatorState* cs = eventPtr ? eventPtr->commState : nullptr;
        const int peer              = op.peer;

        const size_t totalBytes = (size_t)inf.numTransfers * inf.perTransferBytes;
        std::string rankKey     = getScaleUpRankTransferKey(cs, peer);
        rankTransfers[rankKey].totalBytes += totalBytes;
        rankTransfers[rankKey].count += inf.numTransfers;

        // Distribute counts/bytes across channels without creating any time/intervals.
        int nCh  = inf.numChannels > 0 ? inf.numChannels : 1;
        int base = inf.numTransfers / nCh;
        int rem  = inf.numTransfers % nCh;
        for (int ch = 0; ch < nCh; ch++)
        {
            int transfersThisCh = base + (ch < rem ? 1 : 0);
            if (transfersThisCh <= 0) continue;
            std::string channelKey = getScaleUpChannelTransferKey(cs, peer, (uint8_t)ch);
            channelTransfers[channelKey].totalBytes += (size_t)transfersThisCh * inf.perTransferBytes;
            channelTransfers[channelKey].count += transfersThisCh;
        }
    };

    for (auto& pair : handleToOp)
    {
        const void* opHandle    = pair.first;
        InProgressOperation& op = pair.second;

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

        // Always export collective count/time.
        recordCollectiveCountTime(op, collectiveTimeUs);

        // Run inference to get transfer characteristics (sizes/counts + network fraction).
        InferredTransfers inferred = inferTransfers(op);

        if (inferred.numTransfers <= 0 || inferred.perTransferBytes == 0)
        {
            OTEL_TRACE(NCCL_INIT, "Scale-up %s (no inferred transfers): %s, bytes=%zu, duration=%.2f us",
                       isColl ? "Coll" : "P2P", op.key.c_str(), op.bytes, collectiveTimeUs);
            continue;
        }

        // CUDA Graph mode: size/volume only (no time, no intervals, no LR points).
        if (isCudaGraphDriven)
        {
            recordTransferCacheBatch(op, inferred.numTransfers, inferred.perTransferBytes, 0.0 /*time*/);
            addRankChannelVolumeOnly(opHandle, op, inferred);
            OTEL_TRACE(NCCL_INIT, "CUDA-Graph scale-up %s (count + size/volume only): %s, bytes=%zu, duration=%.2f us",
                       isColl ? "Coll" : "P2P", op.key.c_str(), op.bytes, collectiveTimeUs);
            continue;
        }

        // Non-CUDA-Graph mode: emit timing-derived metrics.
        double networkTime = collectiveTimeUs * inferred.networkTimeFraction;

        // Channels operate in parallel: each channel independently steps through
        // networkTime, so per-transfer time is based on per-channel steps only.
        // numTransfers includes all channels (numChannels × stepsPerChannel), but
        // dividing by numChannels gives the sequential steps within one channel.
        int transfersPerChannel =
            inferred.numChannels > 0 ? inferred.numTransfers / inferred.numChannels : inferred.numTransfers;
        if (transfersPerChannel < 1) transfersPerChannel = 1;
        double perTransferTime = networkTime / (double)transfersPerChannel;

        OTEL_TRACE(NCCL_INIT,
                   "Scale-up %s: %s, bytes=%zu, duration=%.2f us, networkTime=%.2f us, "
                   "transfers=%d, perTransfer=%zu bytes / %.2f us, totalRankBytes=%zu, kernelEvents=%s",
                   isColl ? "Coll" : "P2P", op.key.c_str(), op.bytes, collectiveTimeUs, networkTime,
                   inferred.numTransfers, inferred.perTransferBytes, perTransferTime, inferred.totalRankBytes,
                   hasKernelEvents ? "yes" : "no");

        // Feed inferred transfers into the operation's transfer cache (one batch).
        recordTransferCacheBatch(op, inferred.numTransfers, inferred.perTransferBytes, perTransferTime);

        // =====================================================================
        // Generate rank-level and channel-level transfer metrics from inferred data
        // =====================================================================

        // Find the commState from the operation's handle (stored in the event buffer)
        const otelEventHandle_t* eventPtr  = static_cast<const otelEventHandle_t*>(opHandle);
        const CommunicatorState* commState = eventPtr ? eventPtr->commState : nullptr;

        // Use the peer from the operation (ring neighbor for Coll, explicit peer for P2P)
        int peer = op.peer;

        std::string rankKey = getScaleUpRankTransferKey(commState, peer);

        // Synthesize transfer intervals spread across the collective timeline.
        // All channels overlap in time, so wrap each transfer's index back into
        // the per-channel window [op.startTs, op.startTs + networkTime] using
        // modulo. This produces numTransfers data points for regression while
        // keeping timestamps within the actual collective duration.
        for (int i = 0; i < inferred.numTransfers; i++)
        {
            int step             = i % transfersPerChannel;
            double intervalStart = op.startTs + (double)step * perTransferTime;
            double intervalEnd   = intervalStart + perTransferTime;

            rankTransfers[rankKey].addTransferWithTimestamps(inferred.perTransferBytes, perTransferTime, intervalStart,
                                                             intervalEnd);
        }

        // Per-channel metrics: distribute transfers across channels
        if (hasKernelEvents && kernelEvents)
        {
            // Use actual KernelCh events for per-channel breakdown.
            // For the data points (bytes, time) that feed into linear regression and
            // aggregate stats, we use the same evenly-divided perTransferTime as the
            // rank level to avoid GPU timing jitter degrading the regression fit.
            // The actual kernel event timestamps are still used for interval boundaries
            // which feed into rate/bandwidth calculation via getActiveTime().
            for (const auto& kch : *kernelEvents)
            {
                std::string channelKey = getScaleUpChannelTransferKey(commState, peer, kch.kernelCh.channelId);
                double channelStartTs  = kch.startTs;
                double channelEndTs    = kch.endTs;

                double channelDuration = channelEndTs - channelStartTs;
                if (channelDuration <= 0) channelDuration = perTransferTime * transfersPerChannel;

                double perChannelIntervalTime = channelDuration / transfersPerChannel;

                for (int i = 0; i < transfersPerChannel; i++)
                {
                    double intervalStart = channelStartTs + (double)i * perChannelIntervalTime;
                    double intervalEnd   = intervalStart + perChannelIntervalTime;

                    channelTransfers[channelKey].addTransferWithTimestamps(inferred.perTransferBytes, perTransferTime,
                                                                           intervalStart, intervalEnd);
                }
            }
        }
        else
        {
            // No KernelCh events: divide evenly across nChannels.
            // nCh and transfersPerChannel are already computed above.
            int nCh = inferred.numChannels > 0 ? inferred.numChannels : 1;

            for (int ch = 0; ch < nCh; ch++)
            {
                std::string channelKey = getScaleUpChannelTransferKey(commState, peer, (uint8_t)ch);

                // Channels run in parallel: each independently spans [op.startTs, op.startTs + networkTime].
                // Do not offset by ch — all channels start at op.startTs.
                for (int i = 0; i < transfersPerChannel; i++)
                {
                    double intervalStart = op.startTs + (double)i * perTransferTime;
                    double intervalEnd   = intervalStart + perTransferTime;

                    channelTransfers[channelKey].addTransferWithTimestamps(inferred.perTransferBytes, perTransferTime,
                                                                           intervalStart, intervalEnd);
                }
            }
        }
    }
}
