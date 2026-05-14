// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "aggregation.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <limits>
#include <set>
#include <unordered_map>
#include <vector>

#include "param.h"
#include "profiler_otel.h"  // For OTEL_TRACE
#include "scale_up_inference.h"

/**
 * @brief Resolve the configured linear-regression mode for transfer fitting.
 *
 * @return Configured regression mode, defaulting to AVG for unset or invalid input.
 */
static LinearRegression::Mode getLinearRegressionMode()
{
    const char* modeStr = ncclParamLinearRegressionMode();
    if (strcmp(modeStr, "MIN") == 0)
    {
        return LinearRegression::Mode::MIN;
    }
    if (strcmp(modeStr, "AVG") != 0 && strcmp(modeStr, "") != 0)
    {
        OTEL_WARN(NCCL_INIT, "Unknown LinearRegressionMode '%s', defaulting to AVG", modeStr);
    }
    return LinearRegression::Mode::AVG;
}

/**
 * @brief Append an integer value to a string without stream formatting.
 *
 * @tparam Int Integer type to append.
 * @param[in,out] out Destination string.
 * @param[in] value Integer value to append.
 */
template <typename Int>
static inline void appendInteger(std::string& out, Int value)
{
    char buffer[32];
    auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    out.append(buffer, result.ptr);
}

/**
 * @brief Return the provided token or a fallback literal for null pointers.
 *
 * @param[in] value Optional C string token.
 *
 * @return `value` when non-null, otherwise `"NULL"`.
 */
static inline const char* getTokenOrNull(const char* value)
{
    return value ? value : "NULL";
}

/**
 * @brief Return the communicator hostname or a shared fallback string.
 *
 * @param[in] commState Optional communicator state.
 *
 * @return Communicator hostname when available, otherwise `"unknown"`.
 */
static inline const std::string& getHostnameOrUnknown(const CommunicatorState* commState)
{
    static const std::string kUnknownHostname = "unknown";
    return commState ? commState->hostname : kUnknownHostname;
}

/**
 * @brief Append the standard communicator key prefix to an aggregation key.
 *
 * @param[in,out] out Destination key string.
 * @param[in] commHash Communicator hash to encode.
 */
static inline void appendCommPrefix(std::string& out, uint64_t commHash)
{
    out.append("Comm");
    appendInteger(out, commHash);
}

/**
 * @brief Build a rank or channel transfer aggregation key.
 *
 * @param[in] commHash Communicator hash.
 * @param[in] commState Optional communicator state providing hostname and rank.
 * @param[in] fallbackRank Rank value to use when `commState` is null.
 * @param[in] peer Destination peer rank or pipeline.
 * @param[in] isP2P Whether the key should use P2P naming.
 * @param[in] includeChannel Whether to append a channel suffix.
 * @param[in] channelId Channel identifier appended when `includeChannel` is true.
 *
 * @return Fully formatted transfer aggregation key.
 */
static std::string buildTransferKey(uint64_t commHash, const CommunicatorState* commState, int fallbackRank, int peer,
                                    bool isP2P, bool includeChannel, int channelId)
{
    const std::string& hostname = getHostnameOrUnknown(commState);
    const int sourceRank        = commState ? commState->rank : fallbackRank;

    std::string key;
    key.reserve(48 + hostname.size());
    appendCommPrefix(key, commHash);
    key.push_back('_');

    if (isP2P)
    {
        key.append(hostname);
        key.append("_Pipeline");
        appendInteger(key, sourceRank);
        key.append("_ToPipeline");
        appendInteger(key, peer);
    }
    else
    {
        key.append("Rank");
        appendInteger(key, sourceRank);
        key.append("_ToPeer");
        appendInteger(key, peer);
    }

    if (includeChannel)
    {
        key.append("_Chnl");
        appendInteger(key, channelId);
    }

    return key;
}

/**
 * @brief Construct an empty aggregated transfer accumulator.
 */
AggregatedTransfer::AggregatedTransfer() : totalBytes(0), totalTimeUs(0.0), count(0), lr(getLinearRegressionMode()) {}

/**
 * @brief Add a transfer interval alongside the aggregate transfer totals.
 *
 * @param[in] bytes Transfer size in bytes.
 * @param[in] timeUs Transfer duration in microseconds.
 * @param[in] startTs Absolute transfer start timestamp.
 * @param[in] endTs Absolute transfer end timestamp.
 */
void AggregatedTransfer::addTransferWithTimestamps(size_t bytes, double timeUs, double startTs, double endTs)
{
    addTransfer(bytes, timeUs);
    if (startTs < endTs)
    {
        intervals.push_back({startTs, endTs});
    }
}

/**
 * @brief Append transfer intervals from another aggregate.
 *
 * @param[in] other Aggregate whose intervals should be appended.
 */
void AggregatedTransfer::mergeIntervals(const AggregatedTransfer& other)
{
    intervals.insert(intervals.end(), other.intervals.begin(), other.intervals.end());
}

/**
 * @brief Compute the union of recorded transfer intervals.
 *
 * @return Total active transfer time in microseconds.
 */
double AggregatedTransfer::getActiveTime() const
{
    if (intervals.empty()) return 0.0;

    std::vector<std::pair<double, double>> sorted = intervals;
    std::sort(sorted.begin(), sorted.end());

    double activeTime   = 0.0;
    double currentStart = sorted[0].first;
    double currentEnd   = sorted[0].second;

    for (size_t i = 1; i < sorted.size(); i++)
    {
        if (sorted[i].first <= currentEnd)
        {
            currentEnd = std::max(currentEnd, sorted[i].second);
        }
        else
        {
            activeTime += currentEnd - currentStart;
            currentStart = sorted[i].first;
            currentEnd   = sorted[i].second;
        }
    }

    activeTime += currentEnd - currentStart;
    return activeTime;
}

/**
 * @brief Compute bandwidth from the merged active transfer time.
 *
 * @param[out] rateMBps Bandwidth in MB/s using the decimal MB convention.
 *
 * @return true when the rate was computed, false when no valid data exists.
 */
bool AggregatedTransfer::getRateFromActiveTime(double& rateMBps) const
{
    double activeTime = getActiveTime();
    if (activeTime <= 0.0 || totalBytes == 0)
    {
        rateMBps = 0.0;
        return false;
    }

    rateMBps = (double)totalBytes / activeTime;
    return true;
}

/**
 * @brief Estimate latency from the linear-regression fit of transfer size to time.
 *
 * @param[out] latencyUs Estimated latency in microseconds.
 *
 * @return true when a stable latency estimate could be derived.
 */
bool AggregatedTransfer::getLatencyFromLinearRegression(double& latencyUs) const
{
    if (!lr.hasAtLeastThreeDifferentSizes())
    {
        latencyUs = 0.0;
        return false;
    }

    double slope;
    double intercept;
    if (lr.calculate(slope, intercept))
    {
        double rSquared;
        if (lr.calculateRSquared(rSquared) && rSquared >= 0.8)
        {
            latencyUs = (intercept >= 0) ? intercept : 0.0;
            if (slope > 1e-6)
            {
                return true;
            }
        }
    }

    latencyUs = 0.0;
    return false;
}

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
 * @return Key string in format: Comm<hash>_<func>_<algo>_<proto>_<nChannels>Chnl
 */
std::string WindowAggregator::getCollectiveKey(const otelEventHandle_t& event) const
{
    uint64_t commHash = event.commState ? event.commState->comm_hash : 0;
    const char* func  = getTokenOrNull(event.coll.func);
    const char* algo  = getTokenOrNull(event.coll.algo);
    const char* proto = getTokenOrNull(event.coll.proto);

    std::string key;
    key.reserve(32 + std::strlen(func) + std::strlen(algo) + std::strlen(proto));
    appendCommPrefix(key, commHash);
    key.push_back('_');
    key.append(func);
    key.push_back('_');
    key.append(algo);
    key.push_back('_');
    key.append(proto);
    key.push_back('_');
    appendInteger(key, (int)event.coll.nChannels);
    key.append("Chnl");
    return key;
}

/**
 * @brief Generate aggregation key for a P2P event.
 *
 * For P2P communicators, the key captures the communicator hash, hostname,
 * function name, source pipeline, destination pipeline, and channel fan-out.
 *
 * @param[in] event P2P event handle.
 *
 * @return Key string in format: Comm<hash>_(<hostname>)_<func>_Pipeline<src>ToPipeline<peer>_<nChannels>Chnl
 */
std::string WindowAggregator::getP2PKey(const otelEventHandle_t& event) const
{
    uint64_t commHash           = event.commState ? event.commState->comm_hash : 0;
    const std::string& hostname = getHostnameOrUnknown(event.commState);
    // For P2P: rank within the P2P comm (0 or 1) represents the pipeline number
    int src_pipeline = event.commState ? event.commState->rank : rank;
    const char* func = getTokenOrNull(event.p2p.func);

    // Key format: Comm<hash>_(<hostname>)_<func>_Pipeline<src>ToPipeline<dst>_<nChannels>Chnl
    // For P2P comms (nranks=2), both src and peer (0 or 1) represent pipeline numbers
    // Example: Comm123456_(romeo)_Send_Pipeline0ToPipeline1_2Chnl
    std::string key;
    key.reserve(48 + hostname.size() + std::strlen(func));
    appendCommPrefix(key, commHash);
    key.append("_(");
    key.append(hostname);
    key.append(")_");
    key.append(func);
    key.append("_Pipeline");
    appendInteger(key, src_pipeline);
    key.append("ToPipeline");
    appendInteger(key, event.p2p.peer);
    key.push_back('_');
    appendInteger(key, (int)event.p2p.nChannels);
    key.append("Chnl");

    return key;
}

/**
 * @brief Generate aggregation key for rank-to-rank transfers.
 *
 * For COLLECTIVE comms: Uses rank within communicator (no hostname needed)
 * For P2P comms: Uses pipeline numbers (rank within P2P comm represents pipeline)
 *
 * @param[in] commHash Communicator hash.
 * @param[in] peer Destination peer rank within the communicator.
 * @param[in] commState Communicator state for hostname and rank.
 * @param[in] isP2P Whether the owning root operation is P2P.
 *
 * @return Key string:
 *   - COLLECTIVE: Comm<hash>_Rank<X>_ToPeer<peer>
 *   - P2P: Comm<hash>_<hostname>_Pipeline<src>_ToPipeline<peer>
 */
std::string WindowAggregator::getRankTransferKey(uint64_t commHash, int peer, const CommunicatorState* commState,
                                                 bool isP2P) const
{
    return buildTransferKey(commHash, commState, rank, peer, isP2P, false, 0);
}

/**
 * @brief Generate aggregation key for per-channel transfers.
 *
 * For COLLECTIVE comms: Uses rank within communicator
 * For P2P comms: Uses pipeline numbers
 *
 * @param[in] event ProxyOp event handle containing channel and peer info.
 *
 * @param[in] isP2P Whether the owning root operation is P2P.
 *
 * @return Key string:
 *   - COLLECTIVE: Comm<hash>_Rank<X>_ToPeer<peer>_Chnl<id>
 *   - P2P: Comm<hash>_<hostname>_Pipeline<src>_ToPipeline<peer>_Chnl<id>
 */
std::string WindowAggregator::getChannelTransferKey(const otelEventHandle_t& event, bool isP2P) const
{
    uint64_t commHash = event.commState ? event.commState->comm_hash : 0;

    return buildTransferKey(commHash, event.commState, event.rank, event.proxyOp.peer, isP2P, true,
                            (int)event.proxyOp.channelId);
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
    std::string key;
    key.reserve(16);
    key.append("Chnl");
    appendInteger(key, (int)channelId);
    return key;
}

/**
 * @brief Record a collective root event for later aggregation.
 *
 * @param[in] event Collective event handle to track.
 */
void WindowAggregator::trackCollectiveEvent(const otelEventHandle_t& event)
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

    int comm_rank = event.commState ? event.commState->rank : rank;
    op.peer       = op.nRanks > 1 ? (comm_rank + 1) % op.nRanks : 0;

    collHandleToOp[&event] = op;

    OTEL_TRACE(NCCL_INIT, "Tracked Coll: %s, eHandle=%p, endTs=%.2f", op.key.c_str(), &event, op.endTs);
}

/**
 * @brief Record a P2P root event for later aggregation.
 *
 * @param[in] event P2P event handle to track.
 */
void WindowAggregator::trackP2PEvent(const otelEventHandle_t& event)
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

    OTEL_TRACE(NCCL_INIT, "Tracked P2P: %s, eHandle=%p, endTs=%.2f", op.key.c_str(), &event, op.endTs);
}

/**
 * @brief Retain a P2P API marker for grouped AlltoAll reconstruction.
 *
 * @param[in] event P2P API event handle to track.
 */
void WindowAggregator::trackP2pApiEvent(const otelEventHandle_t& event)
{
    if (event.commState) p2pApiEvents.push_back(&event);
}

/**
 * @brief Retain a group event for grouped AlltoAll reconstruction.
 *
 * @param[in] event Group event handle to track.
 */
void WindowAggregator::trackGroupEvent(const otelEventHandle_t& event)
{
    if (event.commState) groupEvents.push_back(&event);
}

/**
 * @brief Accumulate a ProxyStep transfer into its parent ProxyOp aggregate.
 *
 * @param[in] event ProxyStep event handle containing transfer timing.
 */
void WindowAggregator::accumulateProxyStepTransfer(const otelEventHandle_t& event)
{
    if (!event.proxyStep.hasSendWait)
    {
        return;
    }

    double transferTime = event.endTs - event.proxyStep.sendWaitTs;
    if (transferTime <= 0)
    {
        OTEL_WARN(NCCL_INIT, "Skipping ProxyStep with invalid transferTime=%.2f us (size=%zu)", transferTime,
                  event.proxyStep.transSize);
        return;
    }

    const void* proxyOpHandle = event.parentObj;
    if (!proxyOpHandle)
    {
        OTEL_WARN(NCCL_INIT, "ProxyStep with SendWait has NULL parentObj (size=%zu, transferTime=%.2f us)",
                  event.proxyStep.transSize, transferTime);
        return;
    }

    proxyOpTransfers[proxyOpHandle].addTransferWithTimestamps(event.proxyStep.transSize, transferTime,
                                                              event.proxyStep.sendWaitTs, event.endTs);

    OTEL_TRACE(NCCL_INIT, "Aggregated ProxyStep to ProxyOp %p: size=%zu, time=%.2f us, interval=[%.2f, %.2f]",
               proxyOpHandle, event.proxyStep.transSize, transferTime, event.proxyStep.sendWaitTs, event.endTs);
}

/**
 * @brief Store a ProxyOp event and update its parent operation bookkeeping.
 *
 * @param[in] event ProxyOp event handle to retain for finalize().
 */
void WindowAggregator::storeProxyOpForFinalize(const otelEventHandle_t& event)
{
    proxyOps[&event] = event;

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

/**
 * @brief Group a KernelCh event by its parent root operation.
 *
 * @param[in] event KernelCh event handle to track.
 */
void WindowAggregator::trackKernelChannelEvent(const otelEventHandle_t& event)
{
    if (!event.parentObj)
    {
        return;
    }

    kernelChByParent[event.parentObj].push_back(event);
    OTEL_TRACE(NCCL_INIT, "Tracked KernelCh: parent=%p, channelId=%d, hasStop=%d", event.parentObj,
               event.kernelCh.channelId, event.kernelCh.hasStop);
}

/**
 * @brief Record a KernelLaunch event for telemetry-side diagnostics.
 *
 * @param[in] event KernelLaunch event handle to track.
 */
void WindowAggregator::trackKernelLaunchEvent(const otelEventHandle_t& event)
{
    kernelLaunches.push_back(event);
    OTEL_TRACE(NCCL_INIT, "Tracked KernelLaunch: parent=%p", event.parentObj);
}

/**
 * @brief Add an event to the aggregator.
 *
 * Dispatches each event to the helper that owns its aggregation path.
 *
 * @param[in] event Event handle to process.
 */
void WindowAggregator::addEvent(const otelEventHandle_t& event)
{
    switch (event.type)
    {
        case ncclProfileColl:
            trackCollectiveEvent(event);
            return;
        case ncclProfileP2p:
            trackP2PEvent(event);
            return;
        case ncclProfileP2pApi:
            trackP2pApiEvent(event);
            return;
        case ncclProfileGroup:
            trackGroupEvent(event);
            return;
        case ncclProfileProxyStep:
            accumulateProxyStepTransfer(event);
            return;
        case ncclProfileProxyOp:
            storeProxyOpForFinalize(event);
            return;
        case ncclProfileKernelCh:
            trackKernelChannelEvent(event);
            return;
        case ncclProfileKernelLaunch:
            trackKernelLaunchEvent(event);
            return;
        default:
            return;
    }
}

/**
 * @brief Identify grouped P2P windows that should be reconstructed as AlltoAll.
 *
 * @param[out] alltoAllGroups Group event to grouped P2P handle mapping.
 * @param[out] alltoAllExpectedSendCounts Expected send fan-out for each group.
 */
void WindowAggregator::identifyGroupedAlltoAllOperations(
    std::map<const void*, std::vector<const void*>>& alltoAllGroups,
    std::map<const void*, size_t>& alltoAllExpectedSendCounts)
{
    alltoAllP2PHandles.clear();

    for (const otelEventHandle_t* groupEvent : groupEvents)
    {
        if (!groupEvent || !groupEvent->commState) continue;

        std::vector<const void*> groupedP2PHandles;
        for (const auto& p2pPair : p2pHandleToOp)
        {
            const InProgressOperation& op     = p2pPair.second;
            const otelEventHandle_t* p2pEvent = static_cast<const otelEventHandle_t*>(p2pPair.first);
            const CommunicatorState* p2pComm  = p2pEvent ? p2pEvent->commState : nullptr;

            if (p2pComm != groupEvent->commState) continue;
            if (op.startTs < groupEvent->startTs || op.startTs > groupEvent->endTs) continue;

            groupedP2PHandles.push_back(p2pPair.first);
        }

        const size_t sendApiCount = countGroupSendP2pApis(*groupEvent);
        if (isAlltoAllGroup(*groupEvent, groupedP2PHandles, sendApiCount))
        {
            alltoAllGroups[groupEvent]             = groupedP2PHandles;
            alltoAllExpectedSendCounts[groupEvent] = std::max(groupedP2PHandles.size(), sendApiCount);
            alltoAllP2PHandles.insert(groupedP2PHandles.begin(), groupedP2PHandles.end());
        }
    }
}

/**
 * @brief Classify the communicator scale-up execution mode from KernelCh timing.
 */
void WindowAggregator::classifyScaleUpCommunicatorExecutionMode()
{
    CommunicatorState* commState = nullptr;
    if (!collHandleToOp.empty())
        commState = const_cast<CommunicatorState*>(
            static_cast<const otelEventHandle_t*>(collHandleToOp.begin()->first)->commState);
    else if (!p2pHandleToOp.empty())
        commState = const_cast<CommunicatorState*>(
            static_cast<const otelEventHandle_t*>(p2pHandleToOp.begin()->first)->commState);
    else if (!proxyOps.empty())
        commState = const_cast<CommunicatorState*>(proxyOps.begin()->second.commState);

    if (!commState)
    {
        return;
    }

    CommunicatorState::ScaleUpExecMode execMode =
        static_cast<CommunicatorState::ScaleUpExecMode>(commState->scaleUpExecMode.load(std::memory_order_acquire));

    if (execMode == CommunicatorState::ScaleUpExecMode::CUDA_GRAPH)
    {
        return;
    }

    bool cudaGraphDetected     = false;
    bool sawAnyKernelCh        = false;
    bool sawAnyCollPTimerStart = false;

    std::unordered_map<uint64_t, const void*> firstHandleByPTimerStart;
    firstHandleByPTimerStart.reserve(kernelChByParent.size() * 2);

    for (const auto& parentPair : kernelChByParent)
    {
        const void* parentHandle = parentPair.first;
        bool isCollParent        = collHandleToOp.count(parentHandle) > 0;
        for (const otelEventHandle_t& kch : parentPair.second)
        {
            sawAnyKernelCh = true;
            if (kch.kernelCh.pTimerStart == 0) continue;
            if (!isCollParent) continue;
            sawAnyCollPTimerStart = true;

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
    else if (execMode == CommunicatorState::ScaleUpExecMode::UNKNOWN && (sawAnyCollPTimerStart || sawAnyKernelCh))
    {
        commState->scaleUpExecMode.store(static_cast<uint8_t>(CommunicatorState::ScaleUpExecMode::NON_CUDA_GRAPH),
                                         std::memory_order_release);
        OTEL_TRACE(NCCL_INIT, "Scale-up communicator classified: commHash=%lu mode=%s",
                   (unsigned long)commState->comm_hash, commState->getScaleUpExecModeString());
    }
}

/**
 * @brief Link stored ProxyOps to their parent operations and transfer aggregates.
 */
void WindowAggregator::linkProxyOpsToParents()
{
#ifdef PROFILER_OTEL_ENABLE_TRACE
    int proxyOpsWithTransfers    = 0;
    int proxyOpsWithoutTransfers = 0;
#endif

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
                    collIt->second.totalTransferCount += transfers.count;
                    collIt->second.totalTransferBytes += transfers.totalBytes;
                    collIt->second.totalTransferTimeUs += transfers.totalTimeUs;
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
                    p2pIt->second.totalTransferCount += transfers.count;
                    p2pIt->second.totalTransferBytes += transfers.totalBytes;
                    p2pIt->second.totalTransferTimeUs += transfers.totalTimeUs;

                    if (!alltoAllP2PHandles.count(parentHandle))
                    {
                        p2ps[p2pIt->second.key].addTransferBatch(transfers.count, transfers.totalBytes,
                                                                 transfers.totalTimeUs);
                    }

                    OTEL_TRACE(NCCL_INIT, "Linked ProxyOp %p to P2P %s: bytes=%zu, time=%.2f us", proxyOpHandle,
                               p2pIt->second.key.c_str(), transfers.totalBytes, transfers.totalTimeUs);
                }
            }

            // Aggregate for rank/channel metrics (for ALL ProxyOps, with or without parent)
            uint64_t commHash  = proxyOp.commState ? proxyOp.commState->comm_hash : 0;
            bool isP2PTransfer = isP2POperation(parentHandle, proxyOp.commState);
            std::string rankTransferKey =
                getRankTransferKey(commHash, proxyOp.proxyOp.peer, proxyOp.commState, isP2PTransfer);
            AggregatedTransfer& rankTransfer = rankTransfers[rankTransferKey];
            rankTransfer.totalBytes += transfers.totalBytes;
            rankTransfer.totalTimeUs += transfers.totalTimeUs;
            rankTransfer.count += transfers.count;
            // Merge the individual ProxyStep data points from this ProxyOp
            rankTransfer.lr.merge(transfers.lr);
            // Merge transfer intervals for bandwidth calculation based on active transfer time
            rankTransfer.mergeIntervals(transfers);

            std::string channelTransferKey      = getChannelTransferKey(proxyOp, isP2PTransfer);
            AggregatedTransfer& channelTransfer = channelTransfers[channelTransferKey];
            channelTransfer.totalBytes += transfers.totalBytes;
            channelTransfer.totalTimeUs += transfers.totalTimeUs;
            channelTransfer.count += transfers.count;
            // Merge the individual ProxyStep data points from this ProxyOp
            channelTransfer.lr.merge(transfers.lr);
            // Merge transfer intervals for bandwidth calculation based on active transfer time
            channelTransfer.mergeIntervals(transfers);
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
}

/**
 * @brief Finalize operations backed by observed ProxyOp transfer data.
 *
 * @param[in,out] handleToOp Map of root handles to in-progress operations.
 * @param[in] isColl True for collective operations, false for P2P operations.
 */
void WindowAggregator::finalizeOperationsWithProxyData(std::map<const void*, InProgressOperation>& handleToOp,
                                                       bool isColl)
{
    for (auto& pair : handleToOp)
    {
        if (!isColl && alltoAllP2PHandles.count(pair.first)) continue;

        InProgressOperation& op = pair.second;

        if (op.seenProxyOps <= 0) continue;

        double realDuration = op.lastProxyOpEnd - op.startTs;
        OTEL_TRACE(NCCL_INIT,
                   "Finalized %s: %s, bytes=%zu, duration=%.2f us (start=%.2f, lastProxyOpEnd=%.2f, proxyOps=%d)",
                   isColl ? "Coll" : "P2P", op.key.c_str(), op.bytes, realDuration, op.startTs, op.lastProxyOpEnd,
                   op.seenProxyOps);

        if (realDuration <= 0)
        {
            OTEL_WARN(NCCL_INIT, "Skipping %s with invalid duration=%.2f us: %s, bytes=%zu", isColl ? "Coll" : "P2P",
                      realDuration, op.key.c_str(), op.bytes);
            continue;
        }

        if (isColl)
            collectives[op.key].addCollective(op.bytes, realDuration);
        else
            p2ps[op.key].addP2P(op.bytes, realDuration);
    }
}

/**
 * @brief Synthesize grouped AlltoAll collectives from grouped P2P windows.
 *
 * @param[in] alltoAllGroups Group event to P2P membership mapping.
 * @param[in] alltoAllExpectedSendCounts Expected send fan-out for each group.
 */
void WindowAggregator::reconstructGroupedAlltoAllOperations(
    const std::map<const void*, std::vector<const void*>>& alltoAllGroups,
    const std::map<const void*, size_t>& alltoAllExpectedSendCounts)
{
    for (const auto& groupPair : alltoAllGroups)
    {
        const std::vector<const void*>& p2pHandles = groupPair.second;
        const size_t expectedSendCount             = alltoAllExpectedSendCounts.at(groupPair.first);

        size_t totalBytes          = 0;
        double startTs             = std::numeric_limits<double>::max();
        double endTs               = 0.0;
        int totalTransferCount     = 0;
        size_t totalTransferBytes  = 0;
        double totalTransferTimeUs = 0.0;
        size_t trackedSendCount    = 0;

        const otelEventHandle_t* firstEvent = nullptr;
        for (const void* p2pHandle : p2pHandles)
        {
            auto opIt = p2pHandleToOp.find(p2pHandle);
            if (opIt == p2pHandleToOp.end()) continue;
            const InProgressOperation& op = opIt->second;
            if (!firstEvent) firstEvent = static_cast<const otelEventHandle_t*>(p2pHandle);

            totalBytes += op.bytes;
            trackedSendCount++;
            startTs = std::min(startTs, op.startTs);

            double opEnd = (op.seenProxyOps > 0) ? op.lastProxyOpEnd : op.endTs;
            endTs        = std::max(endTs, opEnd);

            totalTransferCount += op.totalTransferCount;
            totalTransferBytes += op.totalTransferBytes;
            totalTransferTimeUs += op.totalTransferTimeUs;
        }

        if (expectedSendCount > trackedSendCount && trackedSendCount > 0)
        {
            const size_t representativeBytes = totalBytes / trackedSendCount;
            totalBytes += (expectedSendCount - trackedSendCount) * representativeBytes;
        }

        if (!firstEvent || !firstEvent->commState) continue;

        double duration = endTs - startTs;
        if (startTs >= endTs || duration <= 0)
        {
            OTEL_WARN(NCCL_INIT, "Skipping AlltoAll collective synthesis with invalid duration=%.2f us", duration);
            continue;
        }

        std::string collKey;
        collKey.reserve(32);
        appendCommPrefix(collKey, firstEvent->commState->comm_hash);
        collKey.append("_AlltoAll_");
        appendInteger(collKey, firstEvent->commState->nranks);
        collKey.append("Ranks");

        AggregatedCollective& collective = collectives[collKey];
        collective.addCollective(totalBytes, duration);
        if (totalTransferCount > 0)
        {
            collective.addTransferBatch(totalTransferCount, totalTransferBytes, totalTransferTimeUs);
        }

        OTEL_TRACE(NCCL_INIT,
                   "Synthesized AlltoAll collective: key=%s, bytes=%zu, duration=%.2f us, transfers=%d, group=%p",
                   collKey.c_str(), totalBytes, duration, totalTransferCount, groupPair.first);
    }
}

/**
 * @brief Finalize aggregation and materialize export-ready operation summaries.
 */
void WindowAggregator::finalize()
{
    OTEL_TRACE(NCCL_INIT, "Finalizing: %zu ProxyOps, %zu ProxyOp transfers", proxyOps.size(), proxyOpTransfers.size());

    std::map<const void*, std::vector<const void*>> alltoAllGroups;
    std::map<const void*, size_t> alltoAllExpectedSendCounts;
    identifyGroupedAlltoAllOperations(alltoAllGroups, alltoAllExpectedSendCounts);

    classifyScaleUpCommunicatorExecutionMode();
    linkProxyOpsToParents();
    finalizeOperationsWithProxyData(collHandleToOp, true);
    finalizeOperationsWithProxyData(p2pHandleToOp, false);
    finalizeScaleUpOperations(collHandleToOp, true);
    finalizeScaleUpOperations(p2pHandleToOp, false);
    reconstructGroupedAlltoAllOperations(alltoAllGroups, alltoAllExpectedSendCounts);
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

/**
 * @brief Determine whether a transfer should use P2P key semantics.
 *
 * @param[in] rootHandle Root collective or P2P handle, if known.
 * @param[in] commState Communicator state used as a fallback when no root handle exists.
 *
 * @return true when the transfer belongs to a P2P operation.
 */
bool WindowAggregator::isP2POperation(const void* rootHandle, const CommunicatorState* commState) const
{
    if (rootHandle)
    {
        if (collHandleToOp.count(rootHandle)) return false;
        if (alltoAllP2PHandles.count(rootHandle)) return false;
        if (p2pHandleToOp.count(rootHandle)) return true;
    }

    return commState && commState->comm_type == CommunicatorState::CommType::P2P;
}

/**
 * @brief Count send-direction P2P API markers associated with a group window.
 *
 * @param[in] groupEvent Group event delimiting the window of interest.
 *
 * @return Number of send-side P2P API markers associated with the group.
 */
size_t WindowAggregator::countGroupSendP2pApis(const otelEventHandle_t& groupEvent) const
{
    double previousGroupEndTs = -std::numeric_limits<double>::infinity();
    for (const otelEventHandle_t* otherGroup : groupEvents)
    {
        if (!otherGroup || otherGroup == &groupEvent) continue;
        if (otherGroup->commState != groupEvent.commState) continue;
        if (otherGroup->endTs > groupEvent.startTs) continue;
        previousGroupEndTs = std::max(previousGroupEndTs, otherGroup->endTs);
    }

    size_t sendApiCount = 0;
    for (const otelEventHandle_t* apiEvent : p2pApiEvents)
    {
        if (!apiEvent || apiEvent->commState != groupEvent.commState) continue;
        if (apiEvent->startTs <= previousGroupEndTs || apiEvent->startTs > groupEvent.startTs) continue;
        if (!apiEvent->p2pApi.func || strstr(apiEvent->p2pApi.func, "Send") == nullptr) continue;
        sendApiCount++;
    }
    return sendApiCount;
}

/**
 * @brief Check whether a grouped set of P2P children matches an AlltoAll pattern.
 *
 * @param[in] groupEvent Group event delimiting the candidate AlltoAll window.
 * @param[in] p2pHandles Candidate P2P child handles within the group window.
 * @param[in] sendApiCount Number of send-side P2P API markers in the group.
 *
 * @return true when the group should be reconstructed as an AlltoAll collective.
 */
bool WindowAggregator::isAlltoAllGroup(const otelEventHandle_t& groupEvent, const std::vector<const void*>& p2pHandles,
                                       size_t sendApiCount) const
{
    if (p2pHandles.empty()) return false;

    const otelEventHandle_t* firstEvent = static_cast<const otelEventHandle_t*>(p2pHandles.front());
    if (!firstEvent || !firstEvent->commState) return false;
    if (firstEvent->commState != groupEvent.commState) return false;

    const int nranks = firstEvent->commState->nranks;
    if (nranks < 2) return false;

    std::set<int> peers;
    std::set<const void*> sendApiParents;
    bool sawSelfSend = false;
    for (const void* p2pHandle : p2pHandles)
    {
        auto opIt = p2pHandleToOp.find(p2pHandle);
        if (opIt == p2pHandleToOp.end()) continue;

        const otelEventHandle_t* p2pEvent = static_cast<const otelEventHandle_t*>(p2pHandle);
        int commRank                      = p2pEvent->commState ? p2pEvent->commState->rank : p2pEvent->rank;

        peers.insert(opIt->second.peer);
        if (p2pEvent->parentObj) sendApiParents.insert(p2pEvent->parentObj);
        if (opIt->second.peer == commRank) sawSelfSend = true;
    }

    if (sawSelfSend && static_cast<int>(peers.size()) == nranks) return true;

    // Runtime AlltoAll windows omit the self-send P2P child because it never spawns
    // proxy work, but the surrounding Group still contains one send-direction P2pApi
    // marker per rank. Use that stable fan-out to recognize the collective shape.
    return !sawSelfSend && static_cast<int>(peers.size()) == nranks - 1 && sendApiParents.size() == p2pHandles.size() &&
           sendApiCount == sendApiParents.size() + 1;
}

/**
 * @brief Build the rank-transfer key for a scale-up inferred transfer.
 *
 * @param[in] commState Communicator state owning the transfer.
 * @param[in] peer Destination peer rank or pipeline.
 * @param[in] isP2P Whether the transfer belongs to a P2P operation.
 *
 * @return Rank-transfer aggregation key.
 */
std::string WindowAggregator::getScaleUpRankTransferKey(const CommunicatorState* commState, int peer, bool isP2P) const
{
    uint64_t commHash = commState ? commState->comm_hash : 0;
    return getRankTransferKey(commHash, peer, commState, isP2P);
}

/**
 * @brief Build the channel-transfer key for a scale-up inferred transfer.
 *
 * @param[in] commState Communicator state owning the transfer.
 * @param[in] peer Destination peer rank or pipeline.
 * @param[in] channelId Logical channel identifier.
 * @param[in] isP2P Whether the transfer belongs to a P2P operation.
 *
 * @return Channel-transfer aggregation key.
 */
std::string WindowAggregator::getScaleUpChannelTransferKey(const CommunicatorState* commState, int peer,
                                                           uint8_t channelId, bool isP2P) const
{
    uint64_t commHash = commState ? commState->comm_hash : 0;

    return buildTransferKey(commHash, commState, rank, peer, isP2P, true, (int)channelId);
}
