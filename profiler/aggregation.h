// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef AGGREGATION_H_
#define AGGREGATION_H_

#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "communicator_state.h"
#include "events.h"
#include "linear_regression.h"

struct AggregatedTransfer
{
    size_t totalBytes;
    double totalTimeUs;
    int count;
    LinearRegression lr;  // For latency calculation via linear regression
    std::vector<std::pair<double, double>> intervals;

    AggregatedTransfer();

    void addTransfer(size_t bytes, double timeUs)
    {
        totalBytes += bytes;
        totalTimeUs += timeUs;
        count++;
        lr.addPoint(bytes, timeUs);
    }

    void addTransferWithTimestamps(size_t bytes, double timeUs, double startTs, double endTs);

    void mergeIntervals(const AggregatedTransfer& other);

    double getActiveTime() const;

    bool getRateFromActiveTime(double& rateMBps) const;

    double getAverageSize() const
    {
        return count > 0 ? (double)totalBytes / count : 0.0;
    }
    double getAverageTime() const
    {
        return count > 0 ? totalTimeUs / count : 0.0;
    }
    // Returns totalBytes / totalTimeUs, where totalTimeUs is the *sum* of individual
    // per-transfer times. When transfers run in parallel across channels this sum
    // overcounts wall-clock time, so the result underestimates true bandwidth.
    // Prefer getRateFromActiveTime() for accurate bandwidth when channels overlap.
    double getAverageRateMBps() const
    {
        return totalTimeUs > 0 ? (double)totalBytes / totalTimeUs : 0.0;
    }

    bool getLatencyFromLinearRegression(double& latencyUs) const;
};

// Base structure for aggregated operations (Collective or P2P)
// Contains common fields and methods shared by both operation types
struct AggregatedOperationBase
{
    size_t totalBytes;
    double totalTimeUs;
    int count;

    // Cached aggregates for underlying transfers (proxy ops or scale-up inferred)
    int cachedTotalTransferCount;
    size_t cachedTotalTransferBytes;
    double cachedTotalTransferTimeUs;

    AggregatedOperationBase()
        : totalBytes(0),
          totalTimeUs(0.0),
          count(0),
          cachedTotalTransferCount(0),
          cachedTotalTransferBytes(0),
          cachedTotalTransferTimeUs(0.0)
    {
    }

    // Add operation data (bytes and time)
    void addOperation(size_t bytes, double timeUs)
    {
        totalBytes += bytes;
        totalTimeUs += timeUs;
        count++;
    }

    // Record a single transfer into the aggregate cache. Used by scale-up inference
    // which emits transfers one at a time.
    void addTransferToCache(size_t bytes, double timeUs)
    {
        cachedTotalTransferCount++;
        cachedTotalTransferBytes += bytes;
        cachedTotalTransferTimeUs += timeUs;
    }

    // Record a batch of transfers into the aggregate cache without looping.
    // Used by the proxy-op path where the per-transfer totals are already summed.
    void addTransferBatch(int transferCount, size_t batchBytes, double batchTimeUs)
    {
        cachedTotalTransferCount += transferCount;
        cachedTotalTransferBytes += batchBytes;
        cachedTotalTransferTimeUs += batchTimeUs;
    }

    // Get average bytes per operation
    double getAverageSize() const
    {
        return count > 0 ? (double)totalBytes / count : 0.0;
    }

    // Get average time per operation
    double getAverageTime() const
    {
        return count > 0 ? totalTimeUs / count : 0.0;
    }

    // Get aggregate transfer statistics from underlying proxy operations - O(1)
    int getTotalTransferCount() const
    {
        return cachedTotalTransferCount;
    }

    double getAverageTransferCount() const
    {
        return count > 0 ? (double)cachedTotalTransferCount / count : 0.0;
    }

    double getAverageTransferSize() const
    {
        return cachedTotalTransferCount > 0 ? (double)cachedTotalTransferBytes / cachedTotalTransferCount : 0.0;
    }

    double getAverageTransferTime() const
    {
        return cachedTotalTransferCount > 0 ? cachedTotalTransferTimeUs / cachedTotalTransferCount : 0.0;
    }
};

// Structure for aggregated P2P information
struct AggregatedP2P : public AggregatedOperationBase
{
    // Inherits all fields and methods from AggregatedOperationBase

    // Convenience method for P2P-specific naming (delegates to base class)
    void addP2P(size_t bytes, double timeUs)
    {
        addOperation(bytes, timeUs);
    }
};

// Structure for aggregated Collective information
struct AggregatedCollective : public AggregatedOperationBase
{
    // Inherits all fields and methods from AggregatedOperationBase

    // Convenience method for collective-specific naming (delegates to base class)
    void addCollective(size_t bytes, double timeUs)
    {
        addOperation(bytes, timeUs);
    }
};

// Structure to track in-progress Collective/P2P for correct timing
struct InProgressOperation
{
    std::string key;        // Aggregate key (Func_Algo_Proto_nChannels)
    double startTs;         // When Coll/P2P started
    double endTs;           // When Coll/P2P stopped (fallback for internal links scenario)
    double lastProxyOpEnd;  // Latest ProxyOp end time (for duration calculation)
    size_t bytes;           // Operation bytes
    int seenProxyOps;       // Count of ProxyOps seen so far

    // Scale-up fields (from Coll/P2P descriptor, used when no ProxyOps exist)
    const char* func;   // Collective function name (e.g. "AllReduce")
    const char* algo;   // Algorithm name (e.g. "Ring", "Tree")
    uint8_t nChannels;  // Number of channels
    int nRanks;         // Number of ranks in communicator
    int peer;           // Peer rank (for P2P: from descriptor, for Coll: derived from ring)
    int totalTransferCount;
    size_t totalTransferBytes;
    double totalTransferTimeUs;

    InProgressOperation()
        : startTs(0),
          endTs(0),
          lastProxyOpEnd(0),
          bytes(0),
          seenProxyOps(0),
          func(nullptr),
          algo(nullptr),
          nChannels(0),
          nRanks(0),
          peer(-1),
          totalTransferCount(0),
          totalTransferBytes(0),
          totalTransferTimeUs(0.0)
    {
    }
};

/**
 * @brief Main aggregator for processing a single window of events.
 *
 * Aggregates NCCL events from a window, links ProxyOps to their parent Collectives/P2Ps,
 * and calculates metrics. Processes events in phases:
 * 1. Track Coll/P2P operations
 * 2. Aggregate ProxyStep transfers
 * 3. Link ProxyOps to parents and calculate durations
 * 4. Export-ready aggregated data
 *
 * @note This class is NOT thread-safe and is designed for single-threaded use
 *       by the telemetry thread. Each window is processed by one thread at a time.
 */
class WindowAggregator
{
public:
    WindowAggregator(int rank);

    void addEvent(const otelEventHandle_t& event);

    void finalize();

    const std::map<std::string, AggregatedCollective>& getCollectives() const
    {
        return collectives;
    }

    const std::map<std::string, AggregatedP2P>& getP2Ps() const
    {
        return p2ps;
    }

    const std::map<std::string, AggregatedTransfer>& getRankTransfers() const
    {
        return rankTransfers;
    }

    const std::map<std::string, AggregatedTransfer>& getChannelTransfers() const
    {
        return channelTransfers;
    }

private:
    int rank;
    std::map<std::string, AggregatedCollective> collectives;     // Key: Comm<hash>_Func_Algo_Proto_nChannels
    std::map<std::string, AggregatedP2P> p2ps;                   // Key: Comm<hash>_Func_RankXToRankY_nChannels
    std::map<std::string, AggregatedTransfer> rankTransfers;     // Key: Comm<hash>_RankXToRankY
    std::map<std::string, AggregatedTransfer> channelTransfers;  // Key: Comm<hash>_RankXToRankY_Chnl<id>

    // Maps to track eHandle -> operation relationship
    std::map<const void*, InProgressOperation> collHandleToOp;  // Coll eHandle -> in-progress operation
    std::map<const void*, InProgressOperation> p2pHandleToOp;   // P2P eHandle -> in-progress operation

    // Map ProxyOp eHandle to its aggregated transfer data (from ProxySteps)
    std::map<const void*, AggregatedTransfer> proxyOpTransfers;  // ProxyOp eHandle -> aggregated ProxyStep transfers

    // Store ProxyOp events for linking in finalize() (after ProxySteps are aggregated)
    std::map<const void*, otelEventHandle_t> proxyOps;  // ProxyOp eHandle -> ProxyOp event

    // KernelCh events grouped by their parent Coll/P2P handle (for scale-up analysis)
    std::map<const void*, std::vector<otelEventHandle_t>> kernelChByParent;

    // KernelLaunch events (informational)
    std::vector<otelEventHandle_t> kernelLaunches;

    // -------------------------------------------------------------------------
    // Group-scoped AlltoAll reconstruction
    // -------------------------------------------------------------------------
    // Keep completed Group events and rebuild membership during finalize().
    // Keep the accompanying P2pApi markers as well: runtime AlltoAll traces
    // record the self-send API marker even though the self-send P2P child is not
    // tracked in the window because it never spawns proxy work.
    std::vector<const otelEventHandle_t*> groupEvents;
    std::vector<const otelEventHandle_t*> p2pApiEvents;
    std::set<const void*> alltoAllP2PHandles;

    std::string getCollectiveKey(const otelEventHandle_t& event) const;

    std::string getP2PKey(const otelEventHandle_t& event) const;

    std::string getRankTransferKey(uint64_t commHash, int peer, const CommunicatorState* commState, bool isP2P) const;

    std::string getChannelTransferKey(const otelEventHandle_t& event, bool isP2P) const;

    std::string getTransferChannelKey(uint8_t channelId) const;

    void trackCollectiveEvent(const otelEventHandle_t& event);

    void trackP2PEvent(const otelEventHandle_t& event);

    void trackP2pApiEvent(const otelEventHandle_t& event);

    void trackGroupEvent(const otelEventHandle_t& event);

    void accumulateProxyStepTransfer(const otelEventHandle_t& event);

    void storeProxyOpForFinalize(const otelEventHandle_t& event);

    void trackKernelChannelEvent(const otelEventHandle_t& event);

    void trackKernelLaunchEvent(const otelEventHandle_t& event);

    const void* getRootCollectiveHandle(const void* parentObj) const;

    std::string getScaleUpRankTransferKey(const CommunicatorState* commState, int peer, bool isP2P) const;

    std::string getScaleUpChannelTransferKey(const CommunicatorState* commState, int peer, uint8_t channelId,
                                             bool isP2P) const;

    bool isP2POperation(const void* rootHandle, const CommunicatorState* commState) const;

    size_t countGroupSendP2pApis(const otelEventHandle_t& groupEvent) const;
    bool isAlltoAllGroup(const otelEventHandle_t& groupEvent, const std::vector<const void*>& p2pHandles,
                         size_t sendApiCount) const;

    void identifyGroupedAlltoAllOperations(std::map<const void*, std::vector<const void*>>& alltoAllGroups,
                                           std::map<const void*, size_t>& alltoAllExpectedSendCounts);

    void classifyScaleUpCommunicatorExecutionMode();

    void linkProxyOpsToParents();

    void finalizeOperationsWithProxyData(std::map<const void*, InProgressOperation>& handleToOp, bool isColl);

    void reconstructGroupedAlltoAllOperations(const std::map<const void*, std::vector<const void*>>& alltoAllGroups,
                                              const std::map<const void*, size_t>& alltoAllExpectedSendCounts);

    void finalizeScaleUpOperations(std::map<const void*, InProgressOperation>& handleToOp, bool isColl);
};

#endif  // AGGREGATION_H_
