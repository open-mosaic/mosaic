// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef AGGREGATION_H_
#define AGGREGATION_H_

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "communicator_state.h"
#include "events.h"
#include "linear_regression.h"
#include "param.h"

// Helper function to get linear regression mode from environment.
// Recognized values: "AVG" → AVG mode. Anything else (including the default
// empty string) → MIN mode. An unrecognized non-empty string emits a warning.
static inline LinearRegression::Mode getLinearRegressionMode()
{
    const char* modeStr = ncclParamLinearRegressionMode();
    if (strcmp(modeStr, "AVG") == 0)
    {
        return LinearRegression::Mode::AVG;
    }
    if (strcmp(modeStr, "MIN") != 0 && strcmp(modeStr, "") != 0)
    {
        OTEL_WARN(NCCL_INIT, "Unknown LinearRegressionMode '%s', defaulting to MIN", modeStr);
    }
    return LinearRegression::Mode::MIN;
}

// Structure for aggregated transfer information
struct AggregatedTransfer
{
    size_t totalBytes;
    double totalTimeUs;
    int count;
    LinearRegression lr;  // For latency calculation via linear regression

    // Transfer intervals for bandwidth calculation based on active transfer time.
    // Each interval represents (startTs, endTs) of a transfer in microseconds.
    // The rate is computed as totalBytes / activeTime where activeTime is the
    // merged (union) of all intervals - representing the time when at least one
    // transfer was in progress between this rank pair.
    std::vector<std::pair<double, double>> intervals;

    AggregatedTransfer() : totalBytes(0), totalTimeUs(0.0), count(0), lr(getLinearRegressionMode()) {}

    void addTransfer(size_t bytes, double timeUs)
    {
        totalBytes += bytes;
        totalTimeUs += timeUs;
        count++;
        lr.addPoint(bytes, timeUs);
    }

    /**
     * @brief Add a transfer with absolute timestamps for interval-based rate calculation.
     *
     * This method extends addTransfer() by also recording the absolute time interval
     * of the transfer for computing bandwidth based on merged active transfer time.
     *
     * @param[in] bytes Transfer size in bytes.
     * @param[in] timeUs Transfer duration in microseconds (endTs - startTs).
     * @param[in] startTs Absolute start timestamp of the transfer (e.g., sendWaitTs).
     * @param[in] endTs Absolute end timestamp of the transfer.
     */
    void addTransferWithTimestamps(size_t bytes, double timeUs, double startTs, double endTs)
    {
        addTransfer(bytes, timeUs);
        if (startTs < endTs)
        {
            intervals.push_back({startTs, endTs});
        }
    }

    /**
     * @brief Merge intervals from another AggregatedTransfer.
     *
     * Appends all intervals from the other transfer to this one.
     * The actual merging (computing union) is done lazily in getActiveTime().
     *
     * @param[in] other Another AggregatedTransfer to merge intervals from.
     */
    void mergeIntervals(const AggregatedTransfer& other)
    {
        intervals.insert(intervals.end(), other.intervals.begin(), other.intervals.end());
    }

    /**
     * @brief Compute the total active transfer time (union of all intervals).
     *
     * Merges overlapping intervals and returns the total duration during which
     * at least one transfer was active. This represents the actual bandwidth
     * utilization window for this rank-to-rank connection.
     *
     * Example:
     *   - Transfer A: t0 to t2
     *   - Transfer B: t1 to t3 (overlaps with A)
     *   - Transfer C: t4 to t5 (gap after B)
     *   Active time = (t3 - t0) + (t5 - t4)
     *
     * @return Total active time in microseconds, or 0 if no intervals.
     */
    double getActiveTime() const
    {
        if (intervals.empty()) return 0.0;

        // Sort intervals by start time
        auto sorted = intervals;
        std::sort(sorted.begin(), sorted.end());

        // Merge overlapping intervals
        double activeTime   = 0.0;
        double currentStart = sorted[0].first;
        double currentEnd   = sorted[0].second;

        for (size_t i = 1; i < sorted.size(); i++)
        {
            if (sorted[i].first <= currentEnd)
            {
                // Overlapping or adjacent - extend current interval
                currentEnd = std::max(currentEnd, sorted[i].second);
            }
            else
            {
                // Gap - add current interval and start new one
                activeTime += currentEnd - currentStart;
                currentStart = sorted[i].first;
                currentEnd   = sorted[i].second;
            }
        }
        activeTime += currentEnd - currentStart;
        return activeTime;
    }

    /**
     * @brief Get bandwidth rate based on active transfer time.
     *
     * Computes the transfer rate as totalBytes / activeTime where activeTime
     * is the merged duration of all transfer intervals. This method assumes
     * that parallel transfers share the available bandwidth, so the rate
     * represents the actual bandwidth utilization between two ranks.
     *
     * @param[out] rateMBps Calculated rate in MB/s (decimal MB convention).
     *
     * @return true if rate was calculated successfully, false if no valid data.
     */
    bool getRateFromActiveTime(double& rateMBps) const
    {
        double activeTime = getActiveTime();
        if (activeTime <= 0.0 || totalBytes == 0)
        {
            rateMBps = 0.0;
            return false;
        }
        // bytes / microseconds = MB/s (decimal MB convention: 1 MB = 1,000,000 bytes)
        // Since 1 byte/us = 1,000,000 bytes/s = 1 MB/s
        rateMBps = (double)totalBytes / activeTime;
        return true;
    }

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

    /**
     * @brief Get latency from linear regression.
     *
     * Linear regression fits: time = intercept + slope * size
     * Where: intercept is latency at size=0 (in microseconds)
     *
     * Requirements:
     * - At least 3 different transfer sizes
     * - Acceptable variance (R-squared >= 0.8)
     *
     * @param[out] latencyUs Calculated latency in microseconds.
     *
     * @return true if latency was calculated successfully, false otherwise.
     */
    bool getLatencyFromLinearRegression(double& latencyUs) const
    {
        // Check if we have at least 3 different transfer sizes
        if (!lr.hasAtLeastThreeDifferentSizes())
        {
            latencyUs = 0.0;
            return false;
        }

        double slope, intercept;
        if (lr.calculate(slope, intercept))
        {
            // Check R-squared for acceptable variance (goodness of fit)
            double rSquared;
            if (lr.calculateRSquared(rSquared) && rSquared >= 0.8)
            {
                latencyUs = (intercept >= 0) ? intercept : 0.0;  // Clamp negative latency to 0

                // Verify slope is positive (time should increase with size)
                if (slope > 1e-6)
                {
                    return true;
                }
            }
        }
        latencyUs = 0.0;
        return false;
    }
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
          peer(-1)
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
    /**
     * @brief Construct a WindowAggregator for a specific rank.
     *
     * @param[in] rank Global rank of the process (used for key generation).
     */
    WindowAggregator(int rank);

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
    void addEvent(const otelEventHandle_t& event);

    /**
     * @brief Finalize aggregation and calculate metrics.
     *
     * Links ProxyOps to their parent Collectives/P2Ps, calculates correct durations
     * (Coll/P2P START → Last ProxyOp STOP), and prepares aggregated data for export.
     * Must be called after all events are added.
     */
    void finalize();

    /**
     * @brief Get aggregated collective operations.
     *
     * @return Map of collective keys to aggregated data.
     *         Key format: Comm<hash>_Func_Algo_Proto_nChannels
     */
    const std::map<std::string, AggregatedCollective>& getCollectives() const
    {
        return collectives;
    }

    /**
     * @brief Get aggregated P2P operations.
     *
     * @return Map of P2P keys to aggregated data.
     *         Key format: Comm<hash>_Func_RankXToRankY_nChannels
     */
    const std::map<std::string, AggregatedP2P>& getP2Ps() const
    {
        return p2ps;
    }

    /**
     * @brief Get aggregated rank-to-rank transfers.
     *
     * @return Map of rank transfer keys to aggregated data.
     *         Key format: Comm<hash>_RankXToRankY
     */
    const std::map<std::string, AggregatedTransfer>& getRankTransfers() const
    {
        return rankTransfers;
    }

    /**
     * @brief Get aggregated per-channel transfers.
     *
     * @return Map of channel transfer keys to aggregated data.
     *         Key format: Comm<hash>_RankXToRankY_Chnl<id>
     */
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
    // AlltoAll collective reconstruction from P2pApi + P2P events
    // -------------------------------------------------------------------------
    // Maps P2pApi event handle -> original collective function name (e.g., "AlltoAll").
    // Populated when a ncclProfileP2pApi event is processed in addEvent().
    std::map<const void*, std::string> p2pApiHandleToFunc;
    // Maps P2pApi event handle -> list of P2P Send event handles that share this parent.
    // Populated when a ncclProfileP2p event has a parentObj that is a P2pApi handle.
    std::map<const void*, std::vector<const void*>> p2pHandlesByApiHandle;

    /**
     * @brief Generate aggregation key for a collective event.
     *
     * @param[in] event Collective event handle.
     *
     * @return Key string in format: Comm<hash>_Func_Algo_Proto_nChannels
     */
    std::string getCollectiveKey(const otelEventHandle_t& event) const;

    /**
     * @brief Generate aggregation key for a P2P event.
     *
     * @param[in] event P2P event handle.
     *
     * @return Key string in format: Comm<hash>_Func_RankXToRankY_nChannels
     */
    std::string getP2PKey(const otelEventHandle_t& event) const;

    /**
     * @brief Generate aggregation key for rank-to-rank transfers.
     *
     * @param[in] commHash Communicator hash.
     * @param[in] peer Destination peer rank.
     *
     * @return Key string in format: Comm<hash>_<hostname>_GPU<local>_ToPeer<peer>
     */
    std::string getRankTransferKey(uint64_t commHash, int peer, const CommunicatorState* commState) const;

    /**
     * @brief Generate aggregation key for per-channel transfers.
     *
     * @param[in] event ProxyOp event handle containing channel and peer info.
     *
     * @return Key string in format: Comm<hash>_RankXToRankY_Chnl<id>
     */
    std::string getChannelTransferKey(const otelEventHandle_t& event) const;

    /**
     * @brief Generate key for transfer channel grouping.
     *
     * @param[in] channelId Channel ID.
     *
     * @return Key string in format: Chnl<id>
     */
    std::string getTransferChannelKey(uint8_t channelId) const;

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
    const void* getRootCollectiveHandle(const void* parentObj) const;

    /**
     * @brief Generate a scale-up rank transfer key with proper source/destination ranks.
     *
     * Reuses the same key format as scale-out (Comm<hash>_Rank<X>_ToPeer<Y>) so the
     * telemetry export parsers can extract source_rank and dest_rank correctly.
     * For ring collectives, the peer is derived as (rank + 1) % nRanks.
     *
     * @param[in] commState Communicator state for the operation.
     * @param[in] peer      Destination peer rank (ring neighbor or P2P peer).
     *
     * @return Key string in the standard rank transfer format.
     */
    std::string getScaleUpRankTransferKey(const CommunicatorState* commState, int peer) const;

    /**
     * @brief Generate a scale-up channel transfer key with proper source/destination ranks.
     *
     * Reuses the same key format as scale-out (Comm<hash>_Rank<X>_ToPeer<Y>_Chnl<id>)
     * so the telemetry export parsers can extract labels correctly.
     *
     * @param[in] commState Communicator state.
     * @param[in] peer      Destination peer rank.
     * @param[in] channelId Channel ID.
     *
     * @return Key string in the standard channel transfer format.
     */
    std::string getScaleUpChannelTransferKey(const CommunicatorState* commState, int peer, uint8_t channelId) const;

    /**
     * @brief Finalize scale-up operations that have no ProxyOps.
     *
     * For Coll/P2P operations where seenProxyOps == 0, uses KernelCh events
     * and transfer inference to derive timing and transfer metrics.
     *
     * @param[in] handleToOp Map of operation handles to in-progress operations.
     * @param[in] isColl true for collective operations, false for P2P.
     */
    void finalizeScaleUpOperations(std::map<const void*, InProgressOperation>& handleToOp, bool isColl);
};

#endif  // AGGREGATION_H_
