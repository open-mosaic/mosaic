// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "telemetry_primer.h"

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "profiler_otel.h"

#ifdef ENABLE_OTEL

#include <opentelemetry/context/context.h>
#include <opentelemetry/metrics/sync_instruments.h>

#include "telemetry_internal.h"

static void exportCollectiveMetricsPrimer(const std::string& key, const AggregatedCollective& coll, int rank,
                                          const std::string& hostname, int local_rank, uint64_t comm_hash,
                                          const std::string& gpu_pci_bus_id, const std::string& gpu_uuid,
                                          const std::string& comm_type, int nranks,
                                          const std::string& scale_up_exec_mode);
static void exportP2PMetricsPrimer(const std::string& key, const AggregatedP2P& p2p, int rank,
                                   const std::string& hostname, int local_rank, uint64_t comm_hash,
                                   const std::string& gpu_pci_bus_id, const std::string& gpu_uuid,
                                   const std::string& comm_type, int nranks, const std::string& scale_up_exec_mode);
static void exportRankMetricsPrimer(const std::string& key, const AggregatedTransfer& transferRef, int rank,
                                    const std::string& hostname, const std::string& gpu_pci_bus_id,
                                    const std::string& gpu_uuid, const std::string& comm_type, int nranks,
                                    int local_rank, const std::string& scale_up_exec_mode);
static void exportTransferMetricsPrimer(const std::string& key, const AggregatedTransfer& transferRef, int rank,
                                        const std::string& hostname, const std::string& gpu_pci_bus_id,
                                        const std::string& gpu_uuid, const std::string& comm_type, int nranks,
                                        int local_rank, const std::string& scale_up_exec_mode);

// =======================================================================================
// Global Primer State Storage
// =======================================================================================

template <typename T>
using PrimerBucket = std::map<std::string, PrimerData<T>>;

template <typename T>
using PrimerStore = std::map<PrimerScopeKey, PrimerBucket<T>>;

using PrimerDoneBucket = std::unordered_set<std::string>;
using PrimerDoneStore  = std::unordered_map<PrimerScopeKey, PrimerDoneBucket, PrimerScopeKeyHash>;

// Primer storage for each metric type
static PrimerStore<AggregatedCollective> g_collectivePrimers;
static PrimerStore<AggregatedP2P> g_p2pPrimers;
static PrimerStore<AggregatedTransfer> g_rankPrimers;
static PrimerStore<AggregatedTransfer> g_transferPrimers;

// Track keys that have completed the primer cycle (to avoid re-priming on subsequent windows)
static PrimerDoneStore g_collectivePrimersDone;
static PrimerDoneStore g_p2pPrimersDone;
static PrimerDoneStore g_rankPrimersDone;
static PrimerDoneStore g_transferPrimersDone;

// =======================================================================================
// Helper Functions
// =======================================================================================

/**
 * @brief Merge two AggregatedCollective structures.
 *
 * Summ up the metrics of one window with another window to keep the history of the metrics.
 * This is used to make sure the first STANDARD exported Collective will contain all the metrics
 * from the previous windows.
 * @param[in] a The first AggregatedCollective to merge.
 * @param[in] b The second AggregatedCollective to merge.
 * @return The merged AggregatedCollective.
 */
static AggregatedCollective mergeAggregatedCollective(const AggregatedCollective& a, const AggregatedCollective& b)
{
    AggregatedCollective merged;
    merged.totalBytes                = a.totalBytes + b.totalBytes;
    merged.totalTimeUs               = a.totalTimeUs + b.totalTimeUs;
    merged.count                     = a.count + b.count;
    merged.cachedTotalTransferCount  = a.cachedTotalTransferCount + b.cachedTotalTransferCount;
    merged.cachedTotalTransferBytes  = a.cachedTotalTransferBytes + b.cachedTotalTransferBytes;
    merged.cachedTotalTransferTimeUs = a.cachedTotalTransferTimeUs + b.cachedTotalTransferTimeUs;
    return merged;
}

/**
 * @brief Merge two AggregatedP2P structures.
 *
 * Summ up the metrics of one window with another window to keep the history of the metrics.
 * This is used to make sure the first STANDARD exported P2P will contain all the metrics
 * from the previous windows.
 * @param[in] a The first AggregatedP2P to merge.
 * @param[in] b The second AggregatedP2P to merge.
 * @return The merged AggregatedP2P.
 */
static AggregatedP2P mergeAggregatedP2P(const AggregatedP2P& a, const AggregatedP2P& b)
{
    AggregatedP2P merged;
    merged.totalBytes                = a.totalBytes + b.totalBytes;
    merged.totalTimeUs               = a.totalTimeUs + b.totalTimeUs;
    merged.count                     = a.count + b.count;
    merged.cachedTotalTransferCount  = a.cachedTotalTransferCount + b.cachedTotalTransferCount;
    merged.cachedTotalTransferBytes  = a.cachedTotalTransferBytes + b.cachedTotalTransferBytes;
    merged.cachedTotalTransferTimeUs = a.cachedTotalTransferTimeUs + b.cachedTotalTransferTimeUs;
    return merged;
}

/**
 * @brief Merge two AggregatedTransfer structures.
 *
 * Summ up the metrics of one window with another window to keep the history of the metrics.
 * This is used to make sure the first STANDARD exported Transfer will contain all the metrics
 * from the previous windows.
 * @param[in] a The first AggregatedTransfer to merge.
 * @param[in] b The second AggregatedTransfer to merge.
 * @return The merged AggregatedTransfer.
 */
static AggregatedTransfer mergeAggregatedTransfer(const AggregatedTransfer& a, const AggregatedTransfer& b)
{
    AggregatedTransfer merged;
    merged.totalBytes  = a.totalBytes + b.totalBytes;
    merged.totalTimeUs = a.totalTimeUs + b.totalTimeUs;
    merged.count       = a.count + b.count;

    // Merge linear regression data using the built-in merge method
    merged.lr = a.lr;
    merged.lr.merge(b.lr);

    // Merge transfer intervals
    merged.intervals = a.intervals;
    merged.intervals.insert(merged.intervals.end(), b.intervals.begin(), b.intervals.end());

    return merged;
}

/**
 * @brief Check if scale_up_exec_mode is known (not UNKNOWN).
 *
 * @param[in] commState Communicator state containing the window to process.
 * @return true if the scale_up_exec_mode is known, false otherwise.
 */
static bool isScaleUpExecModeKnown(CommunicatorState* commState)
{
    CommunicatorState::ScaleUpExecMode mode =
        static_cast<CommunicatorState::ScaleUpExecMode>(commState->scaleUpExecMode.load(std::memory_order_acquire));
    return mode != CommunicatorState::ScaleUpExecMode::UNKNOWN;
}

/**
 * @brief Build the stable communicator identity used by primer bookkeeping.
 *
 * @param[in] commState Communicator state owning the primer data.
 *
 * @return Stable communicator identity for primer storage.
 */
static PrimerScopeKey makePrimerScopeKey(const CommunicatorState* commState)
{
    return PrimerScopeKey{commState ? commState->comm_hash : 0, commState ? commState->rank : -1};
}

/**
 * @brief Build a short debug description for aggregated collective primer data.
 *
 * @param[in] data Aggregated collective data.
 *
 * @return Short debug string describing the primer payload.
 */
static std::string describePrimerData(const AggregatedCollective& data)
{
    return "count=" + std::to_string(data.count) + ", bytes=" + std::to_string(data.totalBytes);
}

/**
 * @brief Build a short debug description for aggregated P2P primer data.
 *
 * @param[in] data Aggregated P2P data.
 *
 * @return Short debug string describing the primer payload.
 */
static std::string describePrimerData(const AggregatedP2P& data)
{
    return "count=" + std::to_string(data.count);
}

/**
 * @brief Build a short debug description for aggregated transfer primer data.
 *
 * @param[in] data Aggregated transfer data.
 *
 * @return Short debug string describing the primer payload.
 */
static std::string describePrimerData(const AggregatedTransfer& data)
{
    return "count=" + std::to_string(data.count);
}

/**
 * @brief Check whether a primer key has already completed its primer cycle.
 *
 * @param[in] commState Communicator state owning the primer key.
 * @param[in] key Aggregation key to check.
 * @param[in] donePrimers Set of completed primer keys for the metric family.
 *
 * @return true when the key has already completed primer emission and real-data export.
 */
static bool isPrimerDone(CommunicatorState* commState, const std::string& key, const PrimerDoneStore& donePrimers)
{
    const PrimerScopeKey scope = makePrimerScopeKey(commState);
    auto doneIt                = donePrimers.find(scope);
    return doneIt != donePrimers.end() && doneIt->second.count(key) > 0;
}

/**
 * @brief Register a new primer key and seed its accumulated data.
 *
 * @param[in] commState Communicator state owning the primer key.
 * @param[in] key Aggregation key to register.
 * @param[in] data First aggregated payload for the key.
 * @param[in,out] primers Primer storage for the metric family.
 * @param[in] metricName Metric-family name used in logs.
 */
template <typename T>
static void registerPrimer(CommunicatorState* commState, const std::string& key, const T& data, PrimerStore<T>& primers,
                           const char* metricName)
{
    PrimerData<T>& primerData = primers[makePrimerScopeKey(commState)][key];
    primerData.aggregatedData = data;
    primerData.state          = PrimerState::PENDING_PRIMER;
    primerData.windowsWaited  = 0;

    const std::string summary = describePrimerData(data);
    if (!isScaleUpExecModeKnown(commState))
    {
        OTEL_INFO(NCCL_INIT, "%s NEW KEY: %s (scale_up_exec_mode UNKNOWN, waiting: %s)", metricName, key.c_str(),
                  summary.c_str());
        return;
    }

    std::string scaleUpMode = commState->getScaleUpExecModeString();
    OTEL_INFO(NCCL_INIT, "%s NEW KEY: %s (scale_up_exec_mode=%s, starting %u-window stabilization: %s)", metricName,
              key.c_str(), scaleUpMode.c_str(), PRIMER_STABILIZATION_WINDOWS, summary.c_str());
}

/**
 * @brief Advance pending primers for one metric family through the primer state machine.
 *
 * @param[in] commState Communicator state owning the current window.
 * @param[in] currentWindowData Aggregated metrics observed in the current window.
 * @param[in,out] primers Pending primer storage for the metric family.
 * @param[in,out] donePrimers Completed primer keys for the metric family.
 * @param[in] mergeFn Function used to merge accumulated payloads.
 * @param[in] exportPrimerFn Function used to emit zero-valued primer metrics.
 * @param[in] exportStandardFn Function used to emit the first real metrics after primer emission.
 * @param[in] metricName Metric-family name used in logs.
 *
 * @return Keys from the current window that were consumed by pending primers.
 */
template <typename T, typename MergeFn, typename ExportPrimerFn, typename ExportStandardFn>
static std::set<std::string> processPendingPrimers(CommunicatorState* commState,
                                                   const std::map<std::string, T>& currentWindowData,
                                                   PrimerStore<T>& primers, PrimerDoneStore& donePrimers,
                                                   MergeFn mergeFn, ExportPrimerFn exportPrimerFn,
                                                   ExportStandardFn exportStandardFn, const char* metricName)
{
    const bool scaleUpModeKnown = isScaleUpExecModeKnown(commState);
    std::set<std::string> handledKeys;
    const PrimerScopeKey scope = makePrimerScopeKey(commState);

    auto primerScopeIt = primers.find(scope);
    if (primerScopeIt == primers.end())
    {
        return handledKeys;
    }

    PrimerBucket<T>& scopePrimers = primerScopeIt->second;
    PrimerDoneBucket& scopeDone   = donePrimers[scope];

    for (auto it = scopePrimers.begin(); it != scopePrimers.end();)
    {
        const std::string& key    = it->first;
        PrimerData<T>& primerData = it->second;

        auto currentIt = currentWindowData.find(key);
        if (currentIt != currentWindowData.end())
        {
            primerData.aggregatedData = mergeFn(primerData.aggregatedData, currentIt->second);
            handledKeys.insert(key);
        }

        const std::string updatedSummary = describePrimerData(primerData.aggregatedData);

        if (primerData.state == PrimerState::PENDING_PRIMER)
        {
            if (primerData.windowsWaited >= PRIMER_MAX_WAIT_WINDOWS)
            {
                std::string scaleUpMode = scaleUpModeKnown ? commState->getScaleUpExecModeString() : "unknown";
                exportPrimerFn(key, primerData.aggregatedData, scaleUpMode);
                primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                OTEL_INFO(NCCL_INIT,
                          "%s PRIMER FORCE-EMITTED: %s (max wait of %u windows exceeded, scale_up_exec_mode=%s, %s)",
                          metricName, key.c_str(), PRIMER_MAX_WAIT_WINDOWS, scaleUpMode.c_str(),
                          updatedSummary.c_str());
                ++it;
            }
            else if (!scaleUpModeKnown)
            {
                primerData.windowsWaited++;
                OTEL_TRACE(NCCL_INIT,
                           "%s PRIMER DELAYED: %s (scale_up_exec_mode still UNKNOWN, waited %u/%u windows, "
                           "accumulating: %s)",
                           metricName, key.c_str(), primerData.windowsWaited, PRIMER_MAX_WAIT_WINDOWS,
                           updatedSummary.c_str());
                ++it;
            }
            else
            {
                std::string scaleUpMode = commState->getScaleUpExecModeString();
                if (scaleUpMode == "cuda_graph")
                {
                    exportPrimerFn(key, primerData.aggregatedData, scaleUpMode);
                    primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                    OTEL_TRACE(NCCL_INIT,
                               "%s PRIMER EMITTED: %s (zeros sent immediately with stable scale_up_exec_mode=%s, "
                               "real data on next window: %s)",
                               metricName, key.c_str(), scaleUpMode.c_str(), updatedSummary.c_str());
                    ++it;
                }
                else if (primerData.windowsWaited < PRIMER_STABILIZATION_WINDOWS)
                {
                    primerData.windowsWaited++;
                    OTEL_TRACE(NCCL_INIT,
                               "%s PRIMER STABILIZING: %s (scale_up_exec_mode=%s, waited %u/%u windows, "
                               "accumulating: %s)",
                               metricName, key.c_str(), scaleUpMode.c_str(), primerData.windowsWaited,
                               PRIMER_STABILIZATION_WINDOWS, updatedSummary.c_str());
                    ++it;
                }
                else
                {
                    exportPrimerFn(key, primerData.aggregatedData, scaleUpMode);
                    primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                    OTEL_TRACE(NCCL_INIT,
                               "%s PRIMER EMITTED: %s (zeros sent with stable scale_up_exec_mode=%s after %u "
                               "windows, real data on next window: %s)",
                               metricName, key.c_str(), scaleUpMode.c_str(), primerData.windowsWaited,
                               updatedSummary.c_str());
                    ++it;
                }
            }
        }
        else if (primerData.state == PrimerState::PRIMER_EMITTED_AWAITING_REAL)
        {
            std::string scaleUpMode = commState->getScaleUpExecModeString();
            exportStandardFn(key, primerData.aggregatedData, scaleUpMode);
            OTEL_TRACE(NCCL_INIT, "%s REAL DATA EXPORTED: %s (primer complete with scale_up_exec_mode=%s: %s)",
                       metricName, key.c_str(), scaleUpMode.c_str(), updatedSummary.c_str());
            scopeDone.insert(key);
            it = scopePrimers.erase(it);
        }
        else
        {
            ++it;
        }
    }

    if (scopePrimers.empty())
    {
        primers.erase(primerScopeIt);
    }

    if (scopeDone.empty())
    {
        donePrimers.erase(scope);
    }

    return handledKeys;
}

/**
 * @brief Remove pending primer payloads for one communicator from a primer store.
 *
 * @tparam T Aggregated metric type stored in the primer map.
 * @param[in] commState Communicator whose pending primer state should be removed.
 * @param[in,out] primers Primer store to clean.
 */
template <typename T>
static void cleanupPrimerStoreForCommunicator(CommunicatorState* commState, PrimerStore<T>& primers)
{
    if (!commState)
    {
        return;
    }

    primers.erase(makePrimerScopeKey(commState));
}

/**
 * @brief Remove completed-primer bookkeeping for one communicator.
 *
 * @param[in] commState Communicator whose completed-primer state should be removed.
 * @param[in,out] donePrimers Completed-primer store to clean.
 */
static void cleanupDonePrimerStoreForCommunicator(CommunicatorState* commState, PrimerDoneStore& donePrimers)
{
    if (!commState)
    {
        return;
    }

    donePrimers.erase(makePrimerScopeKey(commState));
}

/**
 * @brief Process pending Collective primers and go through the primer state machine
 *
 * Process pending Collective primers from previous windows and increase metrics values of the pending operations
 * by one window's worth of metrics. The metrics values are increased by merging the aggregated data of the
 * pending operations with the aggregated data of the current window.
 * Export the Collective PRIMER if cuda_graph scale_up_exec_mode is detected and stable.
 * Export STANDARD metrics on the next window following the emission of the Collective PRIMER.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] collectives Map of aggregated collective data keyed by operation name.
 * @return Set of keys that have been handled and are no longer pending.
 */
std::set<std::string> processPendingCollectivePrimers(CommunicatorState* commState,
                                                      const std::map<std::string, AggregatedCollective>& collectives)
{
    return processPendingPrimers(
        commState, collectives, g_collectivePrimers, g_collectivePrimersDone, mergeAggregatedCollective,
        [&](const std::string& key, const AggregatedCollective& data, const std::string& scaleUpMode)
        {
            exportCollectiveMetricsPrimer(key, data, commState->rank, commState->hostname, commState->local_rank,
                                          commState->comm_hash, commState->gpu_pci_bus_id, commState->gpu_uuid,
                                          commState->getCommTypeString(), commState->nranks, scaleUpMode);
        },
        [&](const std::string& key, const AggregatedCollective& data, const std::string& scaleUpMode)
        {
            CollectiveExportEligibility eligibility = computeCollectiveEligibility(data);
            CollectiveEmitView emit                 = makeStandardCollectiveEmitView(data);
            exportCollectiveMetrics(key, emit, eligibility, commState->rank, commState->hostname, commState->local_rank,
                                    commState->comm_hash, commState->gpu_pci_bus_id, commState->gpu_uuid,
                                    commState->getCommTypeString(), commState->nranks, scaleUpMode, "STANDARD");
        },
        "Collective");
}

/**
 * @brief Process pending P2P primers and go through the primer state machine
 *
 * Process pending P2P primers from previous windows and increase metrics values of the pending operations
 * by one window's worth of metrics. The metrics values are increased by merging the aggregated data of the
 * pending operations with the aggregated data of the current window.
 * Export the P2P PRIMER if cuda_graph scale_up_exec_mode is detected and stable.
 * Export STANDARD metrics on the next window following the emission of the P2P PRIMER.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] p2ps Map of aggregated P2P data keyed by operation name.
 * @return Set of keys that have been handled and are no longer pending.
 */
std::set<std::string> processPendingP2PPrimers(CommunicatorState* commState,
                                               const std::map<std::string, AggregatedP2P>& p2ps)
{
    return processPendingPrimers(
        commState, p2ps, g_p2pPrimers, g_p2pPrimersDone, mergeAggregatedP2P,
        [&](const std::string& key, const AggregatedP2P& data, const std::string& scaleUpMode)
        {
            exportP2PMetricsPrimer(key, data, commState->rank, commState->hostname, commState->local_rank,
                                   commState->comm_hash, commState->gpu_pci_bus_id, commState->gpu_uuid,
                                   commState->getCommTypeString(), commState->nranks, scaleUpMode);
        },
        [&](const std::string& key, const AggregatedP2P& data, const std::string& scaleUpMode)
        {
            P2PExportEligibility eligibility = computeP2PEligibility(data);
            P2PEmitView emit                 = makeStandardP2PEmitView(data);
            exportP2PMetrics(key, emit, eligibility, commState->rank, commState->hostname, commState->local_rank,
                             commState->comm_hash, commState->gpu_pci_bus_id, commState->gpu_uuid,
                             commState->getCommTypeString(), commState->nranks, scaleUpMode, "STANDARD");
        },
        "P2P");
}

/**
 * @brief Process pending Rank transfer primers and go through the primer state machine
 *
 * Process pending Rank transfer primers from previous windows and increase metrics values of the pending operations
 * by one window's worth of metrics. The metrics values are increased by merging the aggregated data of the
 * pending operations with the aggregated data of the current window.
 * Export the Rank transfer PRIMER if cuda_graph scale_up_exec_mode is detected and stable.
 * Export STANDARD metrics on the next window following the emission of the Rank transfer PRIMER.

 * @param[in] commState Communicator state containing the window to process.
 * @param[in] rankTransfers Map of aggregated rank transfer data keyed by operation name.
 * @return Set of keys that have been handled and are no longer pending.
 */
std::set<std::string> processPendingRankPrimers(CommunicatorState* commState,
                                                const std::map<std::string, AggregatedTransfer>& rankTransfers)
{
    return processPendingPrimers(
        commState, rankTransfers, g_rankPrimers, g_rankPrimersDone, mergeAggregatedTransfer,
        [&](const std::string& key, const AggregatedTransfer& data, const std::string& scaleUpMode)
        {
            exportRankMetricsPrimer(key, data, commState->rank, commState->hostname, commState->gpu_pci_bus_id,
                                    commState->gpu_uuid, commState->getCommTypeString(), commState->nranks,
                                    commState->local_rank, scaleUpMode);
        },
        [&](const std::string& key, const AggregatedTransfer& data, const std::string& scaleUpMode)
        {
            RankExportEligibility eligibility = computeRankEligibility(data);
            RankEmitView emit                 = makeStandardRankEmitView(data);
            exportRankMetrics(key, emit, eligibility, commState->rank, commState->hostname, commState->gpu_pci_bus_id,
                              commState->gpu_uuid, commState->getCommTypeString(), commState->nranks,
                              commState->local_rank, scaleUpMode, "STANDARD");
        },
        "Rank");
}

/**
 * @brief Process pending Channel transfer primers and go through the primer state machine
 *
 * Process pending Channel transfer primers from previous windows and increase metrics values of the pending operations
 * by one window's worth of metrics. The metrics values are increased by merging the aggregated data of the
 * pending operations with the aggregated data of the current window.
 * Export the Channel transfer PRIMER if cuda_graph scale_up_exec_mode is detected and stable. Export STANDARD metrics
 * message on the next window following the emission of the Channel transfer PRIMER.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] channelTransfers Map of aggregated channel transfer data keyed by operation name.
 * @return Set of keys that have been handled and are no longer pending.
 */
std::set<std::string> processPendingTransferPrimers(CommunicatorState* commState,
                                                    const std::map<std::string, AggregatedTransfer>& channelTransfers)
{
    return processPendingPrimers(
        commState, channelTransfers, g_transferPrimers, g_transferPrimersDone, mergeAggregatedTransfer,
        [&](const std::string& key, const AggregatedTransfer& data, const std::string& scaleUpMode)
        {
            exportTransferMetricsPrimer(key, data, commState->rank, commState->hostname, commState->gpu_pci_bus_id,
                                        commState->gpu_uuid, commState->getCommTypeString(), commState->nranks,
                                        commState->local_rank, scaleUpMode);
        },
        [&](const std::string& key, const AggregatedTransfer& data, const std::string& scaleUpMode)
        {
            TransferExportEligibility eligibility = computeTransferEligibility(data);
            TransferEmitView emit                 = makeStandardTransferEmitView(data);
            exportTransferMetrics(key, emit, eligibility, commState->rank, commState->hostname,
                                  commState->gpu_pci_bus_id, commState->gpu_uuid, commState->getCommTypeString(),
                                  commState->nranks, commState->local_rank, scaleUpMode, "STANDARD");
        },
        "Transfer");
}

// =======================================================================================
// Helper functions used by the main telemetry code to check PRIMER status for a specific
// key and to inform the PRIMER engine that a new key has been detected and will need to
// be processed in the next window.
// =======================================================================================

/**
 * @brief Helper function which checks if a collective key has completed its primer cycle.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] key The key of the collective operation to check.
 * @return true if the key's primer was already emitted and real data exported
 */
bool isCollectivePrimerDone(CommunicatorState* commState, const std::string& key)
{
    return isPrimerDone(commState, key, g_collectivePrimersDone);
}

/**
 * @brief Helper function which registers a new collective key for primer processing.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] key The key of the collective operation to register.
 * @param[in] data The aggregated data of the collective operation to register.
 */
void registerCollectivePrimer(CommunicatorState* commState, const std::string& key, const AggregatedCollective& data)
{
    registerPrimer(commState, key, data, g_collectivePrimers, "Collective");
}

/**
 * @brief Helper function which checks if a P2P key has completed its primer cycle.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] key The key of the P2P operation to check.
 * @return true if the key's primer was already emitted and real data exported
 */
bool isP2PPrimerDone(CommunicatorState* commState, const std::string& key)
{
    return isPrimerDone(commState, key, g_p2pPrimersDone);
}

/**
 * @brief Helper function which registers a new P2P key for primer processing.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] key The key of the P2P operation to register.
 * @param[in] data The aggregated data of the P2P operation to register.
 */
void registerP2PPrimer(CommunicatorState* commState, const std::string& key, const AggregatedP2P& data)
{
    registerPrimer(commState, key, data, g_p2pPrimers, "P2P");
}

/**
 * @brief Helper function which checks if a rank transfer key has completed its primer cycle.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] key The key of the rank transfer operation to check.
 * @return true if the key's primer was already emitted and real data exported
 */
bool isRankPrimerDone(CommunicatorState* commState, const std::string& key)
{
    return isPrimerDone(commState, key, g_rankPrimersDone);
}

/**
 * @brief Helper function which registers a new rank transfer key for primer processing.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] key The key of the rank transfer operation to register.
 * @param[in] data The aggregated data of the rank transfer operation to register.
 */
void registerRankPrimer(CommunicatorState* commState, const std::string& key, const AggregatedTransfer& data)
{
    registerPrimer(commState, key, data, g_rankPrimers, "Rank");
}

/**
 * @brief Helper function which checks if a channel transfer key has completed its primer cycle.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] key The key of the channel transfer operation to check.
 * @return true if the key's primer was already emitted and real data exported
 */
bool isTransferPrimerDone(CommunicatorState* commState, const std::string& key)
{
    return isPrimerDone(commState, key, g_transferPrimersDone);
}

/**
 * @brief Helper function which registers a new channel transfer key for primer processing.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] key The key of the channel transfer operation to register.
 * @param[in] data The aggregated data of the channel transfer operation to register.
 */
void registerTransferPrimer(CommunicatorState* commState, const std::string& key, const AggregatedTransfer& data)
{
    registerPrimer(commState, key, data, g_transferPrimers, "Transfer");
}

/**
 * @brief Remove all pending and completed primer state for one communicator.
 *
 * @param[in] commState Communicator state whose primer state should be discarded.
 */
void cleanupTelemetryPrimerStateForCommunicator(CommunicatorState* commState)
{
    cleanupPrimerStoreForCommunicator(commState, g_collectivePrimers);
    cleanupPrimerStoreForCommunicator(commState, g_p2pPrimers);
    cleanupPrimerStoreForCommunicator(commState, g_rankPrimers);
    cleanupPrimerStoreForCommunicator(commState, g_transferPrimers);
    cleanupDonePrimerStoreForCommunicator(commState, g_collectivePrimersDone);
    cleanupDonePrimerStoreForCommunicator(commState, g_p2pPrimersDone);
    cleanupDonePrimerStoreForCommunicator(commState, g_rankPrimersDone);
    cleanupDonePrimerStoreForCommunicator(commState, g_transferPrimersDone);
}

// =======================================================================================
// Primer Export Functions (emit zero values using same helpers as real exports)
// =======================================================================================

/**
 * @brief Set the values of the metrics which will be exported for the PRIMER export
 * of a collective operation to zero.
 *
 * @return CollectiveEmitView containing the zero values of the metrics to export.
 */
static CollectiveEmitView makePrimerCollectiveEmitView(const AggregatedCollective&)
{
    return CollectiveEmitView{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

/**
 * @brief Set the values of the metrics which will be exported for the PRIMER export
 * of a P2P operation to zero.
 *
 * @return P2PEmitView containing the zero values of the metrics to export.
 */
static P2PEmitView makePrimerP2PEmitView(const AggregatedP2P&)
{
    return P2PEmitView{0.0, 0.0, 0.0, 0.0, 0.0};
}

/**
 * @brief Set the values of the metrics which will be exported for the PRIMER export
 * of a rank transfer operation to zero.
 *
 * @return RankEmitView containing the zero values of the metrics to export.
 */
static RankEmitView makePrimerRankEmitView(const AggregatedTransfer&)
{
    return RankEmitView{0ULL, 0.0, 0.0, 0.0};
}

/**
 * @brief Set the values of the metrics which will be exported for the PRIMER export
 * of a transfer operation to zero.
 *
 * @return TransferEmitView containing the zero values of the metrics to export.
 */
static TransferEmitView makePrimerTransferEmitView(const AggregatedTransfer&)
{
    return TransferEmitView{0.0, 0.0, 0.0};
}

/**
 * @brief Export PRIMER Collective operation metrics to OpenTelemetry.
 *
 * Exports the PRIMER with zero values for aggregated Collective metrics including bytes, time, transfer counts,
 * transfer sizes, and transfer times. All metrics include communicator, rank,
 * hostname, and local_rank labels.
 * Uses the same ExportEligibility helper as real export to ensure identical labels and conditional logic,
 * but exports zero values to establish Prometheus series
 * * Eligibility determines which metrics are valid to export; emit provides the values to publish.
 *
 * @param[in] key Aggregation key in format: Comm<hash>_<func>_Rank<X>ToRank<Y>_<nChannels>Chnl
 * @param[in] p2p Aggregated P2P data containing statistics
 * @param[in] rank Global rank of the process.
 * @param[in] hostname Hostname of the node.
 * @param[in] local_rank Local rank within the node.
 * @param[in] comm_hash Communicator hash for labeling.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string (tensor_parallel, pipeline_parallel, unknown).
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] scale_up_exec_mode Scale-up execution mode (cuda_graph, non_cuda_graph, or unknown).
 */
static void exportCollectiveMetricsPrimer(const std::string& key, const AggregatedCollective& coll, int rank,
                                          const std::string& hostname, int local_rank, uint64_t comm_hash,
                                          const std::string& gpu_pci_bus_id, const std::string& gpu_uuid,
                                          const std::string& comm_type, int nranks,
                                          const std::string& scale_up_exec_mode)
{
    // Exporter-owned decisions. Selects which metrics will be exported based on per exporter criteria.
    CollectiveExportEligibility eligibility = computeCollectiveEligibility(coll);
    // PRIMER emission uses zero values
    CollectiveEmitView emit = makePrimerCollectiveEmitView(coll);
    exportCollectiveMetrics(key, emit, eligibility, rank, hostname, local_rank, comm_hash, gpu_pci_bus_id, gpu_uuid,
                            comm_type, nranks, scale_up_exec_mode, "PRIMER");
}

/**
 * @brief Export PRIMER P2P operation metrics to OpenTelemetry.
 *
 * Exports the PRIMER with zero values for aggregated P2P metrics including bytes, time, transfer counts,
 * transfer sizes, and transfer times. All metrics include communicator, rank,
 * hostname, and local_rank labels.
 * Uses the same ExportEligibility helper as real export to ensure identical labels and conditional logic,
 * but exports zero values to establish Prometheus series
 * Eligibility determines which metrics are valid to export; emit provides the values to publish.
 *
 * @param[in] key Aggregation key in format: Comm<hash>_<func>_Rank<X>ToRank<Y>_<nChannels>Chnl
 * @param[in] p2p Aggregated P2P data containing statistics
 * @param[in] rank Global rank of the process.
 * @param[in] hostname Hostname of the node.
 * @param[in] local_rank Local rank within the node.
 * @param[in] comm_hash Communicator hash for labeling.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string (tensor_parallel, pipeline_parallel, unknown).
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] scale_up_exec_mode Scale-up execution mode (cuda_graph, non_cuda_graph, or unknown).
 */
static void exportP2PMetricsPrimer(const std::string& key, const AggregatedP2P& p2p, int rank,
                                   const std::string& hostname, int local_rank, uint64_t comm_hash,
                                   const std::string& gpu_pci_bus_id, const std::string& gpu_uuid,
                                   const std::string& comm_type, int nranks, const std::string& scale_up_exec_mode)
{
    // Exporter-owned decisions. Selects which metrics will be exported based on per exporter criteria.
    P2PExportEligibility eligibility = computeP2PEligibility(p2p);
    // PRIMER emission uses zero values
    P2PEmitView emit = makePrimerP2PEmitView(p2p);
    exportP2PMetrics(key, emit, eligibility, rank, hostname, local_rank, comm_hash, gpu_pci_bus_id, gpu_uuid, comm_type,
                     nranks, scale_up_exec_mode, "PRIMER");
}

/**
 * @brief Export PRIMER Rank transfer operation metrics to OpenTelemetry.
 *
 * Exports the PRIMER with zero values for aggregated Rank transfer metrics including bytes, time, transfer counts,
 * transfer sizes, and transfer times. All metrics include communicator, rank,
 * hostname, and local_rank labels.
 * Uses the same ExportEligibility helper as real export to ensure identical labels and conditional logic,
 * but exports zero values to establish Prometheus series
 * Eligibility determines which metrics are valid to export; emit provides the values to publish.
 *
 * @param[in] key Aggregation key in format: Comm<hash>_Rank<X>ToRank<Y>_Chnl<channelId>
 * @param[in] transferRef Aggregated transfer data containing statistics
 * @param[in] rank Global rank of the process.
 * @param[in] hostname Hostname of the node.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string (tensor_parallel, pipeline_parallel, unknown).
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] local_rank Local rank within the node.
 * @param[in] scale_up_exec_mode Scale-up execution mode (cuda_graph, non_cuda_graph, or unknown).
 */
static void exportRankMetricsPrimer(const std::string& key, const AggregatedTransfer& transferRef, int rank,
                                    const std::string& hostname, const std::string& gpu_pci_bus_id,
                                    const std::string& gpu_uuid, const std::string& comm_type, int nranks,
                                    int local_rank, const std::string& scale_up_exec_mode)
{
    // Exporter-owned decisions. Selects which metrics will be exported based on per exporter criteria.
    RankExportEligibility eligibility = computeRankEligibility(transferRef);
    // PRIMER emission uses zero values
    RankEmitView emit = makePrimerRankEmitView(transferRef);
    exportRankMetrics(key, emit, eligibility, rank, hostname, gpu_pci_bus_id, gpu_uuid, comm_type, nranks, local_rank,
                      scale_up_exec_mode, "PRIMER");
    OTEL_TRACE(NCCL_INIT, "Rank PRIMER: %s (scale_up_exec_mode=%s)", key.c_str(), scale_up_exec_mode.c_str());
}

/**
 * @brief Export PRIMER Transfer operation metrics to OpenTelemetry.
 *
 * Exports the PRIMER with zero values for aggregated Transfer metrics including bytes, time, transfer counts,
 * transfer sizes, and transfer times. All metrics include communicator, rank,
 * hostname, and local_rank labels.
 * Uses the same ExportEligibility helper as real export to ensure identical labels and conditional logic,
 * but exports zero values to establish Prometheus series
 * Eligibility determines which metrics are valid to export; emit provides the values to publish.
 *
 * @param[in] key Aggregation key in format: Comm<hash>_Rank<X>ToRank<Y>_Chnl<channelId>
 * @param[in] transferRef Aggregated transfer data containing statistics
 * @param[in] rank Global rank of the process.
 * @param[in] hostname Hostname of the node.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string (tensor_parallel, pipeline_parallel, unknown).
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] local_rank Local rank within the node.
 * @param[in] scale_up_exec_mode Scale-up execution mode (cuda_graph, non_cuda_graph, or unknown).
 */
static void exportTransferMetricsPrimer(const std::string& key, const AggregatedTransfer& transferRef, int rank,
                                        const std::string& hostname, const std::string& gpu_pci_bus_id,
                                        const std::string& gpu_uuid, const std::string& comm_type, int nranks,
                                        int local_rank, const std::string& scale_up_exec_mode)
{
    // Exporter-owned decisions. Selects which metrics will be exported based on per exporter criteria.
    TransferExportEligibility eligibility = computeTransferEligibility(transferRef);
    // PRIMER emission uses zero values
    TransferEmitView emit = makePrimerTransferEmitView(transferRef);
    exportTransferMetrics(key, emit, eligibility, rank, hostname, gpu_pci_bus_id, gpu_uuid, comm_type, nranks,
                          local_rank, scale_up_exec_mode, "PRIMER");
    OTEL_TRACE(NCCL_INIT, "Transfer PRIMER: %s (scale_up_exec_mode=%s)", key.c_str(), scale_up_exec_mode.c_str());
}

#ifdef UNIT_TESTING
/**
 * @brief Reset all primer state used by telemetry unit tests.
 */
void resetTelemetryPrimerStateForTests()
{
    g_collectivePrimers.clear();
    g_p2pPrimers.clear();
    g_rankPrimers.clear();
    g_transferPrimers.clear();
    g_collectivePrimersDone.clear();
    g_p2pPrimersDone.clear();
    g_rankPrimersDone.clear();
    g_transferPrimersDone.clear();
}
#endif

#endif  // ENABLE_OTEL
