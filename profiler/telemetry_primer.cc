// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "telemetry_primer.h"

#include <map>
#include <set>
#include <string>

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

// Primer storage for each metric type
static std::map<PrimerKey, PrimerData<AggregatedCollective>> g_collectivePrimers;
static std::map<PrimerKey, PrimerData<AggregatedP2P>> g_p2pPrimers;
static std::map<PrimerKey, PrimerData<AggregatedTransfer>> g_rankPrimers;
static std::map<PrimerKey, PrimerData<AggregatedTransfer>> g_transferPrimers;

// Track keys that have completed the primer cycle (to avoid re-priming on subsequent windows)
static std::set<PrimerKey> g_collectivePrimersDone;
static std::set<PrimerKey> g_p2pPrimersDone;
static std::set<PrimerKey> g_rankPrimersDone;
static std::set<PrimerKey> g_transferPrimersDone;

// =======================================================================================
// Helper Functions
// =======================================================================================

/**
 * @brief Merge two AggregatedCollective structures.
 *
 * Summ up the metrics of one window with another window to keep the history of the metrics.
 * This is used to make sure the fisrt STANDRD exported Collective will contain all the metrics
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
 * This is used to make sure the fisrt STANDRD exported P2P will contain all the metrics
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
 * This is used to make sure the fisrt STANDRD exported Transfer will contain all the metrics
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
    auto mode =
        static_cast<CommunicatorState::ScaleUpExecMode>(commState->scaleUpExecMode.load(std::memory_order_acquire));
    return mode != CommunicatorState::ScaleUpExecMode::UNKNOWN;
}

/**
 * @brief Process pending Collective primers and go through the primer state machine
 *
 * Process pending Collective primers from previous windows and increase metrics values of the pending operations
 * by one window's worth of metrics. The metrics values are increased by merging the aggregated data of the
 * pending operations with the aggregated data of the current window.
 * Export the Collective PRIMER if cuda_graph scale_up_exec_mode is detected and stable. Export STANDARD metrics message
 * on the next window following the emission of the Collective PRIMER.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] collectives Map of aggregated collective data keyed by operation name.
 * @return Set of keys that have been handled and are no longer pending.
 */
std::set<std::string> processPendingCollectivePrimers(CommunicatorState* commState,
                                                      const std::map<std::string, AggregatedCollective>& collectives)
{
    const bool scaleUpModeKnown = isScaleUpExecModeKnown(commState);
    std::set<std::string> handledKeys;

    // Process pending primers (from previous windows)
    // Go theough the Primer storage and treats the ones associated with the commState
    for (auto it = g_collectivePrimers.begin(); it != g_collectivePrimers.end();)
    {
        if (it->first.first != commState)
        {
            ++it;
            continue;
        }

        const std::string& key                       = it->first.second;
        PrimerData<AggregatedCollective>& primerData = it->second;

        // Merge with current window data if present
        auto currentIt = collectives.find(key);
        if (currentIt != collectives.end())
        {
            primerData.aggregatedData = mergeAggregatedCollective(primerData.aggregatedData, currentIt->second);
            handledKeys.insert(key);
        }

        if (primerData.state == PrimerState::PENDING_PRIMER)
        {
            // Check if we've exceeded maximum wait time - force emit to prevent indefinite waiting
            if (primerData.windowsWaited >= PRIMER_MAX_WAIT_WINDOWS)
            {
                std::string scaleUpMode = scaleUpModeKnown ? commState->getScaleUpExecModeString() : "unknown";
                exportCollectiveMetricsPrimer(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                              commState->local_rank, commState->comm_hash, commState->gpu_pci_bus_id,
                                              commState->gpu_uuid, commState->getCommTypeString(), commState->nranks,
                                              scaleUpMode);
                primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                OTEL_INFO(NCCL_INIT,
                          "Collective PRIMER FORCE-EMITTED: %s (max wait of %u windows exceeded, "
                          "scale_up_exec_mode=%s, count=%d, bytes=%zu)",
                          key.c_str(), PRIMER_MAX_WAIT_WINDOWS, scaleUpMode.c_str(), primerData.aggregatedData.count,
                          primerData.aggregatedData.totalBytes);
                ++it;
            }
            else if (!scaleUpModeKnown)
            {
                primerData.windowsWaited++;
                // Still waiting for scale_up_exec_mode to be known
                OTEL_TRACE(NCCL_INIT,
                           "Collective PRIMER DELAYED: %s (scale_up_exec_mode still UNKNOWN, waited %u/%u windows, "
                           "accumulating: count=%d, bytes=%zu)",
                           key.c_str(), primerData.windowsWaited, PRIMER_MAX_WAIT_WINDOWS,
                           primerData.aggregatedData.count, primerData.aggregatedData.totalBytes);
                ++it;
            }
            else
            {
                std::string scaleUpMode = commState->getScaleUpExecModeString();

                // CUDA_GRAPH is the final stable state - emit immediately!
                // Once CUDA graphs are captured during warmup, they persist and the mode never
                // transitions back to NON_CUDA_GRAPH. Emitting immediately avoids unnecessary delay.
                if (scaleUpMode == std::string("cuda_graph"))
                {
                    exportCollectiveMetricsPrimer(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                                  commState->local_rank, commState->comm_hash,
                                                  commState->gpu_pci_bus_id, commState->gpu_uuid,
                                                  commState->getCommTypeString(), commState->nranks, scaleUpMode);
                    primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                    OTEL_TRACE(NCCL_INIT,
                               "Collective PRIMER EMITTED: %s (zeros sent immediately with stable "
                               "scale_up_exec_mode=%s, real data on next window: count=%d, bytes=%zu)",
                               key.c_str(), scaleUpMode.c_str(), primerData.aggregatedData.count,
                               primerData.aggregatedData.totalBytes);
                    ++it;
                }
                // NON_CUDA_GRAPH might transition to CUDA_GRAPH during warmup - wait to stabilize
                else if (primerData.windowsWaited < PRIMER_STABILIZATION_WINDOWS)
                {
                    primerData.windowsWaited++;
                    OTEL_TRACE(NCCL_INIT,
                               "Collective PRIMER STABILIZING: %s (scale_up_exec_mode=%s, waited %u/%u windows, "
                               "accumulating: count=%d, bytes=%zu)",
                               key.c_str(), scaleUpMode.c_str(), primerData.windowsWaited, PRIMER_STABILIZATION_WINDOWS,
                               primerData.aggregatedData.count, primerData.aggregatedData.totalBytes);
                    ++it;
                }
                else
                {
                    // Mode has been NON_CUDA_GRAPH for N windows - it's stable, emit primer
                    exportCollectiveMetricsPrimer(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                                  commState->local_rank, commState->comm_hash,
                                                  commState->gpu_pci_bus_id, commState->gpu_uuid,
                                                  commState->getCommTypeString(), commState->nranks, scaleUpMode);
                    primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                    OTEL_TRACE(NCCL_INIT,
                               "Collective PRIMER EMITTED: %s (zeros sent with stable scale_up_exec_mode=%s after %u "
                               "windows, real data on next window: count=%d, bytes=%zu)",
                               key.c_str(), scaleUpMode.c_str(), primerData.windowsWaited,
                               primerData.aggregatedData.count, primerData.aggregatedData.totalBytes);
                    ++it;
                }
            }
        }
        else if (primerData.state == PrimerState::PRIMER_EMITTED_AWAITING_REAL)
        {
            // Emit the real accumulated data (mode should now be stable)
            std::string scaleUpMode                 = commState->getScaleUpExecModeString();
            const AggregatedCollective& coll        = primerData.aggregatedData;
            CollectiveExportEligibility eligibility = computeCollectiveEligibility(coll);
            CollectiveEmitView emit                 = makeStandardCollectiveEmitView(coll);
            exportCollectiveMetrics(key, emit, eligibility, commState->rank, commState->hostname, commState->local_rank,
                                    commState->comm_hash, commState->gpu_pci_bus_id, commState->gpu_uuid,
                                    commState->getCommTypeString(), commState->nranks, scaleUpMode, "STANDARD");
            OTEL_TRACE(
                NCCL_INIT,
                "Collective REAL DATA EXPORTED: %s (primer complete with scale_up_exec_mode=%s: count=%d, bytes=%zu)",
                key.c_str(), scaleUpMode.c_str(), primerData.aggregatedData.count,
                primerData.aggregatedData.totalBytes);
            g_collectivePrimersDone.insert(it->first);  // Mark as done
            it = g_collectivePrimers.erase(it);
        }
        else
        {
            ++it;
        }
    }

    return handledKeys;
}

/**
 * @brief Process pending P2P primers and go through the primer state machine
 *
 * Process pending P2P primers from previous windows and increase metrics values of the pending operations
 * by one window's worth of metrics. The metrics values are increased by merging the aggregated data of the
 * pending operations with the aggregated data of the current window.
 * Export the P2P PRIMER if cuda_graph scale_up_exec_mode is detected and stable. Export STANDARD metrics message
 * on the next window following the emission of the P2P PRIMER.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] p2ps Map of aggregated P2P data keyed by operation name.
 * @return Set of keys that have been handled and are no longer pending.
 */
std::set<std::string> processPendingP2PPrimers(CommunicatorState* commState,
                                               const std::map<std::string, AggregatedP2P>& p2ps)
{
    const bool scaleUpModeKnown = isScaleUpExecModeKnown(commState);
    std::set<std::string> handledKeys;

    // Process pending primers (from previous windows)
    for (auto it = g_p2pPrimers.begin(); it != g_p2pPrimers.end();)
    {
        if (it->first.first != commState)
        {
            ++it;
            continue;
        }

        const std::string& key                = it->first.second;
        PrimerData<AggregatedP2P>& primerData = it->second;

        auto currentIt = p2ps.find(key);
        if (currentIt != p2ps.end())
        {
            primerData.aggregatedData = mergeAggregatedP2P(primerData.aggregatedData, currentIt->second);
            handledKeys.insert(key);
        }

        if (primerData.state == PrimerState::PENDING_PRIMER)
        {
            // Check if we've exceeded maximum wait time - force emit to prevent indefinite waiting
            if (primerData.windowsWaited >= PRIMER_MAX_WAIT_WINDOWS)
            {
                std::string scaleUpMode = scaleUpModeKnown ? commState->getScaleUpExecModeString() : "unknown";
                exportP2PMetricsPrimer(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                       commState->local_rank, commState->comm_hash, commState->gpu_pci_bus_id,
                                       commState->gpu_uuid, commState->getCommTypeString(), commState->nranks,
                                       scaleUpMode);
                primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                OTEL_INFO(
                    NCCL_INIT,
                    "P2P PRIMER FORCE-EMITTED: %s (max wait of %u windows exceeded, scale_up_exec_mode=%s, count=%d)",
                    key.c_str(), PRIMER_MAX_WAIT_WINDOWS, scaleUpMode.c_str(), primerData.aggregatedData.count);
                ++it;
            }
            else if (!scaleUpModeKnown)
            {
                primerData.windowsWaited++;
                OTEL_TRACE(NCCL_INIT,
                           "P2P PRIMER DELAYED: %s (scale_up_exec_mode still UNKNOWN, waited %u/%u windows, "
                           "accumulating: count=%d)",
                           key.c_str(), primerData.windowsWaited, PRIMER_MAX_WAIT_WINDOWS,
                           primerData.aggregatedData.count);
                ++it;
            }
            else
            {
                std::string scaleUpMode = commState->getScaleUpExecModeString();

                // CUDA_GRAPH is the final stable state - emit immediately!
                if (scaleUpMode == std::string("cuda_graph"))
                {
                    exportP2PMetricsPrimer(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                           commState->local_rank, commState->comm_hash, commState->gpu_pci_bus_id,
                                           commState->gpu_uuid, commState->getCommTypeString(), commState->nranks,
                                           scaleUpMode);
                    primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                    OTEL_TRACE(NCCL_INIT,
                               "P2P PRIMER EMITTED: %s (zeros sent immediately with stable scale_up_exec_mode=%s, real "
                               "data on next window: count=%d)",
                               key.c_str(), scaleUpMode.c_str(), primerData.aggregatedData.count);
                    ++it;
                }
                // NON_CUDA_GRAPH might transition to CUDA_GRAPH during warmup - wait to stabilize
                else if (primerData.windowsWaited < PRIMER_STABILIZATION_WINDOWS)
                {
                    primerData.windowsWaited++;
                    OTEL_TRACE(NCCL_INIT,
                               "P2P PRIMER STABILIZING: %s (scale_up_exec_mode=%s, waited %u/%u windows, accumulating: "
                               "count=%d)",
                               key.c_str(), scaleUpMode.c_str(), primerData.windowsWaited, PRIMER_STABILIZATION_WINDOWS,
                               primerData.aggregatedData.count);
                    ++it;
                }
                else
                {
                    // Mode has been NON_CUDA_GRAPH for N windows - it's stable, emit primer
                    exportP2PMetricsPrimer(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                           commState->local_rank, commState->comm_hash, commState->gpu_pci_bus_id,
                                           commState->gpu_uuid, commState->getCommTypeString(), commState->nranks,
                                           scaleUpMode);
                    primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                    OTEL_TRACE(NCCL_INIT,
                               "P2P PRIMER EMITTED: %s (zeros sent with stable scale_up_exec_mode=%s after %u windows, "
                               "real data on next window: count=%d)",
                               key.c_str(), scaleUpMode.c_str(), primerData.windowsWaited,
                               primerData.aggregatedData.count);
                    ++it;
                }
            }
        }
        else if (primerData.state == PrimerState::PRIMER_EMITTED_AWAITING_REAL)
        {
            std::string scaleUpMode  = commState->getScaleUpExecModeString();
            const AggregatedP2P& p2p = primerData.aggregatedData;
            // Exporter-owned decisions
            P2PExportEligibility eligibility = computeP2PEligibility(p2p);
            // STANDARD emission uses real values
            P2PEmitView emit = makeStandardP2PEmitView(p2p);
            exportP2PMetrics(key, emit, eligibility, commState->rank, commState->hostname, commState->local_rank,
                             commState->comm_hash, commState->gpu_pci_bus_id, commState->gpu_uuid,
                             commState->getCommTypeString(), commState->nranks, scaleUpMode, "STANDARD");
            OTEL_TRACE(NCCL_INIT, "P2P REAL DATA EXPORTED: %s (primer complete with scale_up_exec_mode=%s: count=%d)",
                       key.c_str(), scaleUpMode.c_str(), primerData.aggregatedData.count);
            g_p2pPrimersDone.insert(it->first);
            it = g_p2pPrimers.erase(it);
        }
        else
        {
            ++it;
        }
    }

    return handledKeys;
}

/**
 * @brief Process pending Rank transfer primers and go through the primer state machine
 *
 * Process pending Rank transfer primers from previous windows and increase metrics values of the pending operations
 * by one window's worth of metrics. The metrics values are increased by merging the aggregated data of the
 * pending operations with the aggregated data of the current window.
 * Export the Rank transfer PRIMER if cuda_graph scale_up_exec_mode is detected and stable. Export STANDARD metrics
 message
 * on the next window following the emission of the Rank transfer PRIMER.

 * @param[in] commState Communicator state containing the window to process.
 * @param[in] rankTransfers Map of aggregated rank transfer data keyed by operation name.
 * @return Set of keys that have been handled and are no longer pending.
 */
std::set<std::string> processPendingRankPrimers(CommunicatorState* commState,
                                                const std::map<std::string, AggregatedTransfer>& rankTransfers)
{
    const bool scaleUpModeKnown = isScaleUpExecModeKnown(commState);
    std::set<std::string> handledKeys;

    // Phase 1: Process pending primers
    for (auto it = g_rankPrimers.begin(); it != g_rankPrimers.end();)
    {
        if (it->first.first != commState)
        {
            ++it;
            continue;
        }

        const std::string& key                     = it->first.second;
        PrimerData<AggregatedTransfer>& primerData = it->second;

        auto currentIt = rankTransfers.find(key);
        if (currentIt != rankTransfers.end())
        {
            primerData.aggregatedData = mergeAggregatedTransfer(primerData.aggregatedData, currentIt->second);
            handledKeys.insert(key);
        }

        if (primerData.state == PrimerState::PENDING_PRIMER)
        {
            if (primerData.windowsWaited >= PRIMER_MAX_WAIT_WINDOWS)
            {
                std::string scaleUpMode = scaleUpModeKnown ? commState->getScaleUpExecModeString() : "unknown";
                exportRankMetricsPrimer(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                        commState->gpu_pci_bus_id, commState->gpu_uuid, commState->getCommTypeString(),
                                        commState->nranks, commState->local_rank, scaleUpMode);
                primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                OTEL_INFO(NCCL_INIT,
                          "Rank PRIMER FORCE-EMITTED: %s (max wait of %u windows exceeded, scale_up_exec_mode=%s, "
                          "count=%d)",
                          key.c_str(), PRIMER_MAX_WAIT_WINDOWS, scaleUpMode.c_str(), primerData.aggregatedData.count);
                ++it;
            }
            else if (!scaleUpModeKnown)
            {
                primerData.windowsWaited++;
                OTEL_TRACE(NCCL_INIT,
                           "Rank PRIMER DELAYED: %s (scale_up_exec_mode still UNKNOWN, waited %u/%u windows, "
                           "accumulating: count=%d)",
                           key.c_str(), primerData.windowsWaited, PRIMER_MAX_WAIT_WINDOWS,
                           primerData.aggregatedData.count);
                ++it;
            }
            else
            {
                std::string scaleUpMode = commState->getScaleUpExecModeString();

                if (scaleUpMode == std::string("cuda_graph"))
                {
                    exportRankMetricsPrimer(key, primerData.aggregatedData, commState->rank, commState->hostname,

                                            commState->gpu_pci_bus_id, commState->gpu_uuid,

                                            commState->getCommTypeString(), commState->nranks, commState->local_rank,

                                            scaleUpMode);
                    primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                    OTEL_TRACE(
                        NCCL_INIT,
                        "Rank PRIMER EMITTED: %s (zeros sent immediately with stable scale_up_exec_mode=%s, real "
                        "data on next window: count=%d)",
                        key.c_str(), scaleUpMode.c_str(), primerData.aggregatedData.count);
                    ++it;
                }
                else if (primerData.windowsWaited < PRIMER_STABILIZATION_WINDOWS)
                {
                    primerData.windowsWaited++;
                    OTEL_TRACE(
                        NCCL_INIT,
                        "Rank PRIMER STABILIZING: %s (scale_up_exec_mode=%s, waited %u/%u windows, accumulating: "
                        "count=%d)",
                        key.c_str(), scaleUpMode.c_str(), primerData.windowsWaited, PRIMER_STABILIZATION_WINDOWS,
                        primerData.aggregatedData.count);
                    ++it;
                }
                else
                {
                    exportRankMetricsPrimer(key, primerData.aggregatedData, commState->rank, commState->hostname,

                                            commState->gpu_pci_bus_id, commState->gpu_uuid,

                                            commState->getCommTypeString(), commState->nranks, commState->local_rank,

                                            scaleUpMode);
                    primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                    OTEL_TRACE(
                        NCCL_INIT,
                        "Rank PRIMER EMITTED: %s (zeros sent with stable scale_up_exec_mode=%s after %u windows, "
                        "real data on next window: count=%d)",
                        key.c_str(), scaleUpMode.c_str(), primerData.windowsWaited, primerData.aggregatedData.count);
                    ++it;
                }
            }
        }
        else if (primerData.state == PrimerState::PRIMER_EMITTED_AWAITING_REAL)
        {
            std::string scaleUpMode           = commState->getScaleUpExecModeString();
            const AggregatedTransfer& xfer    = primerData.aggregatedData;
            RankExportEligibility eligibility = computeRankEligibility(xfer);
            RankEmitView emit                 = makeStandardRankEmitView(xfer);
            exportRankMetrics(key, emit, eligibility, commState->rank, commState->hostname, commState->gpu_pci_bus_id,
                              commState->gpu_uuid, commState->getCommTypeString(), commState->nranks,
                              commState->local_rank, scaleUpMode, "STANDARD");
            OTEL_TRACE(NCCL_INIT, "Rank REAL DATA EXPORTED: %s (primer complete with scale_up_exec_mode=%s: count=%d)",
                       key.c_str(), scaleUpMode.c_str(), primerData.aggregatedData.count);
            g_rankPrimersDone.insert(it->first);
            it = g_rankPrimers.erase(it);
        }
        else
        {
            ++it;
        }
    }

    return handledKeys;
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
    const bool scaleUpModeKnown = isScaleUpExecModeKnown(commState);
    std::set<std::string> handledKeys;

    // Phase 1: Process pending primers
    for (auto it = g_transferPrimers.begin(); it != g_transferPrimers.end();)
    {
        if (it->first.first != commState)
        {
            ++it;
            continue;
        }

        const std::string& key                     = it->first.second;
        PrimerData<AggregatedTransfer>& primerData = it->second;

        auto currentIt = channelTransfers.find(key);
        if (currentIt != channelTransfers.end())
        {
            primerData.aggregatedData = mergeAggregatedTransfer(primerData.aggregatedData, currentIt->second);
            handledKeys.insert(key);
        }

        if (primerData.state == PrimerState::PENDING_PRIMER)
        {
            if (primerData.windowsWaited >= PRIMER_MAX_WAIT_WINDOWS)
            {
                std::string scaleUpMode = scaleUpModeKnown ? commState->getScaleUpExecModeString() : "unknown";
                exportTransferMetricsPrimer(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                            commState->gpu_pci_bus_id, commState->gpu_uuid,
                                            commState->getCommTypeString(), commState->nranks, commState->local_rank,
                                            scaleUpMode);
                primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                OTEL_INFO(NCCL_INIT,
                          "Transfer PRIMER FORCE-EMITTED: %s (max wait of %u windows exceeded, scale_up_exec_mode=%s, "
                          "count=%d)",
                          key.c_str(), PRIMER_MAX_WAIT_WINDOWS, scaleUpMode.c_str(), primerData.aggregatedData.count);
                ++it;
            }
            else if (!scaleUpModeKnown)
            {
                primerData.windowsWaited++;
                OTEL_TRACE(NCCL_INIT,
                           "Transfer PRIMER DELAYED: %s (scale_up_exec_mode still UNKNOWN, waited %u/%u windows, "
                           "accumulating: count=%d)",
                           key.c_str(), primerData.windowsWaited, PRIMER_MAX_WAIT_WINDOWS,
                           primerData.aggregatedData.count);
                ++it;
            }
            else
            {
                std::string scaleUpMode = commState->getScaleUpExecModeString();

                if (scaleUpMode == std::string("cuda_graph"))
                {
                    exportTransferMetricsPrimer(key, primerData.aggregatedData, commState->rank, commState->hostname,

                                                commState->gpu_pci_bus_id, commState->gpu_uuid,

                                                commState->getCommTypeString(), commState->nranks,
                                                commState->local_rank,

                                                scaleUpMode);
                    primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                    OTEL_TRACE(NCCL_INIT,
                               "Transfer PRIMER EMITTED: %s (zeros sent immediately with stable scale_up_exec_mode=%s, "
                               "real data on next window: count=%d)",
                               key.c_str(), scaleUpMode.c_str(), primerData.aggregatedData.count);
                    ++it;
                }
                else if (primerData.windowsWaited < PRIMER_STABILIZATION_WINDOWS)
                {
                    primerData.windowsWaited++;
                    OTEL_TRACE(NCCL_INIT,
                               "Transfer PRIMER STABILIZING: %s (scale_up_exec_mode=%s, waited %u/%u windows, "
                               "accumulating: count=%d)",
                               key.c_str(), scaleUpMode.c_str(), primerData.windowsWaited, PRIMER_STABILIZATION_WINDOWS,
                               primerData.aggregatedData.count);
                    ++it;
                }
                else
                {
                    exportTransferMetricsPrimer(key, primerData.aggregatedData, commState->rank, commState->hostname,

                                                commState->gpu_pci_bus_id, commState->gpu_uuid,

                                                commState->getCommTypeString(), commState->nranks,
                                                commState->local_rank,

                                                scaleUpMode);
                    primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                    OTEL_TRACE(NCCL_INIT,
                               "Transfer PRIMER EMITTED: %s (zeros sent with stable scale_up_exec_mode=%s after %u "
                               "windows, real data on next window: count=%d)",
                               key.c_str(), scaleUpMode.c_str(), primerData.windowsWaited,
                               primerData.aggregatedData.count);
                    ++it;
                }
            }
        }
        else if (primerData.state == PrimerState::PRIMER_EMITTED_AWAITING_REAL)
        {
            std::string scaleUpMode               = commState->getScaleUpExecModeString();
            const AggregatedTransfer& xfer        = primerData.aggregatedData;
            TransferExportEligibility eligibility = computeTransferEligibility(xfer);
            TransferEmitView emit                 = makeStandardTransferEmitView(xfer);
            exportTransferMetrics(key, emit, eligibility, commState->rank, commState->hostname,
                                  commState->gpu_pci_bus_id, commState->gpu_uuid, commState->getCommTypeString(),
                                  commState->nranks, commState->local_rank, scaleUpMode, "STANDARD");
            OTEL_TRACE(NCCL_INIT,
                       "Transfer REAL DATA EXPORTED: %s (primer complete with scale_up_exec_mode=%s: count=%d)",
                       key.c_str(), scaleUpMode.c_str(), primerData.aggregatedData.count);
            g_transferPrimersDone.insert(it->first);
            it = g_transferPrimers.erase(it);
        }
        else
        {
            ++it;
        }
    }

    return handledKeys;
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
    PrimerKey pkey = {commState, key};
    return g_collectivePrimersDone.count(pkey) > 0;
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
    PrimerKey pkey                           = {commState, key};
    g_collectivePrimers[pkey].aggregatedData = data;
    g_collectivePrimers[pkey].state          = PrimerState::PENDING_PRIMER;
    g_collectivePrimers[pkey].windowsWaited  = 0;

    bool scaleUpModeKnown = isScaleUpExecModeKnown(commState);
    if (!scaleUpModeKnown)
    {
        OTEL_INFO(NCCL_INIT, "Collective NEW KEY: %s (scale_up_exec_mode UNKNOWN, waiting: count=%d)", key.c_str(),
                  data.count);
    }
    else
    {
        std::string scaleUpMode = commState->getScaleUpExecModeString();
        OTEL_INFO(NCCL_INIT,
                  "Collective NEW KEY: %s (scale_up_exec_mode=%s, starting %u-window stabilization: count=%d)",
                  key.c_str(), scaleUpMode.c_str(), PRIMER_STABILIZATION_WINDOWS, data.count);
    }
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
    PrimerKey pkey = {commState, key};
    return g_p2pPrimersDone.count(pkey) > 0;
}

/**
 * @brief Help  er function which registers a new P2P key for primer processing.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] key The key of the P2P operation to register.
 * @param[in] data The aggregated data of the P2P operation to register.
 */
void registerP2PPrimer(CommunicatorState* commState, const std::string& key, const AggregatedP2P& data)
{
    PrimerKey pkey                    = {commState, key};
    g_p2pPrimers[pkey].aggregatedData = data;
    g_p2pPrimers[pkey].state          = PrimerState::PENDING_PRIMER;
    g_p2pPrimers[pkey].windowsWaited  = 0;

    bool scaleUpModeKnown = isScaleUpExecModeKnown(commState);
    if (!scaleUpModeKnown)
    {
        OTEL_INFO(NCCL_INIT, "P2P NEW KEY: %s (scale_up_exec_mode UNKNOWN, waiting: count=%d)", key.c_str(),
                  data.count);
    }
    else
    {
        std::string scaleUpMode = commState->getScaleUpExecModeString();
        OTEL_INFO(NCCL_INIT, "P2P NEW KEY: %s (scale_up_exec_mode=%s, starting %u-window stabilization: count=%d)",
                  key.c_str(), scaleUpMode.c_str(), PRIMER_STABILIZATION_WINDOWS, data.count);
    }
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
    PrimerKey pkey = {commState, key};
    return g_rankPrimersDone.count(pkey) > 0;
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
    PrimerKey pkey                     = {commState, key};
    g_rankPrimers[pkey].aggregatedData = data;
    g_rankPrimers[pkey].state          = PrimerState::PENDING_PRIMER;
    g_rankPrimers[pkey].windowsWaited  = 0;

    bool scaleUpModeKnown = isScaleUpExecModeKnown(commState);
    if (!scaleUpModeKnown)
    {
        OTEL_INFO(NCCL_INIT, "Rank NEW KEY: %s (scale_up_exec_mode UNKNOWN, waiting: count=%d)", key.c_str(),
                  data.count);
    }
    else
    {
        std::string scaleUpMode = commState->getScaleUpExecModeString();
        OTEL_INFO(NCCL_INIT, "Rank NEW KEY: %s (scale_up_exec_mode=%s, starting %u-window stabilization: count=%d)",
                  key.c_str(), scaleUpMode.c_str(), PRIMER_STABILIZATION_WINDOWS, data.count);
    }
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
    PrimerKey pkey = {commState, key};
    return g_transferPrimersDone.count(pkey) > 0;
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
    PrimerKey pkey                         = {commState, key};
    g_transferPrimers[pkey].aggregatedData = data;
    g_transferPrimers[pkey].state          = PrimerState::PENDING_PRIMER;
    g_transferPrimers[pkey].windowsWaited  = 0;

    bool scaleUpModeKnown = isScaleUpExecModeKnown(commState);
    if (!scaleUpModeKnown)
    {
        OTEL_INFO(NCCL_INIT, "Transfer NEW KEY: %s (scale_up_exec_mode UNKNOWN, waiting: count=%d)", key.c_str(),
                  data.count);
    }
    else
    {
        std::string scaleUpMode = commState->getScaleUpExecModeString();
        OTEL_INFO(NCCL_INIT, "Transfer NEW KEY: %s (scale_up_exec_mode=%s, starting %u-window stabilization: count=%d)",
                  key.c_str(), scaleUpMode.c_str(), PRIMER_STABILIZATION_WINDOWS, data.count);
    }
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

// =======================================================================================
// Unit Testing Helpers
// =======================================================================================

#ifdef UNIT_TESTING

void test_resetPrimerState()
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

PrimerState test_getCollectivePrimerState(CommunicatorState* commState, const std::string& key)
{
    PrimerKey pkey = {commState, key};
    auto it        = g_collectivePrimers.find(pkey);
    if (it == g_collectivePrimers.end())
    {
        return PrimerState::PENDING_PRIMER;
    }
    return it->second.state;
}

void test_setCollectivePrimerState(CommunicatorState* commState, const std::string& key, PrimerState state)
{
    PrimerKey pkey                  = {commState, key};
    g_collectivePrimers[pkey].state = state;
}

bool test_isCollectivePrimerDone(CommunicatorState* commState, const std::string& key)
{
    PrimerKey pkey = {commState, key};
    return g_collectivePrimersDone.count(pkey) > 0;
}

void test_markCollectivePrimerDone(CommunicatorState* commState, const std::string& key)
{
    PrimerKey pkey = {commState, key};
    g_collectivePrimersDone.insert(pkey);
}

size_t test_getCollectivePrimerCount()
{
    return g_collectivePrimers.size();
}

size_t test_getCollectivePrimerDoneCount()
{
    return g_collectivePrimersDone.size();
}

PrimerState test_getP2PPrimerState(CommunicatorState* commState, const std::string& key)
{
    PrimerKey pkey = {commState, key};
    auto it        = g_p2pPrimers.find(pkey);
    if (it == g_p2pPrimers.end())
    {
        return PrimerState::PENDING_PRIMER;
    }
    return it->second.state;
}

PrimerState test_getRankPrimerState(CommunicatorState* commState, const std::string& key)
{
    PrimerKey pkey = {commState, key};
    auto it        = g_rankPrimers.find(pkey);
    if (it == g_rankPrimers.end())
    {
        return PrimerState::PENDING_PRIMER;
    }
    return it->second.state;
}

PrimerState test_getTransferPrimerState(CommunicatorState* commState, const std::string& key)
{
    PrimerKey pkey = {commState, key};
    auto it        = g_transferPrimers.find(pkey);
    if (it == g_transferPrimers.end())
    {
        return PrimerState::PENDING_PRIMER;
    }
    return it->second.state;
}

uint32_t test_getCollectivePrimerWindowsWaited(CommunicatorState* commState, const std::string& key)
{
    PrimerKey pkey = {commState, key};
    auto it        = g_collectivePrimers.find(pkey);
    if (it == g_collectivePrimers.end())
    {
        return 0;
    }
    return it->second.windowsWaited;
}

#endif  // UNIT_TESTING

#endif  // ENABLE_OTEL
