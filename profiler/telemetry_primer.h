// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef OTEL_TELEMETRY_PRIMER_H_
#define OTEL_TELEMETRY_PRIMER_H_

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>

#include "aggregation.h"
#include "communicator_state.h"

/**
 * @file telemetry_primer.h
 * @brief Primer state machine for establishing Prometheus time series before real data.
 *
 * This module implements the "primer" algorithm to solve three key issues with NCCL profiler metrics:
 * 1. Metrics completing within a single window (Grafana sees no change → rate() = 0)
 * 2. scale_up_exec_mode changing from UNKNOWN causing duplicate label series
 * 3. Zero transfer times not being exported causing missing Grafana series
 *
 * The primer emits zero-value metrics first to establish the Prometheus series structure,
 * then exports real accumulated data. This ensures Prometheus increase() and rate() functions
 * work correctly even for short-lived operations.
 *
 * Primer orchestration and metric export entrypoints are only declared when ENABLE_OTEL is
 * defined at compile time (same guard as telemetry.cc).
 */

// Number of windows to wait after scale_up_exec_mode becomes NON_CUDA_GRAPH before emitting primer.
// This allows time for the NON_CUDA_GRAPH → CUDA_GRAPH transition during model warmup/graph capture.
// Note: CUDA_GRAPH is the final stable state and can be emitted immediately (no transition back).
#define PRIMER_STABILIZATION_WINDOWS 2

// Maximum total windows to wait before forcing primer emission (even if scale_up_exec_mode is still UNKNOWN)
// This prevents primers from waiting indefinitely if mode detection fails or is unsupported.
// Default: 10 windows (~50 seconds at 5s/window)
#define PRIMER_MAX_WAIT_WINDOWS 10

/**
 * @brief Primer state for a metric key.
 */
enum class PrimerState : uint8_t
{
    PENDING_PRIMER,               // New key detected, accumulating data, waiting for scale_up_exec_mode to stabilize
    PRIMER_EMITTED_AWAITING_REAL  // Primer (zeros) emitted, waiting to export real data on next window
};

/**
 * @brief Generic primer data structure for any metric type.
 *
 * @tparam T The aggregated metric type (AggregatedCollective, AggregatedP2P, AggregatedTransfer)
 */
template <typename T>
struct PrimerData
{
    T aggregatedData;        // Accumulated metric data across windows
    PrimerState state;       // Current primer state
    uint32_t windowsWaited;  // Number of windows we've waited for mode to stabilize

    PrimerData() : aggregatedData(), state(PrimerState::PENDING_PRIMER), windowsWaited(0) {}
};

/**
 * @brief Primer key: (CommunicatorState*, operation_key_string)
 *
 * The CommunicatorState* identifies the specific communicator context.
 * The operation_key_string is the Grafana-visible collective name (e.g., "Comm<hash>_AllReduce_RING_LL_1Chnl").
 */
using PrimerKey = std::pair<CommunicatorState*, std::string>;

#ifdef ENABLE_OTEL

// =======================================================================================
// Primer Orchestration Functions - Phase 1 (Process Pending Primers)
// =======================================================================================

/**
 * @brief Process pending collective primers (from previous windows).
 *
 * Handles the primer state machine for primers that were registered in previous windows:
 * - Merges current window data with accumulated primer data
 * - Checks scale_up_exec_mode stabilization
 * - Emits primers when ready (either after stabilization or force-emit after timeout)
 * - Emits real accumulated data after primer is sent
 *
 * @param[in] commState Communicator state being processed
 * @param[in] collectives Map of collective keys to aggregated data for current window
 * @return Set of keys that were handled (exported or updated) by pending primers
 */
std::set<std::string> processPendingCollectivePrimers(CommunicatorState* commState,
                                                      const std::map<std::string, AggregatedCollective>& collectives);

/**
 * @brief Process pending P2P primers (from previous windows).
 *
 * Handles the primer state machine for primers that were registered in previous windows:
 * - Merges current window data with accumulated primer data
 * - Checks scale_up_exec_mode stabilization
 * - Emits primers when ready (either after stabilization or force-emit after timeout)
 * - Emits real accumulated data after primer is sent
 *
 * @param[in] commState Communicator state being processed
 * @param[in] p2ps Map of P2P keys to aggregated data for current window
 * @return Set of keys that were handled (exported or updated) by pending primers
 */
std::set<std::string> processPendingP2PPrimers(CommunicatorState* commState,
                                               const std::map<std::string, AggregatedP2P>& p2ps);

/**
 * @brief Process pending Rank transfer primers (from previous windows).
 *
 * Handles the primer state machine for primers that were registered in previous windows:
 * - Merges current window data with accumulated primer data
 * - Checks scale_up_exec_mode stabilization
 * - Emits primers when ready (either after stabilization or force-emit after timeout)
 * - Emits real accumulated data after primer is sent
 *
 * @param[in] commState Communicator state being processed
 * @param[in] rankTransfers Map of rank transfer keys to aggregated data for current window
 * @return Set of keys that were handled (exported or updated) by pending primers
 */
std::set<std::string> processPendingRankPrimers(CommunicatorState* commState,
                                                const std::map<std::string, AggregatedTransfer>& rankTransfers);

/**
 * @brief Process pending Channel transfer primers (from previous windows).
 *
 * Handles the primer state machine for primers that were registered in previous windows:
 * - Merges current window data with accumulated primer data
 * - Checks scale_up_exec_mode stabilization
 * - Emits primers when ready (either after stabilization or force-emit after timeout)
 * - Emits real accumulated data after primer is sent
 *
 * @param[in] commState Communicator state being processed
 * @param[in] channelTransfers Map of channel transfer keys to aggregated data for current window
 * @return Set of keys that were handled (exported or updated) by pending primers
 */
std::set<std::string> processPendingTransferPrimers(CommunicatorState* commState,
                                                    const std::map<std::string, AggregatedTransfer>& channelTransfers);

// =======================================================================================
// Primer Helper Functions - Phase 2 (Check/Register New Keys)
// =======================================================================================

/**
 * @brief Check if a collective key has completed its primer cycle.
 *
 * @return true if the key's primer was already emitted and real data exported
 */
bool isCollectivePrimerDone(CommunicatorState* commState, const std::string& key);

/**
 * @brief Register a new collective key for primer processing.
 *
 * Initializes the key in PENDING_PRIMER state. The primer will be emitted
 * in subsequent windows after scale_up_exec_mode stabilizes.
 */
void registerCollectivePrimer(CommunicatorState* commState, const std::string& key, const AggregatedCollective& data);

/**
 * @brief Check if a P2P key has completed its primer cycle.
 */
bool isP2PPrimerDone(CommunicatorState* commState, const std::string& key);

/**
 * @brief Register a new P2P key for primer processing.
 */
void registerP2PPrimer(CommunicatorState* commState, const std::string& key, const AggregatedP2P& data);

/**
 * @brief Check if a rank transfer key has completed its primer cycle.
 */
bool isRankPrimerDone(CommunicatorState* commState, const std::string& key);

/**
 * @brief Register a new rank transfer key for primer processing.
 */
void registerRankPrimer(CommunicatorState* commState, const std::string& key, const AggregatedTransfer& data);

/**
 * @brief Check if a channel transfer key has completed its primer cycle.
 */
bool isTransferPrimerDone(CommunicatorState* commState, const std::string& key);

/**
 * @brief Register a new channel transfer key for primer processing.
 */
void registerTransferPrimer(CommunicatorState* commState, const std::string& key, const AggregatedTransfer& data);

#endif  // ENABLE_OTEL
#endif  // OTEL_TELEMETRY_PRIMER_H_
