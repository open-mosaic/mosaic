// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef OTEL_TELEMETRY_PRIMER_H_
#define OTEL_TELEMETRY_PRIMER_H_

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include "aggregation.h"

struct CommunicatorState;

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

#define PRIMER_STABILIZATION_WINDOWS 2U
#define PRIMER_MAX_WAIT_WINDOWS      10U

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
 * @brief Stable communicator identity for primer bookkeeping.
 */
struct PrimerScopeKey
{
    uint64_t commHash;
    int rank;

    bool operator==(const PrimerScopeKey& other) const
    {
        return commHash == other.commHash && rank == other.rank;
    }

    bool operator<(const PrimerScopeKey& other) const
    {
        return std::tie(commHash, rank) < std::tie(other.commHash, other.rank);
    }
};

/**
 * @brief Hash functor for PrimerScopeKey.
 */
struct PrimerScopeKeyHash
{
    size_t operator()(const PrimerScopeKey& key) const noexcept
    {
        return std::hash<uint64_t>{}(key.commHash) ^ (std::hash<int>{}(key.rank) << 1);
    }
};

/**
 * @brief Primer key: (stable communicator identity, operation_key_string)
 */
using PrimerKey = std::pair<PrimerScopeKey, std::string>;

#ifdef ENABLE_OTEL

std::set<std::string> processPendingCollectivePrimers(CommunicatorState* commState,
                                                      const std::map<std::string, AggregatedCollective>& collectives);

std::set<std::string> processPendingP2PPrimers(CommunicatorState* commState,
                                               const std::map<std::string, AggregatedP2P>& p2ps);

std::set<std::string> processPendingRankPrimers(CommunicatorState* commState,
                                                const std::map<std::string, AggregatedTransfer>& rankTransfers);

std::set<std::string> processPendingTransferPrimers(CommunicatorState* commState,
                                                    const std::map<std::string, AggregatedTransfer>& channelTransfers);

bool isCollectivePrimerDone(CommunicatorState* commState, const std::string& key);

void registerCollectivePrimer(CommunicatorState* commState, const std::string& key, const AggregatedCollective& data);

bool isP2PPrimerDone(CommunicatorState* commState, const std::string& key);

void registerP2PPrimer(CommunicatorState* commState, const std::string& key, const AggregatedP2P& data);

bool isRankPrimerDone(CommunicatorState* commState, const std::string& key);

void registerRankPrimer(CommunicatorState* commState, const std::string& key, const AggregatedTransfer& data);

bool isTransferPrimerDone(CommunicatorState* commState, const std::string& key);

void registerTransferPrimer(CommunicatorState* commState, const std::string& key, const AggregatedTransfer& data);

void cleanupTelemetryPrimerStateForCommunicator(CommunicatorState* commState);

#ifdef UNIT_TESTING
void resetTelemetryPrimerStateForTests();
#endif

#endif  // ENABLE_OTEL
#endif  // OTEL_TELEMETRY_PRIMER_H_
