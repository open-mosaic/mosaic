// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef OTEL_TELEMETRY_INTERNAL_H_
#define OTEL_TELEMETRY_INTERNAL_H_

struct CommunicatorState;

/**
 * @file telemetry_internal.h
 * @brief Internal telemetry API shared by telemetry.cc and telemetry_primer.cc when
 *        ENABLE_OTEL is defined at compile time.
 *
 * Emit/eligibility view types, aggregation helpers, and metric export entrypoints.
 * OTEL instruments live in telemetry.cc only.
 */

#ifdef ENABLE_OTEL

#include <opentelemetry/metrics/meter.h>
#include <opentelemetry/metrics/meter_provider.h>
#include <opentelemetry/nostd/shared_ptr.h>
#include <opentelemetry/nostd/unique_ptr.h>

#include <cstdint>
#include <string>

#include "aggregation.h"

namespace metrics_api = opentelemetry::metrics;
namespace nostd       = opentelemetry::nostd;

// =======================================================================================
// Shared Helper Structures
// =======================================================================================

// Contains the metrics values for a collective operation to be exported
struct CollectiveEmitView
{
    double count;
    double totalBytes;
    double totalTimeUs;
    double avgBytes;
    double avgTimeUs;
    double avgNumTransfers;
    double avgTransferSize;
    double avgTransferTime;
};

// Contains information decision whether or not certain metrics will be exported for a collective operation
struct CollectiveExportEligibility
{
    bool export_core;           // avgBytes, avgTime
    bool export_transfers;      // avgNumTransfers, avgTransferSize
    bool export_transfer_time;  // avgTransferTime
};

// Fill the metrics structure with the values to be emitted
CollectiveEmitView makeStandardCollectiveEmitView(const AggregatedCollective& coll);

// Computes wi=hich metrics will be exported for a given collective operation
CollectiveExportEligibility computeCollectiveEligibility(const AggregatedCollective& op);

// Contains the metrics values for a P2P operation to be exported
struct P2PEmitView
{
    double avgBytes;
    double avgTimeUs;
    double avgNumTransfers;
    double avgTransferSize;
    double avgTransferTime;
};

// Contains information decision whether or not certain metrics will be exported for a P2P operation
struct P2PExportEligibility
{
    bool export_core;           // avgBytes, avgTime
    bool export_transfers;      // avgNumTransfers, avgTransferSize
    bool export_transfer_time;  // avgTransferTime
};

// Fill the metrics structure with the values to be emitted
P2PEmitView makeStandardP2PEmitView(const AggregatedP2P& p2p);

// Computes which metrics will be exported for a given P2P operation
P2PExportEligibility computeP2PEligibility(const AggregatedP2P& op);

// Contains the metrics values for a rank transfer operation to be exported
struct RankEmitView
{
    uint64_t totalBytes;
    double latencyUs;
    double rateMBps;
    double activeTimeUs;
};

// Contains information decision whether or not certain metrics will be exported for a rank transfer operation
struct RankExportEligibility
{
    bool export_latency;
    bool export_rate;
};

// Fill the metrics structure with the values to be emitted
RankEmitView makeStandardRankEmitView(const AggregatedTransfer& t);

// Computes which metrics will be exported for a given rank transfer operation
RankExportEligibility computeRankEligibility(const AggregatedTransfer& op);

// Contains the metrics values for a channel transfer operation to be exported
struct TransferEmitView
{
    double avgSize;
    double avgTime;
    double latencyUs;
};

// Contains information decision whether or not certain metrics will be exported for a channel transfer operation
struct TransferExportEligibility
{
    bool export_channel_metrics;
    bool export_avg_time;
    bool export_latency;
};

// Fill the metrics structure with the values to be emitted
TransferEmitView makeStandardTransferEmitView(const AggregatedTransfer& t);

// Computes which metrics will be exported for a given channel transfer operation
TransferExportEligibility computeTransferEligibility(const AggregatedTransfer& op);

// =======================================================================================
// Metric export (definitions in telemetry.cc; called from processWindow in telemetry.cc)
// Primer zero-value export helpers live only in telemetry_primer.cc (file-local).
// =======================================================================================

extern nostd::shared_ptr<metrics_api::MeterProvider> g_meterProvider;
extern nostd::shared_ptr<metrics_api::Meter> g_meter;

extern nostd::unique_ptr<metrics_api::Counter<uint64_t>> g_collBytesCounter;
extern nostd::unique_ptr<metrics_api::Histogram<double>> g_collTimeHist;
extern nostd::unique_ptr<metrics_api::Histogram<double>> g_collCountHist;
extern nostd::unique_ptr<metrics_api::Histogram<double>> g_collNumTransfersHist;
extern nostd::unique_ptr<metrics_api::Histogram<double>> g_collTransferSizeHist;
extern nostd::unique_ptr<metrics_api::Histogram<double>> g_collTransferTimeHist;

extern nostd::unique_ptr<metrics_api::Histogram<double>> g_p2pBytesHist;
extern nostd::unique_ptr<metrics_api::Histogram<double>> g_p2pTimeHist;
extern nostd::unique_ptr<metrics_api::Histogram<double>> g_p2pNumTransfersHist;
extern nostd::unique_ptr<metrics_api::Histogram<double>> g_p2pTransferSizeHist;
extern nostd::unique_ptr<metrics_api::Histogram<double>> g_p2pTransferTimeHist;

extern nostd::unique_ptr<metrics_api::Counter<uint64_t>> g_rankBytesCounter;
extern nostd::unique_ptr<metrics_api::Histogram<double>> g_rankLatencyHist;
extern nostd::unique_ptr<metrics_api::Histogram<double>> g_rankRateHist;

extern nostd::unique_ptr<metrics_api::Histogram<double>> g_transferSizeHist;
extern nostd::unique_ptr<metrics_api::Histogram<double>> g_transferTimeHist;
extern nostd::unique_ptr<metrics_api::Histogram<double>> g_transferLatencyHist;

void exportCollectiveMetrics(const std::string& key, const CollectiveEmitView& emit,
                             const CollectiveExportEligibility& eligibility, int rank, const std::string& hostname,
                             int local_rank, uint64_t comm_hash, const std::string& gpu_pci_bus_id,
                             const std::string& gpu_uuid, const std::string& comm_type, int nranks,
                             const std::string& scale_up_exec_mode, const char* export_tag);

void exportP2PMetrics(const std::string& key, const P2PEmitView& emit, const P2PExportEligibility& eligibility,
                      int rank, const std::string& hostname, int local_rank, uint64_t comm_hash,
                      const std::string& gpu_pci_bus_id, const std::string& gpu_uuid, const std::string& comm_type,
                      int nranks, const std::string& scale_up_exec_mode, const char* export_tag);

void exportRankMetrics(const std::string& key, const RankEmitView& emit, const RankExportEligibility& eligibility,
                       int rank, const std::string& hostname, const std::string& gpu_pci_bus_id,
                       const std::string& gpu_uuid, const std::string& comm_type, int nranks, int local_rank,
                       const std::string& scale_up_exec_mode, const char* export_tag);

void exportTransferMetrics(const std::string& key, const TransferEmitView& emit,
                           const TransferExportEligibility& eligibility, int rank, const std::string& hostname,
                           const std::string& gpu_pci_bus_id, const std::string& gpu_uuid, const std::string& comm_type,
                           int nranks, int local_rank, const std::string& scale_up_exec_mode, const char* export_tag);

void processWindow(CommunicatorState* commState, int window_idx);

#endif  // ENABLE_OTEL

void telemetryRuntimeInit();
void telemetryRuntimeCleanup();
void telemetryRuntimeNotifyWindowReady(CommunicatorState* commState, int window_idx);
void telemetryRuntimeUnregisterCommunicator(CommunicatorState* commState);

#endif  // OTEL_TELEMETRY_INTERNAL_H_
