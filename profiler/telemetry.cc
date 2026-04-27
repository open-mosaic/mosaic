// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "telemetry.h"

#include <pthread.h>

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "aggregation.h"
#include "communicator_state.h"
#include "events.h"
#include "param.h"
#include "profiler_otel.h"

// OpenTelemetry includes - only include if available
#ifdef ENABLE_OTEL
#include <opentelemetry/common/key_value_iterable_view.h>
#include <opentelemetry/exporters/otlp/otlp_http_metric_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_http_metric_exporter_options.h>
#include <opentelemetry/metrics/meter.h>
#include <opentelemetry/metrics/meter_provider.h>
#include <opentelemetry/nostd/shared_ptr.h>
#include <opentelemetry/nostd/unique_ptr.h>
#include <opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader.h>
#include <opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader_factory.h>
#include <opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader_options.h>
#include <opentelemetry/sdk/metrics/meter_provider.h>
#include <opentelemetry/sdk/metrics/meter_provider_factory.h>
#include <opentelemetry/sdk/metrics/push_metric_exporter.h>

namespace metrics_api = opentelemetry::metrics;
namespace nostd       = opentelemetry::nostd;
namespace sdk_metrics = opentelemetry::sdk::metrics;
namespace otlp        = opentelemetry::exporter::otlp;
#endif  // ENABLE_OTEL

// PARAM: TelemetryEnable
// ENV: NCCL_PROFILER_OTEL_TELEMETRY_ENABLE
// DEFAULT: 1
// DESCRIPTION: Enable/disable telemetry thread + OTLP metric export.
OTEL_PARAM(TelemetryEnable, "PROFILER_OTEL_TELEMETRY_ENABLE", 1);

// PARAM: TelemetryEndpoint
// ENV: NCCL_PROFILER_OTEL_TELEMETRY_ENDPOINT
// DEFAULT: http://localhost:4318
// DESCRIPTION: Base OTLP HTTP endpoint; exporter appends /v1/metrics.
OTEL_STRING_PARAM(TelemetryEndpoint, "PROFILER_OTEL_TELEMETRY_ENDPOINT", "http://localhost:4318");

// PARAM: TelemetryIntervalSec
// ENV: NCCL_PROFILER_OTEL_TELEMETRY_INTERVAL_SEC
// DEFAULT: 5
// DESCRIPTION: Export interval (seconds) for periodic metric reader; also used for window timeout.
OTEL_PARAM(TelemetryIntervalSec, "PROFILER_OTEL_TELEMETRY_INTERVAL_SEC", 5);

// PARAM: TelemetryOtelBatchTimeoutMs
// ENV: NCCL_PROFILER_OTEL_TELEMETRY_BATCH_TIMEOUT_MS
// DEFAULT: 3000
// DESCRIPTION: Export timeout (milliseconds) for OTLP HTTP exporter/reader.
OTEL_PARAM(TelemetryOtelBatchTimeoutMs, "PROFILER_OTEL_TELEMETRY_BATCH_TIMEOUT_MS", 3000);

/**
 * @brief Get the OpenTelemetry collector endpoint URL.
 *
 * Reads from NCCL_PROFILER_OTEL_TELEMETRY_ENDPOINT environment variable.
 * The /v1/metrics path is appended in initializeOtelMetrics().
 *
 * @return Endpoint URL string (default: "http://localhost:4318").
 */
static std::string getTelemetryEndpoint()
{
    return std::string(ncclParamTelemetryEndpoint());
}

// Telemetry thread state
static std::atomic<bool> g_telThreadStop{false};
static pthread_t g_telThread;
static pthread_mutex_t g_telLock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_telCond  = PTHREAD_COND_INITIALIZER;

// Global list of communicator states to monitor
static std::vector<CommunicatorState*> g_commStates;
static pthread_mutex_t g_commStatesLock = PTHREAD_MUTEX_INITIALIZER;

#ifdef ENABLE_OTEL
// OpenTelemetry meter and instruments
static nostd::shared_ptr<metrics_api::MeterProvider> g_meterProvider;
static nostd::shared_ptr<metrics_api::Meter> g_meter;

// Metric instruments for Collective Information
static nostd::unique_ptr<metrics_api::Counter<uint64_t>> g_collBytesCounter;
static nostd::unique_ptr<metrics_api::Histogram<double>> g_collTimeHist;
static nostd::unique_ptr<metrics_api::Histogram<double>> g_collCountHist;
static nostd::unique_ptr<metrics_api::Histogram<double>> g_collNumTransfersHist;
static nostd::unique_ptr<metrics_api::Histogram<double>> g_collTransferSizeHist;
static nostd::unique_ptr<metrics_api::Histogram<double>> g_collTransferTimeHist;

// Metric instruments for P2P Information
static nostd::unique_ptr<metrics_api::Histogram<double>> g_p2pBytesHist;
static nostd::unique_ptr<metrics_api::Histogram<double>> g_p2pTimeHist;
static nostd::unique_ptr<metrics_api::Histogram<double>> g_p2pNumTransfersHist;
static nostd::unique_ptr<metrics_api::Histogram<double>> g_p2pTransferSizeHist;
static nostd::unique_ptr<metrics_api::Histogram<double>> g_p2pTransferTimeHist;

// Metric instruments for Rank Information
static nostd::unique_ptr<metrics_api::Counter<uint64_t>> g_rankBytesCounter;
static nostd::unique_ptr<metrics_api::Histogram<double>> g_rankLatencyHist;
static nostd::unique_ptr<metrics_api::Histogram<double>> g_rankRateHist;

// Metric instruments for Transfer Information
static nostd::unique_ptr<metrics_api::Histogram<double>> g_transferSizeHist;
static nostd::unique_ptr<metrics_api::Histogram<double>> g_transferTimeHist;
static nostd::unique_ptr<metrics_api::Histogram<double>> g_transferLatencyHist;

// =======================================================================================
// Generic Primer Infrastructure for Collective, P2P, Rank, and Transfer metrics
// =======================================================================================

/**
 * @brief Primer state for a metric key.
 */
enum class PrimerState : uint8_t
{
    PENDING_PRIMER,               // New key detected, accumulating data, waiting for scale_up_exec_mode to stabilize
    PRIMER_EMITTED_AWAITING_REAL  // Primer (zeros) emitted, waiting to export real data on next window
};

// Number of windows to wait after scale_up_exec_mode becomes NON_CUDA_GRAPH before emitting primer.
// This allows time for the NON_CUDA_GRAPH → CUDA_GRAPH transition during model warmup/graph capture.
// Note: CUDA_GRAPH is the final stable state and can be emitted immediately (no transition back).
// Note: Despite the "scale_up" name, this mode applies to ALL communicators (scale-up and scale-out).
#define PRIMER_STABILIZATION_WINDOWS 2

// Maximum total windows to wait before forcing primer emission (even if scale_up_exec_mode is still UNKNOWN)
// This prevents primers from waiting indefinitely if mode detection fails or is unsupported.
// Default: 10 windows (~50 seconds at 5s/window)
#define PRIMER_MAX_WAIT_WINDOWS 10

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
 */
using PrimerKey = std::pair<CommunicatorState*, std::string>;

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

static pthread_mutex_t g_primerLock = PTHREAD_MUTEX_INITIALIZER;

/**
 * @brief Initialize OpenTelemetry metrics and instruments.
 *
 * Creates the OTLP HTTP exporter, periodic metric reader, meter provider, and
 * all metric instruments (histograms and counters) for collective, P2P, rank,
 * and channel metrics.
 *
 * @note Called once during telemetry initialization.
 * @note Configures export interval from NCCL_PROFILER_OTEL_TELEMETRY_INTERVAL_SEC.
 */
static void initializeOtelMetrics()
{
    OTEL_TRACE(NCCL_INIT, "==> initializeOtelMetrics()");

    // Create OTLP HTTP exporter
    otlp::OtlpHttpMetricExporterOptions exporterOptions;
    std::string endpoint    = getTelemetryEndpoint();
    exporterOptions.url     = endpoint + "/v1/metrics";  // Append /v1/metrics path to base endpoint
    exporterOptions.timeout = std::chrono::milliseconds(OTEL_GET_PARAM(TelemetryOtelBatchTimeoutMs));

    OTEL_INFO(NCCL_INIT, "OpenTelemetry endpoint: %s", exporterOptions.url.c_str());

    auto exporter = otlp::OtlpHttpMetricExporterFactory::Create(exporterOptions);

    // Create periodic exporting metric reader
    sdk_metrics::PeriodicExportingMetricReaderOptions readerOptions;
    readerOptions.export_interval_millis = std::chrono::milliseconds(OTEL_GET_PARAM(TelemetryIntervalSec) * 1000);
    readerOptions.export_timeout_millis  = std::chrono::milliseconds(OTEL_GET_PARAM(TelemetryOtelBatchTimeoutMs));

    auto reader = sdk_metrics::PeriodicExportingMetricReaderFactory::Create(std::move(exporter), readerOptions);

    // Create meter provider
    auto sdk_provider = std::unique_ptr<sdk_metrics::MeterProvider>(new sdk_metrics::MeterProvider());
    sdk_provider->AddMetricReader(std::move(reader));
    g_meterProvider = nostd::shared_ptr<metrics_api::MeterProvider>(sdk_provider.release());

    // Get meter
    g_meter = g_meterProvider->GetMeter("nccl_profiler", "1.0.0");

    // Initialize Collective Information metrics
    g_collBytesCounter = g_meter->CreateUInt64Counter("nccl_profiler_collective_bytes",
                                                      "Total bytes transferred in collective operations", "bytes");

    g_collTimeHist =
        g_meter->CreateDoubleHistogram("nccl_profiler_collective_time", "Average time per collective operation", "us");

    g_collCountHist =
        g_meter->CreateDoubleHistogram("nccl_profiler_collective_count", "Number of collective operations", "count");

    g_collNumTransfersHist = g_meter->CreateDoubleHistogram("nccl_profiler_collective_num_transfers",
                                                            "Average number of transfers per collective", "count");

    g_collTransferSizeHist = g_meter->CreateDoubleHistogram("nccl_profiler_collective_transfer_size",
                                                            "Average transfer size for collective", "bytes");

    g_collTransferTimeHist = g_meter->CreateDoubleHistogram("nccl_profiler_collective_transfer_time",
                                                            "Average transfer time for collective", "us");

    // Initialize P2P Information metrics
    g_p2pBytesHist =
        g_meter->CreateDoubleHistogram("nccl_profiler_p2p_bytes", "Average bytes per P2P operation", "bytes");

    g_p2pTimeHist = g_meter->CreateDoubleHistogram("nccl_profiler_p2p_time", "Average time per P2P operation", "us");

    g_p2pNumTransfersHist = g_meter->CreateDoubleHistogram("nccl_profiler_p2p_num_transfers",
                                                           "Average number of transfers per P2P", "count");

    g_p2pTransferSizeHist =
        g_meter->CreateDoubleHistogram("nccl_profiler_p2p_transfer_size", "Average transfer size for P2P", "bytes");

    g_p2pTransferTimeHist =
        g_meter->CreateDoubleHistogram("nccl_profiler_p2p_transfer_time", "Average transfer time for P2P", "us");

    // Initialize Rank Information metrics
    g_rankBytesCounter =
        g_meter->CreateUInt64Counter("nccl_profiler_rank_bytes", "Bytes sent from rank to rank", "bytes");

    g_rankLatencyHist = g_meter->CreateDoubleHistogram("nccl_profiler_rank_latency",
                                                       "Latency from rank to rank (from linear regression)", "us");

    g_rankRateHist = g_meter->CreateDoubleHistogram(
        "nccl_profiler_rank_rate", "Transfer rate from rank to rank (bandwidth based on active transfer time)", "MB/s");

    // Initialize Transfer Information metrics
    g_transferSizeHist =
        g_meter->CreateDoubleHistogram("nccl_profiler_transfer_size", "Average transfer size per channel", "bytes");

    g_transferTimeHist =
        g_meter->CreateDoubleHistogram("nccl_profiler_transfer_time", "Average transfer time per channel", "us");

    g_transferLatencyHist = g_meter->CreateDoubleHistogram(
        "nccl_profiler_transfer_latency", "Transfer latency per channel (from linear regression)", "us");

    OTEL_INFO(NCCL_INIT, "OpenTelemetry metrics initialized");
    OTEL_TRACE(NCCL_INIT, "<== initializeOtelMetrics()");
}

// =======================================================================================
// Helper Functions for Primer Algorithm
// =======================================================================================

/**
 * @brief Merge two AggregatedCollective structures.
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
 */
static bool isScaleUpExecModeKnown(CommunicatorState* commState)
{
    auto mode =
        static_cast<CommunicatorState::ScaleUpExecMode>(commState->scaleUpExecMode.load(std::memory_order_acquire));
    return mode != CommunicatorState::ScaleUpExecMode::UNKNOWN;
}

/**
 * @brief Export collective operation metrics to OpenTelemetry.
 *
 * Exports aggregated collective metrics including bytes, time, transfer counts,
 * transfer sizes, and transfer times. All metrics include communicator, rank,
 * hostname, and local_rank labels.
 *
 * When is_primer is true, exports 0 values for all metrics to establish the Prometheus
 * time series. When is_primer is false, exports actual aggregated statistics.
 *
 * @param[in] key Aggregation key in format: Comm<hash>_<func>_<algo>_<proto>_<nChannels>Chnl
 * @param[in] coll Aggregated collective data containing statistics.
 * @param[in] rank Global rank of the process.
 * @param[in] hostname Hostname of the node.
 * @param[in] local_rank Local rank within the node.
 * @param[in] comm_hash Communicator hash for labeling.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string (tensor_parallel, pipeline_parallel, unknown).
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] scale_up_exec_mode Scale-up execution mode (cuda_graph, non_cuda_graph, or unknown).
 * @param[in] is_primer If true, exports 0 values (primer); if false, exports actual aggregated values.
 */
static void exportCollectiveMetrics(const std::string& key, const AggregatedCollective& coll, int rank,
                                    const std::string& hostname, int local_rank, uint64_t comm_hash,
                                    const std::string& gpu_pci_bus_id, const std::string& gpu_uuid,
                                    const std::string& comm_type, int nranks, const std::string& scale_up_exec_mode,
                                    bool is_primer)
{
    std::string rank_str       = std::to_string(rank);
    std::string local_rank_str = std::to_string(local_rank);
    std::string communicator   = std::to_string(comm_hash);

    std::map<std::string, std::string> labels = {
        {"communicator",       communicator          },
        {"operation",          key                   },
        {"rank",               rank_str              },
        {"hostname",           hostname              },
        {"local_rank",         local_rank_str        },
        {"gpu_pci_bus_id",     gpu_pci_bus_id        },
        {"gpu_uuid",           gpu_uuid              },
        {"comm_type",          comm_type             },
        {"comm_nranks",        std::to_string(nranks)},
        {"scale_up_exec_mode", scale_up_exec_mode    }
    };

    auto ctx = opentelemetry::context::Context{};

    // Export bytes (counter for real, 0 for primer)
    size_t bytesValue = is_primer ? 0 : coll.totalBytes;
    g_collBytesCounter->Add(bytesValue, labels, ctx);

    // Export time, count, and transfer stats
    double avgTime    = is_primer ? 0.0 : coll.getAverageTime();
    double countValue = is_primer ? 0.0 : (double)coll.count;

    g_collTimeHist->Record(avgTime, labels, ctx);
    g_collCountHist->Record(countValue, labels, ctx);

    // Calculate transfer statistics
    double avgNumTransfers = is_primer ? 0.0 : coll.getAverageTransferCount();
    double avgTransferSize = is_primer ? 0.0 : coll.getAverageTransferSize();
    double avgTransferTime = is_primer ? 0.0 : coll.getAverageTransferTime();

    g_collNumTransfersHist->Record(avgNumTransfers, labels, ctx);
    g_collTransferSizeHist->Record(avgTransferSize, labels, ctx);

    // Only export transfer time if the real data would have it
    // (CUDA-graph scale-up volume-only paths leave cachedTotalTransferTimeUs at 0)
    if (coll.cachedTotalTransferTimeUs > 0.0)
    {
        g_collTransferTimeHist->Record(avgTransferTime, labels, ctx);

        if (is_primer)
        {
            OTEL_TRACE(NCCL_INIT, "Collective PRIMER: %s (all zeros, including avgTransferTime, scale_up_exec_mode=%s)",
                       key.c_str(), scale_up_exec_mode.c_str());
        }
        else if (avgNumTransfers > 0)
        {
            OTEL_TRACE(NCCL_INIT,
                       "Exported Collective: %s, AvgBytes: %.2f, AvgTime: %.2f us, "
                       "AvgNumTransfers: %.2f, AvgTransferSize: %.2f, AvgTransferTime: %.2f us",
                       key.c_str(), coll.getAverageSize(), avgTime, avgNumTransfers, avgTransferSize, avgTransferTime);
        }
    }
    else
    {
        if (is_primer)
        {
            OTEL_TRACE(NCCL_INIT,
                       "Collective PRIMER: %s (all zeros, avgTransferTime OMITTED - will be 0 in real data, "
                       "scale_up_exec_mode=%s)",
                       key.c_str(), scale_up_exec_mode.c_str());
        }
        else if (avgNumTransfers > 0)
        {
            OTEL_TRACE(NCCL_INIT, "Exported Collective: %s, AvgBytes: %.2f, AvgTime: %.2f us (no transfer timing)",
                       key.c_str(), coll.getAverageSize(), avgTime);
        }
    }

    if (!is_primer && avgNumTransfers == 0)
    {
        OTEL_TRACE(NCCL_INIT, "Exported Collective: %s, AvgBytes: %.2f, AvgTime: %.2f us (no transfers)", key.c_str(),
                   coll.getAverageSize(), avgTime);
    }
}

/**
 * @brief Export P2P operation metrics to OpenTelemetry.
 *
 * Exports aggregated P2P metrics including bytes, time, transfer counts,
 * transfer sizes, and transfer times. All metrics include communicator, rank,
 * hostname, and local_rank labels.
 *
 * @param[in] key Aggregation key in format: Comm<hash>_<func>_Rank<X>ToRank<Y>_<nChannels>Chnl
 * @param[in] p2p Aggregated P2P data containing statistics.
 * @param[in] rank Global rank of the process.
 * @param[in] hostname Hostname of the node.
 * @param[in] local_rank Local rank within the node.
 * @param[in] comm_hash Communicator hash for labeling.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string (tensor_parallel, pipeline_parallel, unknown).
 * @param[in] nranks Number of ranks in the communicator.
 */
/**
 * @brief Export P2P operation metrics to OpenTelemetry.
 *
 * Exports aggregated P2P metrics including bytes, time, transfer counts,
 * transfer sizes, and transfer times. All metrics include communicator, rank,
 * hostname, and local_rank labels.
 *
 * When is_primer is true, exports 0 values for all metrics to establish the Prometheus
 * time series. When is_primer is false, exports actual aggregated statistics.
 *
 * @param[in] key Aggregation key in format: Comm<hash>_<func>_Rank<X>ToRank<Y>_<nChannels>Chnl
 * @param[in] p2p Aggregated P2P data containing statistics.
 * @param[in] rank Global rank of the process.
 * @param[in] hostname Hostname of the node.
 * @param[in] local_rank Local rank within the node.
 * @param[in] comm_hash Communicator hash for labeling.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string (tensor_parallel, pipeline_parallel, unknown).
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] scale_up_exec_mode Scale-up execution mode (cuda_graph, non_cuda_graph, or unknown).
 * @param[in] is_primer If true, exports 0 values (primer); if false, exports actual aggregated values.
 */
static void exportP2PMetrics(const std::string& key, const AggregatedP2P& p2p, int rank, const std::string& hostname,
                             int local_rank, uint64_t comm_hash, const std::string& gpu_pci_bus_id,
                             const std::string& gpu_uuid, const std::string& comm_type, int nranks,
                             const std::string& scale_up_exec_mode, bool is_primer)
{
    std::string rank_str       = std::to_string(rank);
    std::string local_rank_str = std::to_string(local_rank);
    std::string communicator   = std::to_string(comm_hash);

    // Extract function name (Send/Recv) and pipeline info from key
    // P2P Key format: Comm<hash>_(<hostname>)_<func>_Pipeline<src>ToPipeline<dst>_<nChannels>Chnl
    std::string func_name    = "P2P";
    std::string src_pipeline = "";
    std::string dst_pipeline = "";

    // Find function name after the )_ pattern
    size_t paren_pos = key.find(")_");
    if (paren_pos != std::string::npos)
    {
        size_t func_start = paren_pos + 2;
        size_t func_end   = key.find("_Pipeline", func_start);
        if (func_end != std::string::npos)
        {
            func_name = key.substr(func_start, func_end - func_start);
        }
    }

    // Find source pipeline (Pipeline<N>ToPipeline)
    size_t pipeline_pos = key.find("_Pipeline");
    if (pipeline_pos != std::string::npos)
    {
        size_t src_start = pipeline_pos + 9;  // After "_Pipeline"
        size_t to_pos    = key.find("ToPipeline", src_start);
        if (to_pos != std::string::npos)
        {
            src_pipeline     = key.substr(src_start, to_pos - src_start);
            size_t dst_start = to_pos + 10;  // After "ToPipeline"
            size_t dst_end   = key.find("_", dst_start);
            if (dst_end != std::string::npos)
            {
                dst_pipeline = key.substr(dst_start, dst_end - dst_start);
            }
        }
    }

    // Build operation label with rank info from local_rank
    std::string rank_info = " (Rank" + local_rank_str + ")";
    std::string operation = func_name + " Pipeline" + src_pipeline + "→Pipeline" + dst_pipeline + rank_info;

    std::map<std::string, std::string> labels = {
        {"communicator",       communicator          },
        {"operation",          operation             },
        {"rank",               rank_str              },
        {"hostname",           hostname              },
        {"local_rank",         local_rank_str        },
        {"gpu_pci_bus_id",     gpu_pci_bus_id        },
        {"gpu_uuid",           gpu_uuid              },
        {"comm_type",          comm_type             },
        {"comm_nranks",        std::to_string(nranks)},
        {"scale_up_exec_mode", scale_up_exec_mode    }
    };

    auto ctx = opentelemetry::context::Context{};

    // Calculate values (0 for primer, actual for real data)
    double avgBytes = is_primer ? 0.0 : ((double)p2p.totalBytes / p2p.count);
    double avgTime  = is_primer ? 0.0 : (p2p.totalTimeUs / p2p.count);

    g_p2pBytesHist->Record(avgBytes, labels, ctx);
    g_p2pTimeHist->Record(avgTime, labels, ctx);

    // Export transfer statistics
    double avgNumTransfers = is_primer ? 0.0 : p2p.getAverageTransferCount();
    double avgTransferSize = is_primer ? 0.0 : p2p.getAverageTransferSize();
    double avgTransferTime = is_primer ? 0.0 : p2p.getAverageTransferTime();

    // Only export transfer metrics if real data will have them
    if (avgNumTransfers > 0 || (is_primer && p2p.getAverageTransferCount() > 0))
    {
        g_p2pNumTransfersHist->Record(avgNumTransfers, labels, ctx);
        g_p2pTransferSizeHist->Record(avgTransferSize, labels, ctx);

        if (p2p.cachedTotalTransferTimeUs > 0.0)
        {
            g_p2pTransferTimeHist->Record(avgTransferTime, labels, ctx);

            if (is_primer)
            {
                OTEL_TRACE(NCCL_INIT, "P2P PRIMER: %s (all zeros, including transfer metrics)", key.c_str());
            }
            else
            {
                OTEL_TRACE(NCCL_INIT,
                           "Exported P2P: %s, AvgBytes: %.2f, AvgTime: %.2f us, "
                           "AvgNumTransfers: %.2f, AvgTransferSize: %.2f, AvgTransferTime: %.2f us",
                           key.c_str(), avgBytes, avgTime, avgNumTransfers, avgTransferSize, avgTransferTime);
            }
        }
        else
        {
            if (is_primer)
            {
                OTEL_TRACE(NCCL_INIT, "P2P PRIMER: %s (all zeros, transfer metrics WITHOUT time)", key.c_str());
            }
            else
            {
                OTEL_TRACE(NCCL_INIT,
                           "Exported P2P: %s, AvgBytes: %.2f, AvgTime: %.2f us, "
                           "AvgNumTransfers: %.2f, AvgTransferSize: %.2f (no transfer timing)",
                           key.c_str(), avgBytes, avgTime, avgNumTransfers, avgTransferSize);
            }
        }
    }
    else
    {
        if (is_primer)
        {
            OTEL_TRACE(NCCL_INIT, "P2P PRIMER: %s (basic metrics only, no transfer details)", key.c_str());
        }
        else
        {
            OTEL_TRACE(NCCL_INIT, "Exported P2P: %s, AvgBytes: %.2f, AvgTime: %.2f us (no transfers)", key.c_str(),
                       avgBytes, avgTime);
        }
    }
}

/**
 * @brief Export rank-to-rank transfer metrics to OpenTelemetry.
 *
 * Exports aggregated transfer metrics between ranks including total bytes,
 * latency (from linear regression), and transfer rate (from active transfer time).
 *
 * Rate calculation: The rate is computed as totalBytes / activeTime where activeTime
 * is the merged (union) of all transfer intervals - representing the time during which
 * at least one transfer was in progress between this rank pair. This approach assumes
 * that parallel transfers share the available bandwidth, so the rate represents the
 * actual bandwidth utilization between two ranks.
 *
 * Metrics include communicator, source_rank, dest_rank, and hostname labels.
 *
 * When is_primer is true, exports 0 values for all metrics to establish the Prometheus
 * time series. When is_primer is false, exports actual aggregated statistics. Note that
 * latency and rate are only exported if available (requires sufficient data for calculation).
 *
 * @param[in] key Aggregation key in format: Comm<hash>_Rank<X>ToRank<Y>
 * @param[in] transferRef Aggregated transfer data containing statistics.
 * @param[in] rank Global rank of the process (unused, extracted from key).
 * @param[in] hostname Hostname of the node.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string (tensor_parallel, pipeline_parallel, unknown).
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] local_rank Local rank within the node.
 * @param[in] scale_up_exec_mode Scale-up execution mode (cuda_graph, non_cuda_graph, or unknown).
 * @param[in] is_primer If true, exports 0 values (primer); if false, exports actual aggregated values.
 */
static void exportRankMetrics(const std::string& key, const AggregatedTransfer& transferRef, int rank,
                              const std::string& hostname, const std::string& gpu_pci_bus_id,
                              const std::string& gpu_uuid, const std::string& comm_type, int nranks, int local_rank,
                              const std::string& scale_up_exec_mode, bool is_primer)
{
    (void)rank;  // Unused - rank info is parsed from key or derived from local_rank
    std::string communicator = "";
    std::string source_rank  = "";
    std::string dest_rank    = "";

    // Parse key
    size_t comm_pos     = key.find("Comm");
    size_t first_sep    = key.find("_", comm_pos + 4);
    size_t pipeline_pos = key.find("_Pipeline");
    size_t peer_pos     = key.find("_ToPeer");

    if (comm_pos != std::string::npos && first_sep != std::string::npos)
    {
        communicator = key.substr(comm_pos + 4, first_sep - comm_pos - 4);
    }

    if (pipeline_pos != std::string::npos && peer_pos == std::string::npos)
    {
        size_t src_start = pipeline_pos + 9;
        size_t to_pos    = key.find("_ToPipeline", src_start);
        if (to_pos != std::string::npos)
        {
            std::string src_pipeline = key.substr(src_start, to_pos - src_start);
            std::string dst_pipeline = key.substr(to_pos + 11);
            source_rank              = "Pipeline" + src_pipeline + " (Rank" + std::to_string(local_rank) + ")";
            dest_rank                = "Pipeline" + dst_pipeline;
        }
    }
    else if (peer_pos != std::string::npos)
    {
        size_t rank_pos = key.find("_Rank");
        if (rank_pos != std::string::npos)
        {
            source_rank = key.substr(rank_pos + 5, peer_pos - rank_pos - 5);
        }
        dest_rank = key.substr(peer_pos + 7);
    }

    std::map<std::string, std::string> labels = {
        {"communicator",       communicator              },
        {"source_rank",        source_rank               },
        {"dest_rank",          dest_rank                 },
        {"hostname",           hostname                  },
        {"gpu_pci_bus_id",     gpu_pci_bus_id            },
        {"gpu_uuid",           gpu_uuid                  },
        {"comm_type",          comm_type                 },
        {"comm_nranks",        std::to_string(nranks)    },
        {"local_rank",         std::to_string(local_rank)},
        {"scale_up_exec_mode", scale_up_exec_mode        }
    };

    auto ctx = opentelemetry::context::Context{};

    // Export bytes
    size_t bytesValue = is_primer ? 0 : transferRef.totalBytes;
    g_rankBytesCounter->Add(bytesValue, labels, ctx);

    // Export latency (if available)
    double latencyUs;
    bool hasLatency = transferRef.getLatencyFromLinearRegression(latencyUs);
    if (hasLatency)
    {
        double latencyValue = is_primer ? 0.0 : latencyUs;
        g_rankLatencyHist->Record(latencyValue, labels, ctx);
        if (!is_primer)
        {
            OTEL_TRACE(NCCL_INIT, "Exported Rank Latency: %s, Latency: %.2f us", key.c_str(), latencyUs);
        }
    }

    // Export rate (if available)
    double rateMBps;
    bool hasRate = transferRef.getRateFromActiveTime(rateMBps);
    if (hasRate)
    {
        double rateValue = is_primer ? 0.0 : rateMBps;
        g_rankRateHist->Record(rateValue, labels, ctx);
        if (!is_primer)
        {
            OTEL_TRACE(NCCL_INIT, "Exported Rank Rate: %s, Bytes: %zu, ActiveTime: %.2f us, Rate: %.2f MB/s",
                       key.c_str(), transferRef.totalBytes, transferRef.getActiveTime(), rateMBps);
        }
    }

    // Log summary
    if (is_primer)
    {
        if (hasLatency && hasRate)
        {
            OTEL_TRACE(NCCL_INIT, "Rank PRIMER: %s (all zeros, including latency and rate)", key.c_str());
        }
        else if (hasLatency)
        {
            OTEL_TRACE(NCCL_INIT, "Rank PRIMER: %s (all zeros, including latency, rate OMITTED)", key.c_str());
        }
        else if (hasRate)
        {
            OTEL_TRACE(NCCL_INIT, "Rank PRIMER: %s (all zeros, including rate, latency OMITTED)", key.c_str());
        }
        else
        {
            OTEL_TRACE(NCCL_INIT, "Rank PRIMER: %s (bytes only, latency and rate OMITTED)", key.c_str());
        }
    }
    else if (!hasRate)
    {
        OTEL_TRACE(NCCL_INIT, "Exported Rank Metrics: %s, Bytes: %zu (no rate data)", key.c_str(),
                   transferRef.totalBytes);
    }
}

/**
 * @brief Export per-channel transfer metrics to OpenTelemetry.
 *
 * Exports aggregated transfer metrics per communicator, rank pair, and channel
 * including average transfer size, average transfer time, and latency (from
 * linear regression). Metrics include communicator, source_rank, dest_rank, channel, and hostname labels.
 *
 * @param[in] key Aggregation key in format: Comm<hash>_Rank<X>ToRank<Y>_Chnl<channelId>
 * @param[in] transferRef Aggregated transfer data containing statistics.
 * @param[in] rank Global rank of the process (unused, extracted from key).
 * @param[in] hostname Hostname of the node.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string (tensor_parallel, pipeline_parallel, unknown).
 * @param[in] nranks Number of ranks in the communicator.
 */
/**
 * @brief Export per-channel transfer metrics to OpenTelemetry.
 *
 * Exports aggregated transfer metrics per communicator, rank pair, and channel
 * including average transfer size, average transfer time, and latency (from
 * linear regression). Metrics include communicator, source_rank, dest_rank, channel,
 * and hostname labels.
 *
 * When is_primer is true, exports 0 values for all metrics to establish the Prometheus
 * time series. When is_primer is false, exports actual aggregated statistics. Note that
 * latency and time are only exported if available (requires sufficient data for calculation).
 *
 * @param[in] key Aggregation key in format: Comm<hash>_Rank<X>ToRank<Y>_Chnl<channelId>
 * @param[in] transferRef Aggregated transfer data containing statistics.
 * @param[in] rank Global rank of the process (unused, extracted from key).
 * @param[in] hostname Hostname of the node.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string (tensor_parallel, pipeline_parallel, unknown).
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] local_rank Local rank within the node.
 * @param[in] scale_up_exec_mode Scale-up execution mode (cuda_graph, non_cuda_graph, or unknown).
 * @param[in] is_primer If true, exports 0 values (primer); if false, exports actual aggregated values.
 */
static void exportTransferMetrics(const std::string& key, const AggregatedTransfer& transferRef, int rank,
                                  const std::string& hostname, const std::string& gpu_pci_bus_id,
                                  const std::string& gpu_uuid, const std::string& comm_type, int nranks, int local_rank,
                                  const std::string& scale_up_exec_mode, bool is_primer)
{
    (void)rank;  // Unused - rank info is parsed from key or derived from local_rank
    std::string communicator = "";
    std::string source_rank  = "";
    std::string dest_rank    = "";
    std::string channel      = "";

    // Parse key
    size_t comm_pos     = key.find("Comm");
    size_t first_sep    = key.find("_", comm_pos + 4);
    size_t pipeline_pos = key.find("_Pipeline");
    size_t peer_pos     = key.find("_ToPeer");
    size_t chnl_pos     = key.find("_Chnl");

    if (comm_pos != std::string::npos && first_sep != std::string::npos)
    {
        communicator = key.substr(comm_pos + 4, first_sep - comm_pos - 4);
    }

    if (pipeline_pos != std::string::npos && peer_pos == std::string::npos)
    {
        size_t src_start = pipeline_pos + 9;
        size_t to_pos    = key.find("_ToPipeline", src_start);
        if (to_pos != std::string::npos)
        {
            std::string src_pipeline = key.substr(src_start, to_pos - src_start);
            std::string dst_pipeline;
            if (chnl_pos != std::string::npos)
            {
                dst_pipeline = key.substr(to_pos + 11, chnl_pos - to_pos - 11);
            }
            else
            {
                dst_pipeline = key.substr(to_pos + 11);
            }
            source_rank = "Pipeline" + src_pipeline + " (Rank" + std::to_string(local_rank) + ")";
            dest_rank   = "Pipeline" + dst_pipeline;
        }
    }
    else if (peer_pos != std::string::npos)
    {
        size_t rank_pos = key.find("_Rank");
        if (rank_pos != std::string::npos)
        {
            source_rank = key.substr(rank_pos + 5, peer_pos - rank_pos - 5);
        }
        if (chnl_pos != std::string::npos)
        {
            dest_rank = key.substr(peer_pos + 7, chnl_pos - peer_pos - 7);
        }
    }

    if (chnl_pos != std::string::npos)
    {
        channel = key.substr(chnl_pos + 5);
    }

    // Only export if we have actual transfer data OR if this is a primer
    if (transferRef.count > 0 || is_primer)
    {
        std::map<std::string, std::string> labels = {
            {"communicator",       communicator              },
            {"source_rank",        source_rank               },
            {"dest_rank",          dest_rank                 },
            {"channel",            channel                   },
            {"hostname",           hostname                  },
            {"gpu_pci_bus_id",     gpu_pci_bus_id            },
            {"gpu_uuid",           gpu_uuid                  },
            {"comm_type",          comm_type                 },
            {"comm_nranks",        std::to_string(nranks)    },
            {"local_rank",         std::to_string(local_rank)},
            {"scale_up_exec_mode", scale_up_exec_mode        }
        };

        auto ctx = opentelemetry::context::Context{};

        // Export size
        double avgSize = is_primer ? 0.0 : transferRef.getAverageSize();
        g_transferSizeHist->Record(avgSize, labels, ctx);

        // Export time if available
        bool hasTime = (transferRef.totalTimeUs > 0.0);
        if (hasTime)
        {
            double avgTime = is_primer ? 0.0 : transferRef.getAverageTime();
            g_transferTimeHist->Record(avgTime, labels, ctx);
        }

        // Export latency if available (requires linear regression)
        double latencyUs;
        bool hasLatency = transferRef.getLatencyFromLinearRegression(latencyUs);
        if (hasLatency)
        {
            double latencyValue = is_primer ? 0.0 : latencyUs;
            g_transferLatencyHist->Record(latencyValue, labels, ctx);

            if (!is_primer)
            {
                double avgTime = transferRef.getAverageTime();
                OTEL_TRACE(NCCL_INIT, "Exported Transfer: %s, AvgSize: %.2f, AvgTime: %.2f us, Latency: %.2f us",
                           key.c_str(), avgSize, avgTime, latencyUs);
            }
        }
        else if (!is_primer)
        {
            double avgTime = transferRef.getAverageTime();
            OTEL_TRACE(NCCL_INIT, "Exported Transfer: %s, AvgSize: %.2f, AvgTime: %.2f us", key.c_str(), avgSize,
                       avgTime);
        }

        // Log primer summary
        if (is_primer)
        {
            if (hasLatency && hasTime)
            {
                OTEL_TRACE(NCCL_INIT, "Transfer PRIMER: %s (all zeros, including latency and time)", key.c_str());
            }
            else if (hasLatency)
            {
                OTEL_TRACE(NCCL_INIT, "Transfer PRIMER: %s (all zeros, including latency, time OMITTED)", key.c_str());
            }
            else if (hasTime)
            {
                OTEL_TRACE(NCCL_INIT, "Transfer PRIMER: %s (all zeros, including time, latency OMITTED)", key.c_str());
            }
            else
            {
                OTEL_TRACE(NCCL_INIT, "Transfer PRIMER: %s (size only, latency and time OMITTED)", key.c_str());
            }
        }
    }  // End: if (transferRef.count > 0 || is_primer)
}

/**
 * @brief Process a window of events and export metrics.
 *
 * Aggregates all events in a window, links ProxyOps to their parent Collectives/P2Ps,
 * calculates metrics, and exports them to OpenTelemetry. Called by the telemetry thread
 * when a window transitions to PROCESSING state.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] window_idx Index of the window to process (0-3).
 *
 * @note This function is called by the telemetry thread and is not thread-safe.
 * @note Windows are processed sequentially, one at a time.
 */
static void processWindow(CommunicatorState* commState, int window_idx)
{
    OTEL_TRACE(NCCL_INIT, "==> processWindow(window_idx=%d, rank=%d)", window_idx, commState->rank);

    WindowMetadata* window = commState->get_window_metadata(window_idx);

    // Create aggregator and process all events in the window
    WindowAggregator aggregator(commState->rank);

    // Get the buffer for this window
    otelEventHandle_t* buffer = commState->buffers[window_idx];
    uint32_t count            = window->element_count.load(std::memory_order_acquire);

    // Process all events in the buffer
    uint32_t skippedEvents = 0;
    for (uint32_t i = 0; i < count && i < BUFFER_SIZE; ++i)
    {
        const otelEventHandle_t& event = buffer[i];
        if (event.endTs > 0 && event.endTs >= event.startTs)
        {
            aggregator.addEvent(event);
        }
        else
        {
            skippedEvents++;
            OTEL_TRACE(NCCL_INIT, "Event %u type=%lu has endTs %f < startTs %f (skipped)", i, (unsigned long)event.type,
                       event.endTs, event.startTs);
        }
    }
    if (skippedEvents > 0)
    {
        OTEL_WARN(NCCL_INIT, "Window %u: skipped %u incomplete events", window_idx, skippedEvents);
    }

    // Finalize aggregation - calculates correct Coll/P2P durations based on ProxyOp completion
    aggregator.finalize();

    const bool scaleUpModeKnown = isScaleUpExecModeKnown(commState);

    // Use primer algorithm to fix issues with:
    // 1. Metrics completing within single window (Grafana sees no change)
    // 2. scale_up_exec_mode changing from UNKNOWN causing duplicate label series
    // 3. Zero transfer times not being exported causing missing Grafana series
    pthread_mutex_lock(&g_primerLock);

    // =======================================================================================
    // COLLECTIVE METRICS WITH PRIMER
    // =======================================================================================
    const auto& collectives = aggregator.getCollectives();
    std::set<std::string> exportedCollectives;

    // Phase 1: Process pending primers (from previous windows)
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
            exportedCollectives.insert(key);
        }

        if (primerData.state == PrimerState::PENDING_PRIMER)
        {
            // Check if we've exceeded maximum wait time - force emit to prevent indefinite waiting
            if (primerData.windowsWaited >= PRIMER_MAX_WAIT_WINDOWS)
            {
                std::string scaleUpMode = scaleUpModeKnown ? commState->getScaleUpExecModeString() : "unknown";
                exportCollectiveMetrics(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                        commState->local_rank, commState->comm_hash, commState->gpu_pci_bus_id,
                                        commState->gpu_uuid, commState->getCommTypeString(), commState->nranks,
                                        scaleUpMode, true);
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
                    exportCollectiveMetrics(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                            commState->local_rank, commState->comm_hash, commState->gpu_pci_bus_id,
                                            commState->gpu_uuid, commState->getCommTypeString(), commState->nranks,
                                            scaleUpMode, true);
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
                    exportCollectiveMetrics(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                            commState->local_rank, commState->comm_hash, commState->gpu_pci_bus_id,
                                            commState->gpu_uuid, commState->getCommTypeString(), commState->nranks,
                                            scaleUpMode, true);
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
            std::string scaleUpMode = commState->getScaleUpExecModeString();
            exportCollectiveMetrics(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                    commState->local_rank, commState->comm_hash, commState->gpu_pci_bus_id,
                                    commState->gpu_uuid, commState->getCommTypeString(), commState->nranks, scaleUpMode,
                                    false);
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

    // Phase 2: Process keys from current window (either new keys needing primers, or keys with primers already done)
    for (const auto& pair : collectives)
    {
        if (exportedCollectives.count(pair.first)) continue;  // Already handled

        PrimerKey pkey = {commState, pair.first};

        // Check if this key already completed its primer cycle - if so, export directly
        if (g_collectivePrimersDone.count(pkey))
        {
            std::string scaleUpMode = commState->getScaleUpExecModeString();
            exportCollectiveMetrics(pair.first, pair.second, commState->rank, commState->hostname,
                                    commState->local_rank, commState->comm_hash, commState->gpu_pci_bus_id,
                                    commState->gpu_uuid, commState->getCommTypeString(), commState->nranks, scaleUpMode,
                                    false);
            OTEL_TRACE(NCCL_INIT,
                       "Collective: exporting real data (primer already done, scale_up_exec_mode=%s: %s, count=%d)",
                       scaleUpMode.c_str(), pair.first.c_str(), pair.second.count);
            continue;
        }

        // New key - always start in PENDING_PRIMER state to allow stabilization
        g_collectivePrimers[pkey].aggregatedData = pair.second;
        g_collectivePrimers[pkey].state          = PrimerState::PENDING_PRIMER;
        g_collectivePrimers[pkey].windowsWaited  = 0;

        if (!scaleUpModeKnown)
        {
            OTEL_INFO(NCCL_INIT, "Collective NEW KEY: %s (scale_up_exec_mode UNKNOWN, waiting: count=%d)",
                      pair.first.c_str(), pair.second.count);
        }
        else
        {
            std::string scaleUpMode = commState->getScaleUpExecModeString();
            OTEL_INFO(NCCL_INIT,
                      "Collective NEW KEY: %s (scale_up_exec_mode=%s, starting %u-window stabilization: count=%d)",
                      pair.first.c_str(), scaleUpMode.c_str(), PRIMER_STABILIZATION_WINDOWS, pair.second.count);
        }
    }

    // =======================================================================================
    // P2P METRICS WITH PRIMER (same logic as Collective)
    // =======================================================================================
    const auto& p2ps = aggregator.getP2Ps();
    std::set<std::string> exportedP2Ps;

    // Phase 1: Process pending primers (from previous windows)
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
            exportedP2Ps.insert(key);
        }

        if (primerData.state == PrimerState::PENDING_PRIMER)
        {
            // Check if we've exceeded maximum wait time - force emit to prevent indefinite waiting
            if (primerData.windowsWaited >= PRIMER_MAX_WAIT_WINDOWS)
            {
                std::string scaleUpMode = scaleUpModeKnown ? commState->getScaleUpExecModeString() : "unknown";
                exportP2PMetrics(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                 commState->local_rank, commState->comm_hash, commState->gpu_pci_bus_id,
                                 commState->gpu_uuid, commState->getCommTypeString(), commState->nranks, scaleUpMode,
                                 true);
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
                // Once CUDA graphs are captured during warmup, they persist and the mode never
                // transitions back to NON_CUDA_GRAPH. Emitting immediately avoids unnecessary delay.
                if (scaleUpMode == std::string("cuda_graph"))
                {
                    exportP2PMetrics(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                     commState->local_rank, commState->comm_hash, commState->gpu_pci_bus_id,
                                     commState->gpu_uuid, commState->getCommTypeString(), commState->nranks,
                                     scaleUpMode, true);
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
                    exportP2PMetrics(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                     commState->local_rank, commState->comm_hash, commState->gpu_pci_bus_id,
                                     commState->gpu_uuid, commState->getCommTypeString(), commState->nranks,
                                     scaleUpMode, true);
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
            std::string scaleUpMode = commState->getScaleUpExecModeString();
            exportP2PMetrics(key, primerData.aggregatedData, commState->rank, commState->hostname,
                             commState->local_rank, commState->comm_hash, commState->gpu_pci_bus_id,
                             commState->gpu_uuid, commState->getCommTypeString(), commState->nranks, scaleUpMode,
                             false);
            OTEL_TRACE(NCCL_INIT, "P2P REAL DATA EXPORTED: %s (primer complete with scale_up_exec_mode=%s: count=%d)",
                       key.c_str(), scaleUpMode.c_str(), primerData.aggregatedData.count);
            g_p2pPrimersDone.insert(it->first);  // Mark as done
            it = g_p2pPrimers.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // Phase 2: Process keys from current window (either new keys needing primers, or keys with primers already done)
    for (const auto& pair : p2ps)
    {
        if (exportedP2Ps.count(pair.first)) continue;

        PrimerKey pkey = {commState, pair.first};

        // Check if this key already completed its primer cycle
        if (g_p2pPrimersDone.count(pkey))
        {
            exportP2PMetrics(pair.first, pair.second, commState->rank, commState->hostname, commState->local_rank,
                             commState->comm_hash, commState->gpu_pci_bus_id, commState->gpu_uuid,
                             commState->getCommTypeString(), commState->nranks, commState->getScaleUpExecModeString(),
                             false);
            continue;
        }

        // New key - always start in PENDING_PRIMER state to allow stabilization
        g_p2pPrimers[pkey].aggregatedData = pair.second;
        g_p2pPrimers[pkey].state          = PrimerState::PENDING_PRIMER;
        g_p2pPrimers[pkey].windowsWaited  = 0;

        if (!scaleUpModeKnown)
        {
            OTEL_INFO(NCCL_INIT, "P2P NEW KEY: %s (scale_up_exec_mode UNKNOWN, waiting: count=%d)", pair.first.c_str(),
                      pair.second.count);
        }
        else
        {
            std::string scaleUpMode = commState->getScaleUpExecModeString();
            OTEL_INFO(NCCL_INIT, "P2P NEW KEY: %s (scale_up_exec_mode=%s, starting %u-window stabilization: count=%d)",
                      pair.first.c_str(), scaleUpMode.c_str(), PRIMER_STABILIZATION_WINDOWS, pair.second.count);
        }
    }

    // =======================================================================================
    // RANK METRICS WITH PRIMER
    // =======================================================================================
    const auto& rankTransfers = aggregator.getRankTransfers();
    std::set<std::string> exportedRanks;

    // Phase 1: Process pending primers (from previous windows)
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
            exportedRanks.insert(key);
        }

        if (primerData.state == PrimerState::PENDING_PRIMER)
        {
            // Check if we've exceeded maximum wait time - force emit to prevent indefinite waiting
            if (primerData.windowsWaited >= PRIMER_MAX_WAIT_WINDOWS)
            {
                std::string scaleUpMode = scaleUpModeKnown ? commState->getScaleUpExecModeString() : "unknown";
                exportRankMetrics(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                  commState->gpu_pci_bus_id, commState->gpu_uuid, commState->getCommTypeString(),
                                  commState->nranks, commState->local_rank, scaleUpMode, true);
                primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                OTEL_INFO(
                    NCCL_INIT,
                    "Rank PRIMER FORCE-EMITTED: %s (max wait of %u windows exceeded, scale_up_exec_mode=%s, count=%d)",
                    key.c_str(), PRIMER_MAX_WAIT_WINDOWS, scaleUpMode.c_str(), primerData.aggregatedData.count);
                ++it;
            }
            else if (!scaleUpModeKnown)
            {
                primerData.windowsWaited++;
                OTEL_TRACE(NCCL_INIT,
                           "Rank PRIMER DELAYED: %s (scale_up_exec_mode still UNKNOWN, waited %u/%u windows: count=%d)",
                           key.c_str(), primerData.windowsWaited, PRIMER_MAX_WAIT_WINDOWS,
                           primerData.aggregatedData.count);
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
                    exportRankMetrics(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                      commState->gpu_pci_bus_id, commState->gpu_uuid, commState->getCommTypeString(),
                                      commState->nranks, commState->local_rank, scaleUpMode, true);
                    primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                    OTEL_TRACE(
                        NCCL_INIT,
                        "Rank PRIMER EMITTED: %s (zeros sent immediately with stable scale_up_exec_mode=%s, real "
                        "data on next window: count=%d)",
                        key.c_str(), scaleUpMode.c_str(), primerData.aggregatedData.count);
                    ++it;
                }
                // NON_CUDA_GRAPH might transition to CUDA_GRAPH during warmup - wait to stabilize
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
                    // Mode has been NON_CUDA_GRAPH for N windows - it's stable, emit primer
                    exportRankMetrics(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                      commState->gpu_pci_bus_id, commState->gpu_uuid, commState->getCommTypeString(),
                                      commState->nranks, commState->local_rank, scaleUpMode, true);
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
            std::string scaleUpMode = commState->getScaleUpExecModeString();
            exportRankMetrics(key, primerData.aggregatedData, commState->rank, commState->hostname,
                              commState->gpu_pci_bus_id, commState->gpu_uuid, commState->getCommTypeString(),
                              commState->nranks, commState->local_rank, scaleUpMode, false);
            OTEL_TRACE(NCCL_INIT, "Rank REAL DATA EXPORTED: %s (primer complete with scale_up_exec_mode=%s: count=%d)",
                       key.c_str(), scaleUpMode.c_str(), primerData.aggregatedData.count);
            g_rankPrimersDone.insert(it->first);  // Mark as done
            it = g_rankPrimers.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // Phase 2: Process keys from current window (either new keys needing primers, or keys with primers already done)
    for (const auto& pair : rankTransfers)
    {
        if (exportedRanks.count(pair.first)) continue;

        PrimerKey pkey = {commState, pair.first};

        // Check if this key already completed its primer cycle
        if (g_rankPrimersDone.count(pkey))
        {
            exportRankMetrics(pair.first, pair.second, commState->rank, commState->hostname, commState->gpu_pci_bus_id,
                              commState->gpu_uuid, commState->getCommTypeString(), commState->nranks,
                              commState->local_rank, commState->getScaleUpExecModeString(), false);
            continue;
        }

        // New key - always start in PENDING_PRIMER state to allow stabilization
        g_rankPrimers[pkey].aggregatedData = pair.second;
        g_rankPrimers[pkey].state          = PrimerState::PENDING_PRIMER;
        g_rankPrimers[pkey].windowsWaited  = 0;

        if (!scaleUpModeKnown)
        {
            OTEL_INFO(NCCL_INIT, "Rank NEW KEY: %s (scale_up_exec_mode UNKNOWN, waiting: count=%d)", pair.first.c_str(),
                      pair.second.count);
        }
        else
        {
            std::string scaleUpMode = commState->getScaleUpExecModeString();
            OTEL_INFO(NCCL_INIT, "Rank NEW KEY: %s (scale_up_exec_mode=%s, starting %u-window stabilization: count=%d)",
                      pair.first.c_str(), scaleUpMode.c_str(), PRIMER_STABILIZATION_WINDOWS, pair.second.count);
        }
    }

    // =======================================================================================
    // TRANSFER METRICS WITH PRIMER
    // =======================================================================================
    const auto& channelTransfers = aggregator.getChannelTransfers();
    std::set<std::string> exportedTransfers;

    // Phase 1: Process pending primers (from previous windows)
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
            exportedTransfers.insert(key);
        }

        if (primerData.state == PrimerState::PENDING_PRIMER)
        {
            // Check if we've exceeded maximum wait time - force emit to prevent indefinite waiting
            if (primerData.windowsWaited >= PRIMER_MAX_WAIT_WINDOWS)
            {
                std::string scaleUpMode = scaleUpModeKnown ? commState->getScaleUpExecModeString() : "unknown";
                exportTransferMetrics(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                      commState->gpu_pci_bus_id, commState->gpu_uuid, commState->getCommTypeString(),
                                      commState->nranks, commState->local_rank, scaleUpMode, true);
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
                OTEL_TRACE(
                    NCCL_INIT,
                    "Transfer PRIMER DELAYED: %s (scale_up_exec_mode still UNKNOWN, waited %u/%u windows: count=%d)",
                    key.c_str(), primerData.windowsWaited, PRIMER_MAX_WAIT_WINDOWS, primerData.aggregatedData.count);
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
                    exportTransferMetrics(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                          commState->gpu_pci_bus_id, commState->gpu_uuid,
                                          commState->getCommTypeString(), commState->nranks, commState->local_rank,
                                          scaleUpMode, true);
                    primerData.state = PrimerState::PRIMER_EMITTED_AWAITING_REAL;
                    OTEL_TRACE(NCCL_INIT,
                               "Transfer PRIMER EMITTED: %s (zeros sent immediately with stable scale_up_exec_mode=%s, "
                               "real data on next window: count=%d)",
                               key.c_str(), scaleUpMode.c_str(), primerData.aggregatedData.count);
                    ++it;
                }
                // NON_CUDA_GRAPH might transition to CUDA_GRAPH during warmup - wait to stabilize
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
                    // Mode has been NON_CUDA_GRAPH for N windows - it's stable, emit primer
                    exportTransferMetrics(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                          commState->gpu_pci_bus_id, commState->gpu_uuid,
                                          commState->getCommTypeString(), commState->nranks, commState->local_rank,
                                          scaleUpMode, true);
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
            std::string scaleUpMode = commState->getScaleUpExecModeString();
            exportTransferMetrics(key, primerData.aggregatedData, commState->rank, commState->hostname,
                                  commState->gpu_pci_bus_id, commState->gpu_uuid, commState->getCommTypeString(),
                                  commState->nranks, commState->local_rank, scaleUpMode, false);
            OTEL_TRACE(NCCL_INIT,
                       "Transfer REAL DATA EXPORTED: %s (primer complete with scale_up_exec_mode=%s: count=%d)",
                       key.c_str(), scaleUpMode.c_str(), primerData.aggregatedData.count);
            g_transferPrimersDone.insert(it->first);  // Mark as done
            it = g_transferPrimers.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // Phase 2: Process keys from current window (either new keys needing primers, or keys with primers already done)
    for (const auto& pair : channelTransfers)
    {
        if (exportedTransfers.count(pair.first)) continue;

        PrimerKey pkey = {commState, pair.first};

        // Check if this key already completed its primer cycle
        if (g_transferPrimersDone.count(pkey))
        {
            exportTransferMetrics(pair.first, pair.second, commState->rank, commState->hostname,
                                  commState->gpu_pci_bus_id, commState->gpu_uuid, commState->getCommTypeString(),
                                  commState->nranks, commState->local_rank, commState->getScaleUpExecModeString(),
                                  false);
            continue;
        }
        // New key - always start in PENDING_PRIMER state to allow stabilization
        g_transferPrimers[pkey].aggregatedData = pair.second;
        g_transferPrimers[pkey].state          = PrimerState::PENDING_PRIMER;
        g_transferPrimers[pkey].windowsWaited  = 0;

        if (!scaleUpModeKnown)
        {
            OTEL_INFO(NCCL_INIT, "Transfer NEW KEY: %s (scale_up_exec_mode UNKNOWN, waiting: count=%d)",
                      pair.first.c_str(), pair.second.count);
        }
        else
        {
            std::string scaleUpMode = commState->getScaleUpExecModeString();
            OTEL_INFO(NCCL_INIT,
                      "Transfer NEW KEY: %s (scale_up_exec_mode=%s, starting %u-window stabilization: count=%d)",
                      pair.first.c_str(), scaleUpMode.c_str(), PRIMER_STABILIZATION_WINDOWS, pair.second.count);
        }
    }

    pthread_mutex_unlock(&g_primerLock);

    // Transition window back to READY state
    window->state.store(WINDOW_READY, std::memory_order_release);
    window->element_count.store(0, std::memory_order_release);
    window->in_progress_count.store(0, std::memory_order_release);

    OTEL_INFO(NCCL_INIT, "Window %d processed: %zu collectives, %zu P2Ps, %zu rank-transfers, %zu channel-transfers",
              window_idx, collectives.size(), p2ps.size(), rankTransfers.size(), channelTransfers.size());
    OTEL_TRACE(NCCL_INIT, "<== processWindow()");
}

#endif  // ENABLE_OTEL

/**
 * @brief Main function for the telemetry background thread.
 *
 * Runs in a loop, periodically checking for windows in PROCESSING state and
 * processing them. Uses condition variable with timeout for interruptible waiting.
 *
 * @param[in] arg Unused thread argument (nullptr).
 *
 * @return nullptr on exit.
 *
 * @note Thread exits when g_telThreadStop is set to true.
 * @note Processes windows at intervals specified by NCCL_PROFILER_OTEL_TELEMETRY_INTERVAL_SEC.
 */
static void* profiler_otel_telemetry_thread_main(void*)
{
    OTEL_TRACE(NCCL_INIT, "==> profiler_otel_telemetry_thread_main()");

    int interval = (int)OTEL_GET_PARAM(TelemetryIntervalSec);
    if (interval <= 0)
    {
        interval = 5;  // Default to 5 seconds
    }

    OTEL_INFO(NCCL_INIT, "Telemetry thread started (interval: %ds)", interval);

    while (!g_telThreadStop.load(std::memory_order_acquire))
    {
        // Use condition variable with timeout for interruptible wait
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += interval;

        pthread_mutex_lock(&g_telLock);
        int ret = pthread_cond_timedwait(&g_telCond, &g_telLock, &ts);
        pthread_mutex_unlock(&g_telLock);

        // If we are signaled (ret == 0), check if we should exit
        if (ret == 0 && g_telThreadStop.load(std::memory_order_acquire))
        {
            OTEL_TRACE(NCCL_INIT, "Telemetry thread exiting due to stop signal");
            break;
        }

        // Process windows that are ready for export
#ifdef ENABLE_OTEL
        pthread_mutex_lock(&g_commStatesLock);
        for (auto* commState : g_commStates)
        {
            if (!commState) continue;

            // Check all windows for PROCESSING state
            for (int i = 0; i < NUM_BUFFERS; ++i)
            {
                WindowMetadata* window = commState->get_window_metadata(i);
                if (!window) continue;
                WindowState state = window->state.load(std::memory_order_acquire);
                if (state == WINDOW_PROCESSING)
                {
                    OTEL_TRACE(NCCL_INIT, "Processing window %d for comm %s", i, commState->commName.c_str());
                    processWindow(commState, i);
                }
            }
        }
        pthread_mutex_unlock(&g_commStatesLock);
#endif
    }

    OTEL_TRACE(NCCL_INIT, "<== profiler_otel_telemetry_thread_main() -> thread exiting");
    return nullptr;
}

/**
 * @brief Initialize the telemetry collection system.
 *
 * Sets up OpenTelemetry metrics, creates metric instruments, and starts the background
 * telemetry thread for asynchronous metric processing and export.
 *
 * @note Only initializes if NCCL_PROFILER_OTEL_TELEMETRY_ENABLE is set (default: 1).
 * @note Called automatically on first communicator initialization.
 * @note Thread-safe: uses atomic counters to ensure single initialization.
 */
void profiler_otel_telemetry_init()
{
    OTEL_TRACE(NCCL_INIT, "==> profiler_otel_telemetry_init()");

    if (!OTEL_GET_PARAM(TelemetryEnable))
    {
        OTEL_INFO(NCCL_INIT, "Telemetry disabled by NCCL_PROFILER_OTEL_TELEMETRY_ENABLE");
        return;
    }

#ifdef ENABLE_OTEL
    // Initialize OpenTelemetry metrics
    initializeOtelMetrics();
#else
    OTEL_WARN(NCCL_INIT, "OpenTelemetry not enabled at compile time. Telemetry will not export metrics.");
#endif

    // Start telemetry thread
    g_telThreadStop.store(false, std::memory_order_release);
    int rc = pthread_create(&g_telThread, nullptr, profiler_otel_telemetry_thread_main, nullptr);
    if (rc != 0)
    {
        OTEL_WARN(NCCL_INIT, "Failed to create telemetry thread: %d", rc);
    }
    else
    {
        OTEL_INFO(NCCL_INIT, "Telemetry thread created successfully");
    }

    OTEL_TRACE(NCCL_INIT, "<== profiler_otel_telemetry_init()");
}

/**
 * @brief Cleanup the telemetry collection system.
 *
 * Stops the telemetry thread, cleans up OpenTelemetry resources, and resets state.
 * Called when the last communicator is finalized.
 *
 * @note Only cleans up if telemetry was initialized.
 * @note Thread-safe: uses atomic counters to track active communicators.
 */
void profiler_otel_telemetry_cleanup()
{
    OTEL_TRACE(NCCL_INIT, "==> profiler_otel_telemetry_cleanup()");

    if (!OTEL_GET_PARAM(TelemetryEnable))
    {
        return;
    }

    // Signal thread to stop
    g_telThreadStop.store(true, std::memory_order_release);

    // Wake up the thread
    pthread_mutex_lock(&g_telLock);
    pthread_cond_signal(&g_telCond);
    pthread_mutex_unlock(&g_telLock);

    // Wait for thread to exit
    pthread_join(g_telThread, nullptr);

    OTEL_INFO(NCCL_INIT, "Telemetry thread stopped");

    // Clear communicator registration list (pointers become invalid after finalization).
    pthread_mutex_lock(&g_commStatesLock);
    g_commStates.clear();
    pthread_mutex_unlock(&g_commStatesLock);

#ifdef ENABLE_OTEL
    // Cleanup OpenTelemetry resources
    g_collBytesCounter.reset();
    g_collTimeHist.reset();
    g_collCountHist.reset();
    g_collNumTransfersHist.reset();
    g_collTransferSizeHist.reset();
    g_collTransferTimeHist.reset();
    g_p2pBytesHist.reset();
    g_p2pTimeHist.reset();
    g_p2pNumTransfersHist.reset();
    g_p2pTransferSizeHist.reset();
    g_p2pTransferTimeHist.reset();
    g_rankBytesCounter.reset();
    g_rankLatencyHist.reset();
    g_rankRateHist.reset();
    g_transferSizeHist.reset();
    g_transferTimeHist.reset();
    g_transferLatencyHist.reset();
    // Note: nostd::shared_ptr doesn't have reset(), just set to nullptr
    g_meter         = nullptr;
    g_meterProvider = nullptr;
#endif

    OTEL_TRACE(NCCL_INIT, "<== profiler_otel_telemetry_cleanup()");
}

/**
 * @brief Notify the telemetry thread that a window is ready for processing.
 *
 * Called when a window transitions to PROCESSING state. Registers the communicator
 * state if not already registered and wakes up the telemetry thread to process the window.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] window_idx Index of the window that is ready (0-3).
 *
 * @note Thread-safe: uses mutexes for communicator state registration.
 */
void profiler_otel_telemetry_notify_window_ready(struct CommunicatorState* commState, int window_idx)
{
    OTEL_TRACE(NCCL_INIT, "Window %d ready for processing", window_idx);
    (void)window_idx;  // Suppress unused parameter warning when TRACE is disabled

    // Register the communicator state if not already registered
    pthread_mutex_lock(&g_commStatesLock);
    bool found = false;
    for (auto* cs : g_commStates)
    {
        if (cs == commState)
        {
            found = true;
            break;
        }
    }
    if (!found)
    {
        g_commStates.push_back(commState);
        OTEL_INFO(NCCL_INIT, "Registered communicator %s for telemetry", commState->commName.c_str());
    }
    pthread_mutex_unlock(&g_commStatesLock);

    // Wake up telemetry thread
    pthread_mutex_lock(&g_telLock);
    pthread_cond_signal(&g_telCond);
    pthread_mutex_unlock(&g_telLock);
}
