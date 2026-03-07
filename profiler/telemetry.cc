// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "telemetry.h"

#include <pthread.h>

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
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

/**
 * @brief Export collective operation metrics to OpenTelemetry.
 *
 * Exports aggregated collective metrics including bytes, time, transfer counts,
 * transfer sizes, and transfer times. All metrics include communicator, rank,
 * hostname, and local_rank labels.
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
 */
static void exportCollectiveMetrics(const std::string& key, const AggregatedCollective& coll, int rank,
                                    const std::string& hostname, int local_rank, uint64_t comm_hash,
                                    const std::string& gpu_pci_bus_id, const std::string& gpu_uuid,
                                    const std::string& comm_type, int nranks, const std::string& scale_up_exec_mode)
{
    // Parse key to extract collective name, algo, proto, nChannels
    // Key format: Comm<hash>_<func>_<algo>_<proto>_<nChannels>Chnl
    std::string rank_str       = std::to_string(rank);
    std::string local_rank_str = std::to_string(local_rank);
    std::string communicator   = std::to_string(comm_hash);
    std::string coll_name      = key;
    std::string nchannels      = "unknown";

    // Parse the key (simplified for now - should be enhanced)
    size_t last_underscore = key.rfind('_');
    if (last_underscore != std::string::npos)
    {
        coll_name = key.substr(0, last_underscore);
        nchannels = key.substr(last_underscore + 1);
    }

    // Export collective bytes and time
    if (coll.count > 0)
    {
        double avgTime = coll.getAverageTime();

        OTEL_TRACE(
            NCCL_INIT,
            "Exporting Collective: %s, count=%u, totalBytes=%zu, totalTime=%.2f us -> AvgBytes=%.2f, AvgTime=%.2f us",
            key.c_str(), coll.count, coll.totalBytes, coll.totalTimeUs, coll.getAverageSize(), avgTime);

        // Create attributes for labeling
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

        g_collBytesCounter->Add(coll.totalBytes, labels, opentelemetry::context::Context{});
        g_collTimeHist->Record(avgTime, labels, opentelemetry::context::Context{});
        g_collCountHist->Record((double)coll.count, labels, opentelemetry::context::Context{});

        // Export transfer statistics (from underlying proxy operations or scale-up inference)
        double avgNumTransfers = coll.getAverageTransferCount();
        double avgTransferSize = coll.getAverageTransferSize();
        double avgTransferTime = coll.getAverageTransferTime();

        if (avgNumTransfers > 0)
        {
            g_collNumTransfersHist->Record(avgNumTransfers, labels, opentelemetry::context::Context{});
            g_collTransferSizeHist->Record(avgTransferSize, labels, opentelemetry::context::Context{});
            // Only export transfer time when aggregation produced real timing data.
            // CUDA-graph scale-up volume-only paths intentionally leave this at 0,
            // while scale-out proxy paths still carry actual ProxyStep timing even
            // if the communicator is tagged as cuda_graph.
            if (coll.cachedTotalTransferTimeUs > 0.0)
            {
                g_collTransferTimeHist->Record(avgTransferTime, labels, opentelemetry::context::Context{});
            }

            OTEL_TRACE(NCCL_INIT,
                       "Exported Collective: %s, AvgBytes: %.2f, AvgTime: %.2f us, "
                       "AvgNumTransfers: %.2f, AvgTransferSize: %.2f, AvgTransferTime: %.2f us",
                       key.c_str(), coll.getAverageSize(), avgTime, avgNumTransfers, avgTransferSize, avgTransferTime);
        }
        else
        {
            OTEL_TRACE(NCCL_INIT, "Exported Collective: %s, AvgBytes: %.2f, AvgTime: %.2f us (no transfers)",
                       key.c_str(), coll.getAverageSize(), avgTime);
        }
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
static void exportP2PMetrics(const std::string& key, const AggregatedP2P& p2p, int rank, const std::string& hostname,
                             int local_rank, uint64_t comm_hash, const std::string& gpu_pci_bus_id,
                             const std::string& gpu_uuid, const std::string& comm_type, int nranks,
                             const std::string& scale_up_exec_mode)
{
    // Parse key to extract function name and pipeline info
    // P2P Key format: Comm<hash>_(<hostname>)_<func>_Pipeline<src>ToPipeline<dst>_<nChannels>Chnl
    std::string rank_str       = std::to_string(rank);
    std::string local_rank_str = std::to_string(local_rank);
    std::string communicator   = std::to_string(comm_hash);

    // Extract function name (Send/Recv) and pipeline info from key
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
            else
            {
                dst_pipeline = key.substr(dst_start);
            }
        }
    }

    // Build operation label with rank info from local_rank
    // local_rank is now correctly set from the GPU ID → rank map (populated by COLLECTIVE comms)
    std::string rank_info = " (Rank" + local_rank_str + ")";
    std::string operation = func_name + " Pipeline" + src_pipeline + "→Pipeline" + dst_pipeline + rank_info;

    // Export P2P bytes and time
    if (p2p.count > 0)
    {
        double avgBytes = (double)p2p.totalBytes / p2p.count;
        double avgTime  = p2p.totalTimeUs / p2p.count;

        // Create attributes for labeling
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

        g_p2pBytesHist->Record(avgBytes, labels, opentelemetry::context::Context{});
        g_p2pTimeHist->Record(avgTime, labels, opentelemetry::context::Context{});

        // Export transfer statistics (from underlying proxy operations)
        double avgNumTransfers = p2p.getAverageTransferCount();
        double avgTransferSize = p2p.getAverageTransferSize();
        double avgTransferTime = p2p.getAverageTransferTime();

        if (avgNumTransfers > 0)
        {
            g_p2pNumTransfersHist->Record(avgNumTransfers, labels, opentelemetry::context::Context{});
            g_p2pTransferSizeHist->Record(avgTransferSize, labels, opentelemetry::context::Context{});
            // Same policy as collectives: suppress only when no timing data exists.
            if (p2p.cachedTotalTransferTimeUs > 0.0)
            {
                g_p2pTransferTimeHist->Record(avgTransferTime, labels, opentelemetry::context::Context{});
            }

            OTEL_TRACE(NCCL_INIT,
                       "Exported P2P: %s, AvgBytes: %.2f, AvgTime: %.2f us, "
                       "AvgNumTransfers: %.2f, AvgTransferSize: %.2f, AvgTransferTime: %.2f us",
                       key.c_str(), avgBytes, avgTime, avgNumTransfers, avgTransferSize, avgTransferTime);
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
 * @param[in] key Aggregation key in format: Comm<hash>_Rank<X>ToRank<Y>
 * @param[in] transferRef Aggregated transfer data containing statistics.
 * @param[in] rank Global rank of the process (unused, extracted from key).
 * @param[in] hostname Hostname of the node.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string (tensor_parallel, pipeline_parallel, unknown).
 * @param[in] nranks Number of ranks in the communicator.
 */
static void exportRankMetrics(const std::string& key, const AggregatedTransfer& transferRef, int rank,
                              const std::string& hostname, const std::string& gpu_pci_bus_id,
                              const std::string& gpu_uuid, const std::string& comm_type, int nranks, int local_rank,
                              const std::string& scale_up_exec_mode)
{
    (void)rank;  // Unused - rank info is parsed from key or derived from local_rank
    std::string communicator = "";
    std::string source_rank  = "";
    std::string dest_rank    = "";

    // Parse key based on format:
    // P2P: Comm<hash>_<hostname>_Pipeline<src>_ToPipeline<peer>
    // COLLECTIVE: Comm<hash>_Rank<X>_ToPeer<peer>
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
        // P2P format: Parse Pipeline<src>_ToPipeline<dst>
        // Include the actual GPU rank (local_rank) for clarity
        size_t src_start = pipeline_pos + 9;  // After "_Pipeline"
        size_t to_pos    = key.find("_ToPipeline", src_start);
        if (to_pos != std::string::npos)
        {
            std::string src_pipeline = key.substr(src_start, to_pos - src_start);
            std::string dst_pipeline = key.substr(to_pos + 11);
            // Format: "Pipeline0 (Rank5)" to show both pipeline and GPU rank
            source_rank = "Pipeline" + src_pipeline + " (Rank" + std::to_string(local_rank) + ")";
            dest_rank   = "Pipeline" + dst_pipeline;
        }
    }
    else if (peer_pos != std::string::npos)
    {
        // COLLECTIVE format: parse source rank from key
        size_t rank_pos = key.find("_Rank");
        if (rank_pos != std::string::npos)
        {
            source_rank = key.substr(rank_pos + 5, peer_pos - rank_pos - 5);
        }
        dest_rank = key.substr(peer_pos + 7);
    }

    // Create attributes for labeling
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

    // Export rank bytes
    g_rankBytesCounter->Add(transferRef.totalBytes, labels, opentelemetry::context::Context{});

    // Export rank latency (from linear regression)
    double latencyUs;
    if (transferRef.getLatencyFromLinearRegression(latencyUs))
    {
        g_rankLatencyHist->Record(latencyUs, labels, opentelemetry::context::Context{});
        OTEL_TRACE(NCCL_INIT, "Exported Rank Latency: %s, Latency: %.2f us", key.c_str(), latencyUs);
    }

    // Export rank rate (from active transfer time - merged intervals)
    // This represents the actual bandwidth between ranks by measuring total bytes
    // transferred divided by the time during which at least one transfer was active.
    double rateMBps;
    if (transferRef.getRateFromActiveTime(rateMBps))
    {
        g_rankRateHist->Record(rateMBps, labels, opentelemetry::context::Context{});
        OTEL_TRACE(NCCL_INIT, "Exported Rank Rate: %s, Bytes: %zu, ActiveTime: %.2f us, Rate: %.2f MB/s", key.c_str(),
                   transferRef.totalBytes, transferRef.getActiveTime(), rateMBps);
    }
    else
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
static void exportTransferMetrics(const std::string& key, const AggregatedTransfer& transferRef, int rank,
                                  const std::string& hostname, const std::string& gpu_pci_bus_id,
                                  const std::string& gpu_uuid, const std::string& comm_type, int nranks, int local_rank,
                                  const std::string& scale_up_exec_mode)
{
    (void)rank;  // Unused - rank info is parsed from key or derived from local_rank
    std::string communicator = "";
    std::string source_rank  = "";
    std::string dest_rank    = "";
    std::string channel      = "";

    // Parse key based on format:
    // P2P: Comm<hash>_<hostname>_Pipeline<src>_ToPipeline<peer>_Chnl<id>
    // COLLECTIVE: Comm<hash>_Rank<X>_ToPeer<peer>_Chnl<id>
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
        // P2P format: Parse Pipeline<src>_ToPipeline<dst>
        // Include the actual GPU rank (local_rank) for clarity
        size_t src_start = pipeline_pos + 9;  // After "_Pipeline"
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
            // Format: "Pipeline0 (Rank5)" to show both pipeline and GPU rank
            source_rank = "Pipeline" + src_pipeline + " (Rank" + std::to_string(local_rank) + ")";
            dest_rank   = "Pipeline" + dst_pipeline;
        }
    }
    else if (peer_pos != std::string::npos)
    {
        // COLLECTIVE format: ranks within communicator
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

    if (transferRef.count > 0)
    {
        double avgSize = transferRef.getAverageSize();
        double avgTime = transferRef.getAverageTime();

        // Create attributes for labeling
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

        g_transferSizeHist->Record(avgSize, labels, opentelemetry::context::Context{});
        if (transferRef.totalTimeUs > 0.0)
        {
            g_transferTimeHist->Record(avgTime, labels, opentelemetry::context::Context{});
        }

        // Export transfer latency (from linear regression)
        double latencyUs;
        if (transferRef.getLatencyFromLinearRegression(latencyUs))
        {
            g_transferLatencyHist->Record(latencyUs, labels, opentelemetry::context::Context{});

            OTEL_TRACE(NCCL_INIT, "Exported Transfer: %s, AvgSize: %.2f, AvgTime: %.2f us, Latency: %.2f us",
                       key.c_str(), avgSize, avgTime, latencyUs);
        }
        else
        {
            OTEL_TRACE(NCCL_INIT, "Exported Transfer: %s, AvgSize: %.2f, AvgTime: %.2f us", key.c_str(), avgSize,
                       avgTime);
        }
    }
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

    // Export Collective Information
    const auto& collectives = aggregator.getCollectives();
    for (const auto& pair : collectives)
    {
        exportCollectiveMetrics(pair.first, pair.second, commState->rank, commState->hostname, commState->local_rank,
                                commState->comm_hash, commState->gpu_pci_bus_id, commState->gpu_uuid,
                                commState->getCommTypeString(), commState->nranks,
                                commState->getScaleUpExecModeString());
    }

    // Export P2P Information
    const auto& p2ps = aggregator.getP2Ps();
    for (const auto& pair : p2ps)
    {
        exportP2PMetrics(pair.first, pair.second, commState->rank, commState->hostname, commState->local_rank,
                         commState->comm_hash, commState->gpu_pci_bus_id, commState->gpu_uuid,
                         commState->getCommTypeString(), commState->nranks, commState->getScaleUpExecModeString());
    }

    // Export Rank Information
    const auto& rankTransfers = aggregator.getRankTransfers();
    for (const auto& pair : rankTransfers)
    {
        exportRankMetrics(pair.first, pair.second, commState->rank, commState->hostname, commState->gpu_pci_bus_id,
                          commState->gpu_uuid, commState->getCommTypeString(), commState->nranks, commState->local_rank,
                          commState->getScaleUpExecModeString());
    }

    // Export Transfer Information
    const auto& channelTransfers = aggregator.getChannelTransfers();
    for (const auto& pair : channelTransfers)
    {
        exportTransferMetrics(pair.first, pair.second, commState->rank, commState->hostname, commState->gpu_pci_bus_id,
                              commState->gpu_uuid, commState->getCommTypeString(), commState->nranks,
                              commState->local_rank, commState->getScaleUpExecModeString());
    }

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
