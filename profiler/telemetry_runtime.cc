// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <pthread.h>
#include <time.h>

#include <atomic>
#include <chrono>
#include <string>
#include <vector>

#ifdef __linux__
#include <sys/prctl.h>
#endif

#include "communicator_state.h"
#include "param.h"
#include "profiler_otel.h"
#include "telemetry_internal.h"
#include "telemetry_primer.h"

// OpenTelemetry includes - only include if available
#ifdef ENABLE_OTEL
#include <opentelemetry/exporters/otlp/otlp_http_metric_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_http_metric_exporter_options.h>
#include <opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader.h>
#include <opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader_factory.h>
#include <opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader_options.h>
#include <opentelemetry/sdk/metrics/meter_provider.h>

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

static std::atomic<bool> g_telThreadStop{false};
static pthread_t g_telThread;
static pthread_mutex_t g_telLock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_telCond  = PTHREAD_COND_INITIALIZER;

static std::vector<CommunicatorState*> g_commStates;
static pthread_mutex_t g_commStatesLock = PTHREAD_MUTEX_INITIALIZER;

#ifdef ENABLE_OTEL
nostd::shared_ptr<metrics_api::MeterProvider> g_meterProvider;
nostd::shared_ptr<metrics_api::Meter> g_meter;

nostd::unique_ptr<metrics_api::Counter<uint64_t>> g_collBytesCounter;
nostd::unique_ptr<metrics_api::Histogram<double>> g_collTimeHist;
nostd::unique_ptr<metrics_api::Histogram<double>> g_collCountHist;
nostd::unique_ptr<metrics_api::Histogram<double>> g_collNumTransfersHist;
nostd::unique_ptr<metrics_api::Histogram<double>> g_collTransferSizeHist;
nostd::unique_ptr<metrics_api::Histogram<double>> g_collTransferTimeHist;

nostd::unique_ptr<metrics_api::Histogram<double>> g_p2pBytesHist;
nostd::unique_ptr<metrics_api::Histogram<double>> g_p2pTimeHist;
nostd::unique_ptr<metrics_api::Histogram<double>> g_p2pNumTransfersHist;
nostd::unique_ptr<metrics_api::Histogram<double>> g_p2pTransferSizeHist;
nostd::unique_ptr<metrics_api::Histogram<double>> g_p2pTransferTimeHist;

nostd::unique_ptr<metrics_api::Counter<uint64_t>> g_rankBytesCounter;
nostd::unique_ptr<metrics_api::Histogram<double>> g_rankLatencyHist;
nostd::unique_ptr<metrics_api::Histogram<double>> g_rankRateHist;

nostd::unique_ptr<metrics_api::Histogram<double>> g_transferSizeHist;
nostd::unique_ptr<metrics_api::Histogram<double>> g_transferTimeHist;
nostd::unique_ptr<metrics_api::Histogram<double>> g_transferLatencyHist;

/**
 * @brief Build the configured OTLP endpoint URL prefix.
 *
 * @return Configured telemetry endpoint base URL.
 */
static std::string getTelemetryEndpoint()
{
    return std::string(ncclParamTelemetryEndpoint());
}

/**
 * @brief Initialize the OpenTelemetry exporter, meter provider, and instruments.
 */
static void initializeOtelMetrics()
{
    OTEL_TRACE(NCCL_INIT, "==> initializeOtelMetrics()");

    otlp::OtlpHttpMetricExporterOptions exporterOptions;
    std::string endpoint    = getTelemetryEndpoint();
    exporterOptions.url     = endpoint + "/v1/metrics";
    exporterOptions.timeout = std::chrono::milliseconds(OTEL_GET_PARAM(TelemetryOtelBatchTimeoutMs));

    OTEL_INFO(NCCL_INIT, "OpenTelemetry endpoint: %s", exporterOptions.url.c_str());

    auto exporter = otlp::OtlpHttpMetricExporterFactory::Create(exporterOptions);

    sdk_metrics::PeriodicExportingMetricReaderOptions readerOptions;
    readerOptions.export_interval_millis = std::chrono::milliseconds(OTEL_GET_PARAM(TelemetryIntervalSec) * 1000);
    readerOptions.export_timeout_millis  = std::chrono::milliseconds(OTEL_GET_PARAM(TelemetryOtelBatchTimeoutMs));

    auto reader = sdk_metrics::PeriodicExportingMetricReaderFactory::Create(std::move(exporter), readerOptions);

    auto sdk_provider = std::unique_ptr<sdk_metrics::MeterProvider>(new sdk_metrics::MeterProvider());
    sdk_provider->AddMetricReader(std::move(reader));
    g_meterProvider = nostd::shared_ptr<metrics_api::MeterProvider>(sdk_provider.release());

    g_meter = g_meterProvider->GetMeter("nccl_profiler", "1.0.0");

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

    g_p2pBytesHist =
        g_meter->CreateDoubleHistogram("nccl_profiler_p2p_bytes", "Average bytes per P2P operation", "bytes");
    g_p2pTimeHist = g_meter->CreateDoubleHistogram("nccl_profiler_p2p_time", "Average time per P2P operation", "us");
    g_p2pNumTransfersHist = g_meter->CreateDoubleHistogram("nccl_profiler_p2p_num_transfers",
                                                           "Average number of transfers per P2P", "count");
    g_p2pTransferSizeHist =
        g_meter->CreateDoubleHistogram("nccl_profiler_p2p_transfer_size", "Average transfer size for P2P", "bytes");
    g_p2pTransferTimeHist =
        g_meter->CreateDoubleHistogram("nccl_profiler_p2p_transfer_time", "Average transfer time for P2P", "us");

    g_rankBytesCounter =
        g_meter->CreateUInt64Counter("nccl_profiler_rank_bytes", "Bytes sent from rank to rank", "bytes");
    g_rankLatencyHist = g_meter->CreateDoubleHistogram("nccl_profiler_rank_latency",
                                                       "Latency from rank to rank (from linear regression)", "us");
    g_rankRateHist    = g_meter->CreateDoubleHistogram(
        "nccl_profiler_rank_rate", "Transfer rate from rank to rank (bandwidth based on active transfer time)", "MB/s");

    g_transferSizeHist =
        g_meter->CreateDoubleHistogram("nccl_profiler_transfer_size", "Average transfer size per channel", "bytes");
    g_transferTimeHist =
        g_meter->CreateDoubleHistogram("nccl_profiler_transfer_time", "Average transfer time per channel", "us");
    g_transferLatencyHist = g_meter->CreateDoubleHistogram(
        "nccl_profiler_transfer_latency", "Transfer latency per channel (from linear regression)", "us");

    OTEL_INFO(NCCL_INIT, "OpenTelemetry metrics initialized");
    OTEL_TRACE(NCCL_INIT, "<== initializeOtelMetrics()");
}

#endif  // ENABLE_OTEL

/**
 * @brief Run the background telemetry thread that drains ready windows.
 *
 * @param[in] Unused thread entry argument.
 *
 * @return Always returns nullptr when the thread exits.
 */
static void* profiler_otel_telemetry_thread_main(void*)
{
    OTEL_TRACE(NCCL_INIT, "==> profiler_otel_telemetry_thread_main()");

#ifdef __linux__
    (void)prctl(PR_SET_NAME, "nccl-prof-tel", 0, 0, 0);
#endif

    int interval = (int)OTEL_GET_PARAM(TelemetryIntervalSec);
    if (interval <= 0)
    {
        interval = 5;
    }

    OTEL_INFO(NCCL_INIT, "Telemetry thread started (interval: %ds)", interval);

    while (!g_telThreadStop.load(std::memory_order_acquire))
    {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += interval;

        pthread_mutex_lock(&g_telLock);
        int ret = pthread_cond_timedwait(&g_telCond, &g_telLock, &ts);
        pthread_mutex_unlock(&g_telLock);

        if (ret == 0 && g_telThreadStop.load(std::memory_order_acquire))
        {
            OTEL_TRACE(NCCL_INIT, "Telemetry thread exiting due to stop signal");
            break;
        }

#ifdef ENABLE_OTEL
        pthread_mutex_lock(&g_commStatesLock);
        for (CommunicatorState* commState : g_commStates)
        {
            if (!commState) continue;

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
 * @brief Initialize telemetry runtime state and spawn the telemetry thread.
 */
void telemetryRuntimeInit()
{
    OTEL_TRACE(NCCL_INIT, "==> profiler_otel_telemetry_init()");

    if (!OTEL_GET_PARAM(TelemetryEnable))
    {
        OTEL_INFO(NCCL_INIT, "Telemetry disabled by NCCL_PROFILER_OTEL_TELEMETRY_ENABLE");
        return;
    }

#ifdef ENABLE_OTEL
    initializeOtelMetrics();
#else
    OTEL_WARN(NCCL_INIT, "OpenTelemetry not enabled at compile time. Telemetry will not export metrics.");
#endif

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
 * @brief Stop the telemetry thread and release runtime telemetry resources.
 */
void telemetryRuntimeCleanup()
{
    OTEL_TRACE(NCCL_INIT, "==> profiler_otel_telemetry_cleanup()");

    if (!OTEL_GET_PARAM(TelemetryEnable))
    {
        return;
    }

    g_telThreadStop.store(true, std::memory_order_release);

    pthread_mutex_lock(&g_telLock);
    pthread_cond_signal(&g_telCond);
    pthread_mutex_unlock(&g_telLock);

    pthread_join(g_telThread, nullptr);

    OTEL_INFO(NCCL_INIT, "Telemetry thread stopped");

    pthread_mutex_lock(&g_commStatesLock);
    g_commStates.clear();
    pthread_mutex_unlock(&g_commStatesLock);

#ifdef ENABLE_OTEL
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
    g_meter         = nullptr;
    g_meterProvider = nullptr;
#endif

    OTEL_TRACE(NCCL_INIT, "<== profiler_otel_telemetry_cleanup()");
}

/**
 * @brief Register a communicator for telemetry processing and wake the worker thread.
 *
 * @param[in] commState Communicator state owning the ready window.
 * @param[in] window_idx Ready window index.
 */
void telemetryRuntimeNotifyWindowReady(CommunicatorState* commState, int window_idx)
{
    OTEL_TRACE(NCCL_INIT, "Window %d ready for processing", window_idx);
    (void)window_idx;

    pthread_mutex_lock(&g_commStatesLock);
    bool found = false;
    for (CommunicatorState* cs : g_commStates)
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

    pthread_mutex_lock(&g_telLock);
    pthread_cond_signal(&g_telCond);
    pthread_mutex_unlock(&g_telLock);
}

/**
 * @brief Unregister a communicator from telemetry processing.
 *
 * @param[in] commState Communicator state to remove.
 */
void telemetryRuntimeUnregisterCommunicator(CommunicatorState* commState)
{
    if (!commState)
    {
        return;
    }

    pthread_mutex_lock(&g_commStatesLock);
    for (auto it = g_commStates.begin(); it != g_commStates.end(); ++it)
    {
        if (*it == commState)
        {
            g_commStates.erase(it);
            break;
        }
    }
#ifdef ENABLE_OTEL
    cleanupTelemetryPrimerStateForCommunicator(commState);
#endif
    pthread_mutex_unlock(&g_commStatesLock);
}