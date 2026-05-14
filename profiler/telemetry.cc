// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "telemetry.h"

#include "telemetry_internal.h"

/**
 * @brief Initialize the profiler telemetry subsystem.
 */
void profiler_otel_telemetry_init()
{
    telemetryRuntimeInit();
}

/**
 * @brief Shut down the profiler telemetry subsystem.
 */
void profiler_otel_telemetry_cleanup()
{
    telemetryRuntimeCleanup();
}

/**
 * @brief Notify the telemetry runtime that a communicator window is ready.
 *
 * @param[in] commState Communicator state owning the ready window.
 * @param[in] window_idx Ready window index.
 */
void profiler_otel_telemetry_notify_window_ready(struct CommunicatorState* commState, int window_idx)
{
    telemetryRuntimeNotifyWindowReady(commState, window_idx);
}

/**
 * @brief Unregister a communicator from telemetry processing.
 *
 * @param[in] commState Communicator state to remove from telemetry tracking.
 */
void profiler_otel_telemetry_unregister_communicator(struct CommunicatorState* commState)
{
    telemetryRuntimeUnregisterCommunicator(commState);
}
