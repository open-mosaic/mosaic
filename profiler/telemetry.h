// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef OTEL_TELEMETRY_H_
#define OTEL_TELEMETRY_H_

struct CommunicatorState;

void profiler_otel_telemetry_init();

void profiler_otel_telemetry_cleanup();

void profiler_otel_telemetry_notify_window_ready(struct CommunicatorState* commState, int window_idx);

void profiler_otel_telemetry_unregister_communicator(struct CommunicatorState* commState);

#endif  // OTEL_TELEMETRY_H_
