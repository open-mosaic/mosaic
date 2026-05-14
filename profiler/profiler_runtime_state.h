// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef PROFILER_RUNTIME_STATE_H_
#define PROFILER_RUNTIME_STATE_H_

#include <pthread.h>
#include <sys/types.h>

#include "profiler_nccl_compat.h"

extern int initialized;
extern double startTime;
extern ncclDebugLogger_t otel_log_func;
extern pthread_mutex_t otelLock;
extern pid_t pid;
extern int telemetry_initialized;
extern int active_communicators;

#endif  // PROFILER_RUNTIME_STATE_H_