// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "profiler_runtime_state.h"

#include "profiler_otel.h"

int initialized                 = 0;
double startTime                = 0.0;
ncclDebugLogger_t otel_log_func = nullptr;
pthread_mutex_t otelLock        = PTHREAD_MUTEX_INITIALIZER;
pid_t pid                       = 0;
int telemetry_initialized       = 0;
int active_communicators        = 0;

#ifdef UNIT_TESTING
int getInitialized()
{
    return initialized;
}

void setInitialized(int value)
{
    initialized = value;
}

double getStartTime()
{
    return startTime;
}

void setStartTime(double value)
{
    startTime = value;
}

pid_t getPid()
{
    return pid;
}

void setPid(pid_t value)
{
    pid = value;
}
#endif  // UNIT_TESTING