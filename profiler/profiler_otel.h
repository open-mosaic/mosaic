// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef PROFILER_OTEL_H
#define PROFILER_OTEL_H

#include <sys/types.h>

#include <cstdint>
#include <string>

#include "profiler_nccl_compat.h"

// Make functions hidden - only accessible via plugin structure
#define OTEL_HIDDEN __attribute__((visibility("hidden")))

// Global log function pointer (set during otelProfilerInit)
extern ncclDebugLogger_t otel_log_func;

// Test interface functions for unit testing
#ifdef UNIT_TESTING
int getInitialized();
void setInitialized(int value);
double getStartTime();
void setStartTime(double value);
pid_t getPid();
void setPid(pid_t value);

// Function declaration for mocking
double gettime();

// Utility functions exposed for testing
size_t test_ncclTypeSize(const char* datatype);
std::string test_gpuUuidToString(const unsigned char* uuid_bytes);
#endif  // UNIT_TESTING

// Logging macros that use NCCL's logging system with PROF/OTEL prefix
#define OTEL_WARN(FLAGS, fmt, ...)                                                                                     \
    if (otel_log_func)                                                                                                 \
    (*otel_log_func)(NCCL_LOG_WARN, (FLAGS), __FUNCTION__, __LINE__, "[PROFILER/OTEL] " fmt, ##__VA_ARGS__)

#define OTEL_INFO(FLAGS, fmt, ...)                                                                                     \
    if (otel_log_func)                                                                                                 \
    (*otel_log_func)(NCCL_LOG_INFO, (FLAGS), __FUNCTION__, __LINE__, "[PROFILER/OTEL] " fmt, ##__VA_ARGS__)

// Compile-time TRACE gating: if PROFILER_OTEL_ENABLE_TRACE is not defined, OTEL_TRACE compiles to a no-op.
#ifdef PROFILER_OTEL_ENABLE_TRACE
#define OTEL_TRACE(FLAGS, fmt, ...)                                                                                    \
    if (otel_log_func)                                                                                                 \
    (*otel_log_func)(NCCL_LOG_TRACE, (FLAGS), __FUNCTION__, __LINE__, "[PROFILER/OTEL] " fmt, ##__VA_ARGS__)
#else
#define OTEL_TRACE(FLAGS, fmt, ...)                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#endif

#ifdef __cplusplus
extern "C"
{
#endif  // __cplusplus

    OTEL_HIDDEN ncclResult_t profiler_otel_init_v5(void** context, uint64_t commId, int* eActivationMask,
                                                   const char* commName, int nNodes, int nranks, int rank,
                                                   ncclDebugLogger_t logfn);

    OTEL_HIDDEN ncclResult_t profiler_otel_start_event_v5(void* context, void** eHandle,
                                                          ncclProfilerEventDescr_v5_t* eDescr);

    OTEL_HIDDEN ncclResult_t profiler_otel_stop_event_v5(void* eHandle);

    OTEL_HIDDEN ncclResult_t profiler_otel_record_event_state_v5(void* eHandle, ncclProfilerEventState_v5_t eState,
                                                                 ncclProfilerEventStateArgs_v5_t* eStateArgs);

    OTEL_HIDDEN ncclResult_t profiler_otel_finalize_v5(void* context);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // PROFILER_OTEL_H
