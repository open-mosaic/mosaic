// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef PROFILER_OTEL_LIFECYCLE_H_
#define PROFILER_OTEL_LIFECYCLE_H_

#include <cstdint>

#include "profiler_nccl_compat.h"

ncclResult_t initializeProfilerContext(void** context, uint64_t commId, int* eActivationMask, const char* commName,
                                       int nNodes, int nranks, int rank);
ncclResult_t finalizeProfilerContext(void* context);

#endif  // PROFILER_OTEL_LIFECYCLE_H_