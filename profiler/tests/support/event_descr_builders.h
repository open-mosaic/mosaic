// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef TESTS_SUPPORT_EVENT_DESCR_BUILDERS_H_
#define TESTS_SUPPORT_EVENT_DESCR_BUILDERS_H_

#include <cstddef>
#include <cstdint>

#include "../../profiler_nccl_compat.h"

namespace testsupport
{
ncclProfilerEventDescr_v5_t makeCollDescr(const char* func = "AllReduce", size_t count = 1024,
                                          const char* datatype = "ncclInt32", uint8_t nChannels = 2,
                                          const char* algo = "Ring", const char* proto = "Simple",
                                          void* parent = nullptr);

ncclProfilerEventDescr_v5_t makeP2pDescr(const char* func = "Send", size_t count = 512,
                                         const char* datatype = "ncclFloat32", int peer = 3, uint8_t nChannels = 1,
                                         void* parent = nullptr);

ncclProfilerEventDescr_v5_t makeProxyOpDescr(uint8_t channelId = 0, int peer = 2, int chunkSize = 256, int isSend = 1,
                                             void* parent = nullptr);

ncclProfilerEventDescr_v5_t makeProxyStepDescr(int step = 0, void* parent = nullptr);
ncclProfilerEventDescr_v5_t makeGroupDescr(void* parent = nullptr);
ncclProfilerEventDescr_v5_t makeP2pApiDescr(const char* func = "Send", void* parent = nullptr);
ncclProfilerEventDescr_v5_t makeKernelChDescr(uint8_t channelId = 0, void* parent = nullptr);
}  // namespace testsupport

#endif  // TESTS_SUPPORT_EVENT_DESCR_BUILDERS_H_