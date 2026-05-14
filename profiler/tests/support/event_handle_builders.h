// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef TESTS_SUPPORT_EVENT_HANDLE_BUILDERS_H_
#define TESTS_SUPPORT_EVENT_HANDLE_BUILDERS_H_

#include <cstddef>
#include <cstdint>

#include "../../communicator_state.h"
#include "../../events.h"

namespace testsupport
{
otelEventHandle_t makeCollectiveEvent(const char* func, const char* algo, const char* proto, uint8_t channels,
                                      size_t bytes, double startTs, double endTs);

otelEventHandle_t makeP2PEvent(const char* func, int peer, uint8_t channels, size_t bytes, double startTs,
                               double endTs);

otelEventHandle_t makeProxyOpEvent(int peer, uint8_t channelId, int chunkSize, double startTs, double endTs,
                                   void* parentObj = nullptr);

otelEventHandle_t makeProxyStepEvent(int step, size_t transSize, double startTs, double sendWaitTs, double endTs,
                                     void* parentObj = nullptr);

otelEventHandle_t makeKernelChEvent(uint8_t channelId, uint64_t pTimerStart, uint64_t pTimerStop, double startTs,
                                    double endTs, void* parentObj = nullptr);

otelEventHandle_t makeGroupEvent(double startTs, double endTs, CommunicatorState* commState);
otelEventHandle_t makeP2pApiEvent(const char* func, double startTs, double endTs, CommunicatorState* commState);
otelEventHandle_t makeCollectiveEventWithCommState(const char* func, const char* algo, const char* proto,
                                                   uint8_t channels, size_t bytes, double startTs, double endTs,
                                                   CommunicatorState* commState);
}  // namespace testsupport

#endif  // TESTS_SUPPORT_EVENT_HANDLE_BUILDERS_H_