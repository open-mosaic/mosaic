// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef TESTS_SUPPORT_WINDOW_AGGREGATOR_FIXTURE_H_
#define TESTS_SUPPORT_WINDOW_AGGREGATOR_FIXTURE_H_

#include <gtest/gtest.h>

#include "../../aggregation.h"
#include "../../communicator_state.h"
#include "../../events.h"
#include "event_handle_builders.h"

class WindowAggregatorTest : public ::testing::Test
{
protected:
    WindowAggregator* aggregator;

    void SetUp() override
    {
        aggregator = new WindowAggregator(0);
    }

    void TearDown() override
    {
        delete aggregator;
    }

    otelEventHandle_t createCollectiveEvent(const char* func, const char* algo, const char* proto, uint8_t channels,
                                            size_t bytes, double startTs, double endTs)
    {
        return testsupport::makeCollectiveEvent(func, algo, proto, channels, bytes, startTs, endTs);
    }

    otelEventHandle_t createP2PEvent(const char* func, int peer, uint8_t channels, size_t bytes, double startTs,
                                     double endTs)
    {
        return testsupport::makeP2PEvent(func, peer, channels, bytes, startTs, endTs);
    }

    otelEventHandle_t createProxyOpEvent(int peer, uint8_t channelId, int chunkSize, double startTs, double endTs,
                                         void* parentObj = nullptr)
    {
        return testsupport::makeProxyOpEvent(peer, channelId, chunkSize, startTs, endTs, parentObj);
    }

    otelEventHandle_t createProxyStepEvent(int step, size_t transSize, double startTs, double sendWaitTs, double endTs,
                                           void* parentObj = nullptr)
    {
        return testsupport::makeProxyStepEvent(step, transSize, startTs, sendWaitTs, endTs, parentObj);
    }

    otelEventHandle_t createKernelChEvent(uint8_t channelId, uint64_t pTimerStart, uint64_t pTimerStop, double startTs,
                                          double endTs, void* parentObj = nullptr)
    {
        return testsupport::makeKernelChEvent(channelId, pTimerStart, pTimerStop, startTs, endTs, parentObj);
    }

    otelEventHandle_t createGroupEvent(double startTs, double endTs, CommunicatorState* commState)
    {
        return testsupport::makeGroupEvent(startTs, endTs, commState);
    }

    otelEventHandle_t createP2pApiEvent(const char* func, double startTs, double endTs, CommunicatorState* commState)
    {
        return testsupport::makeP2pApiEvent(func, startTs, endTs, commState);
    }

    otelEventHandle_t createCollectiveEventWithCommState(const char* func, const char* algo, const char* proto,
                                                         uint8_t channels, size_t bytes, double startTs, double endTs,
                                                         CommunicatorState* commState)
    {
        return testsupport::makeCollectiveEventWithCommState(func, algo, proto, channels, bytes, startTs, endTs,
                                                             commState);
    }
};

#endif  // TESTS_SUPPORT_WINDOW_AGGREGATOR_FIXTURE_H_