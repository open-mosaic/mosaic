// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "../../support/profiler_event_fixture.h"

using testsupport::makeCollDescr;
using testsupport::makeGroupDescr;
using testsupport::makeKernelChDescr;
using testsupport::makeProxyOpDescr;
using testsupport::makeProxyStepDescr;

TEST_F(ProfilerEventTest, StopEventBasic)
{
    auto descr    = makeCollDescr();
    void* eHandle = nullptr;
    profiler_otel_start_event_v5(context, &eHandle, &descr);
    ASSERT_NE(eHandle, nullptr);

    auto* event  = static_cast<otelEventHandle_t*>(eHandle);
    double start = event->startTs;
    EXPECT_EQ(event->endTs, 0.0);

    ncclResult_t result = profiler_otel_stop_event_v5(eHandle);
    EXPECT_EQ(result, ncclSuccess);
    EXPECT_GT(event->endTs, 0.0);
    EXPECT_GE(event->endTs, start);
}

TEST_F(ProfilerEventTest, StopEventWithNullHandle)
{
    ncclResult_t result = profiler_otel_stop_event_v5(nullptr);
    EXPECT_EQ(result, ncclSuccess);
}

TEST_F(ProfilerEventTest, ProxyStepStopEventWithSendWait)
{
    auto proxyOpDescr   = makeProxyOpDescr(0, 1, 1024, 1, (void*)0x1234);
    void* proxyOpHandle = nullptr;
    profiler_otel_start_event_v5(context, &proxyOpHandle, &proxyOpDescr);
    ASSERT_NE(proxyOpHandle, nullptr);

    auto proxyStepDescr   = makeProxyStepDescr(0, proxyOpHandle);
    void* proxyStepHandle = nullptr;
    profiler_otel_start_event_v5(context, &proxyStepHandle, &proxyStepDescr);
    ASSERT_NE(proxyStepHandle, nullptr);

    auto* event  = static_cast<otelEventHandle_t*>(proxyStepHandle);
    double start = event->startTs;

    ncclProfilerEventStateArgs_v5_t args = {};
    args.proxyStep.transSize             = 4096;
    profiler_otel_record_event_state_v5(proxyStepHandle, ncclProfilerProxyStepSendWait, &args);

    double sendWaitTs = event->proxyStep.sendWaitTs;
    EXPECT_GE(sendWaitTs, start);

    ncclResult_t result = profiler_otel_stop_event_v5(proxyStepHandle);
    EXPECT_EQ(result, ncclSuccess);
    EXPECT_GT(event->endTs, 0.0);
    EXPECT_GE(event->endTs, sendWaitTs);
}

TEST_F(ProfilerEventTest, StopGroupEvent)
{
    auto descr    = makeGroupDescr();
    void* eHandle = nullptr;
    profiler_otel_start_event_v5(context, &eHandle, &descr);
    ASSERT_NE(eHandle, nullptr);

    auto* event  = static_cast<otelEventHandle_t*>(eHandle);
    double start = event->startTs;
    EXPECT_EQ(event->endTs, 0.0);

    ncclResult_t result = profiler_otel_stop_event_v5(eHandle);
    EXPECT_EQ(result, ncclSuccess);
    EXPECT_GT(event->endTs, 0.0);
    EXPECT_GE(event->endTs, start);
}

TEST_F(ProfilerEventTest, StopKernelChEvent)
{
    auto collDescr   = makeCollDescr();
    void* collHandle = nullptr;
    profiler_otel_start_event_v5(context, &collHandle, &collDescr);
    ASSERT_NE(collHandle, nullptr);

    auto kcDescr   = makeKernelChDescr(0, collHandle);
    void* kcHandle = nullptr;
    profiler_otel_start_event_v5(context, &kcHandle, &kcDescr);
    ASSERT_NE(kcHandle, nullptr);

    auto* event = static_cast<otelEventHandle_t*>(kcHandle);
    EXPECT_EQ(event->endTs, 0.0);

    ncclResult_t result = profiler_otel_stop_event_v5(kcHandle);
    EXPECT_EQ(result, ncclSuccess);
    EXPECT_GT(event->endTs, 0.0);
}

TEST_F(ProfilerEventTest, FinalizeWithNullContext)
{
    ncclResult_t result = profiler_otel_finalize_v5(nullptr);
    EXPECT_EQ(result, ncclSuccess);
}