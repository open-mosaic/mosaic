// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "../../support/profiler_event_fixture.h"

using testsupport::makeCollDescr;
using testsupport::makeKernelChDescr;
using testsupport::makeProxyOpDescr;
using testsupport::makeProxyStepDescr;

TEST_F(ProfilerEventTest, RecordEventStateForColl)
{
    auto descr    = makeCollDescr();
    void* eHandle = nullptr;
    profiler_otel_start_event_v5(context, &eHandle, &descr);
    ASSERT_NE(eHandle, nullptr);

    ncclProfilerEventStateArgs_v5_t args = {};
    ncclResult_t result = profiler_otel_record_event_state_v5(eHandle, (ncclProfilerEventState_v5_t)0, &args);
    EXPECT_EQ(result, ncclSuccess);
}

TEST_F(ProfilerEventTest, WindowTriggeringAfterTriggerCount)
{
    for (int i = 0; i < WINDOW_TRIGGER_COUNT; i++)
    {
        auto descr    = makeCollDescr("AllReduce", 100);
        void* eHandle = nullptr;
        profiler_otel_start_event_v5(context, &eHandle, &descr);
    }
}

TEST_F(ProfilerEventTest, ProxyStepStateTransitionSendWait)
{
    auto proxyOpDescr   = makeProxyOpDescr(0, 1, 1024, 1, (void*)0x1234);
    void* proxyOpHandle = nullptr;
    profiler_otel_start_event_v5(context, &proxyOpHandle, &proxyOpDescr);
    ASSERT_NE(proxyOpHandle, nullptr);

    auto proxyStepDescr   = makeProxyStepDescr(0, proxyOpHandle);
    void* proxyStepHandle = nullptr;
    profiler_otel_start_event_v5(context, &proxyStepHandle, &proxyStepDescr);
    ASSERT_NE(proxyStepHandle, nullptr);

    auto* event = static_cast<otelEventHandle_t*>(proxyStepHandle);
    EXPECT_FALSE(event->proxyStep.hasSendWait);

    ncclProfilerEventStateArgs_v5_t args = {};
    args.proxyStep.transSize             = 4096;

    ncclResult_t result = profiler_otel_record_event_state_v5(proxyStepHandle, ncclProfilerProxyStepSendWait, &args);
    EXPECT_EQ(result, ncclSuccess);

    EXPECT_TRUE(event->proxyStep.hasSendWait);
    EXPECT_EQ(event->proxyStep.transSize, 4096u);
    EXPECT_GT(event->proxyStep.sendWaitTs, 0.0);
}

TEST_F(ProfilerEventTest, ProxyStepStateTransitionWithoutArgs)
{
    auto proxyOpDescr   = makeProxyOpDescr(0, 1, 1024, 1, (void*)0x1234);
    void* proxyOpHandle = nullptr;
    profiler_otel_start_event_v5(context, &proxyOpHandle, &proxyOpDescr);
    ASSERT_NE(proxyOpHandle, nullptr);

    auto proxyStepDescr   = makeProxyStepDescr(0, proxyOpHandle);
    void* proxyStepHandle = nullptr;
    profiler_otel_start_event_v5(context, &proxyStepHandle, &proxyStepDescr);
    ASSERT_NE(proxyStepHandle, nullptr);

    ncclResult_t result = profiler_otel_record_event_state_v5(proxyStepHandle, ncclProfilerProxyStepSendWait, nullptr);
    EXPECT_EQ(result, ncclSuccess);

    auto* event = static_cast<otelEventHandle_t*>(proxyStepHandle);
    EXPECT_FALSE(event->proxyStep.hasSendWait);
}

TEST_F(ProfilerEventTest, ProxyStepMultipleSteps)
{
    auto proxyOpDescr   = makeProxyOpDescr(0, 1, 1024, 1, (void*)0x1234);
    void* proxyOpHandle = nullptr;
    profiler_otel_start_event_v5(context, &proxyOpHandle, &proxyOpDescr);
    ASSERT_NE(proxyOpHandle, nullptr);

    std::vector<void*> stepHandles;
    for (int step = 0; step < 8; step++)
    {
        auto proxyStepDescr   = makeProxyStepDescr(step, proxyOpHandle);
        void* proxyStepHandle = nullptr;
        profiler_otel_start_event_v5(context, &proxyStepHandle, &proxyStepDescr);

        if (proxyStepHandle != nullptr)
        {
            auto* event = static_cast<otelEventHandle_t*>(proxyStepHandle);
            EXPECT_EQ(event->proxyStep.step, step);
            stepHandles.push_back(proxyStepHandle);
        }
    }

    EXPECT_GT(stepHandles.size(), 0u);
}

TEST_F(ProfilerEventTest, KernelChTracksInProgressCount)
{
    auto* ctx                = static_cast<eventContext*>(context);
    CommunicatorState* state = ctx->commState;
    uint8_t bufIdx           = state->get_active_buffer_idx();

    auto collDescr   = makeCollDescr();
    void* collHandle = nullptr;
    profiler_otel_start_event_v5(context, &collHandle, &collDescr);
    ASSERT_NE(collHandle, nullptr);

    WindowMetadata* window = state->get_window_metadata(bufIdx);
    uint32_t before        = window->kernel_ch_in_progress.load();

    auto kcDescr   = makeKernelChDescr(0, collHandle);
    void* kcHandle = nullptr;
    profiler_otel_start_event_v5(context, &kcHandle, &kcDescr);
    ASSERT_NE(kcHandle, nullptr);

    EXPECT_EQ(window->kernel_ch_in_progress.load(), before + 1);

    profiler_otel_stop_event_v5(kcHandle);
    EXPECT_EQ(window->kernel_ch_in_progress.load(), before);
}

TEST_F(ProfilerEventTest, KernelChWithNullParentIsSkipped)
{
    auto* ctx                = static_cast<eventContext*>(context);
    CommunicatorState* state = ctx->commState;
    uint8_t bufIdx           = state->get_active_buffer_idx();
    WindowMetadata* window   = state->get_window_metadata(bufIdx);

    uint32_t kernelBefore = window->kernel_ch_in_progress.load();
    uint32_t countBefore  = window->element_count.load();

    auto kcDescr        = makeKernelChDescr(0, nullptr);
    void* kcHandle      = nullptr;
    ncclResult_t result = profiler_otel_start_event_v5(context, &kcHandle, &kcDescr);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(kcHandle, nullptr);
    EXPECT_EQ(window->kernel_ch_in_progress.load(), kernelBefore);
    EXPECT_EQ(window->element_count.load(), countBefore);
}