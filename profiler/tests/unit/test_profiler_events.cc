// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../../communicator_state.h"
#include "../../events.h"
#include "../../profiler_otel.h"
#include "../test_helpers.h"

static void mock_logger(ncclDebugLogLevel level, unsigned long flags, const char* file, int line, const char* fmt, ...)
{
    (void)level;
    (void)flags;
    (void)file;
    (void)line;
    (void)fmt;
}

class ProfilerEventTest : public ::testing::Test
{
protected:
    void* context;
    int eActivationMask;

    void SetUp() override
    {
        context         = nullptr;
        eActivationMask = 0;

        resetProfilerState();

        ncclResult_t result =
            profiler_otel_init_v5(&context, 12345, &eActivationMask, "test_comm", 2, 4, 0, mock_logger);
        ASSERT_EQ(result, ncclSuccess);
        ASSERT_NE(context, nullptr);
    }

    void TearDown() override
    {
        if (context)
        {
            profiler_otel_finalize_v5(context);
            context = nullptr;
        }
    }

    ncclProfilerEventDescr_v5_t makeCollDescr(const char* func = "AllReduce", size_t count = 1024,
                                              const char* datatype = "ncclInt32", uint8_t nChannels = 2,
                                              const char* algo = "Ring", const char* proto = "Simple",
                                              void* parent = nullptr)
    {
        ncclProfilerEventDescr_v5_t descr = {};
        descr.type                        = ncclProfileColl;
        descr.coll.func                   = func;
        descr.coll.datatype               = datatype;
        descr.coll.count                  = count;
        descr.coll.nChannels              = nChannels;
        descr.coll.algo                   = algo;
        descr.coll.proto                  = proto;
        descr.parentObj                   = parent;
        return descr;
    }

    ncclProfilerEventDescr_v5_t makeP2pDescr(const char* func = "Send", size_t count = 512,
                                             const char* datatype = "ncclFloat32", int peer = 3, uint8_t nChannels = 1,
                                             void* parent = nullptr)
    {
        ncclProfilerEventDescr_v5_t descr = {};
        descr.type                        = ncclProfileP2p;
        descr.p2p.func                    = func;
        descr.p2p.datatype                = datatype;
        descr.p2p.count                   = count;
        descr.p2p.peer                    = peer;
        descr.p2p.nChannels               = nChannels;
        descr.parentObj                   = parent;
        return descr;
    }

    ncclProfilerEventDescr_v5_t makeProxyOpDescr(uint8_t channelId = 0, int peer = 2, int chunkSize = 256,
                                                 int isSend = 1, void* parent = nullptr)
    {
        ncclProfilerEventDescr_v5_t descr = {};
        descr.type                        = ncclProfileProxyOp;
        descr.proxyOp.channelId           = channelId;
        descr.proxyOp.peer                = peer;
        descr.proxyOp.chunkSize           = chunkSize;
        descr.proxyOp.isSend              = isSend;
        descr.proxyOp.pid                 = getpid();
        descr.parentObj                   = parent;
        return descr;
    }

    ncclProfilerEventDescr_v5_t makeProxyStepDescr(int step = 0, void* parent = nullptr)
    {
        ncclProfilerEventDescr_v5_t descr = {};
        descr.type                        = ncclProfileProxyStep;
        descr.proxyStep.step              = step;
        descr.parentObj                   = parent;
        return descr;
    }

    ncclProfilerEventDescr_v5_t makeGroupDescr(void* parent = nullptr)
    {
        ncclProfilerEventDescr_v5_t descr = {};
        descr.type                        = ncclProfileGroup;
        descr.parentObj                   = parent;
        return descr;
    }

    ncclProfilerEventDescr_v5_t makeKernelChDescr(uint8_t channelId = 0, void* parent = nullptr)
    {
        ncclProfilerEventDescr_v5_t descr = {};
        descr.type                        = ncclProfileKernelCh;
        descr.kernelCh.channelId          = channelId;
        descr.kernelCh.pTimer             = 0;
        descr.parentObj                   = parent;
        return descr;
    }
};

// =============================================================================
// Collective Events
// =============================================================================

TEST_F(ProfilerEventTest, StartCollectiveEventBasic)
{
    auto descr = makeCollDescr();

    void* eHandle       = nullptr;
    ncclResult_t result = profiler_otel_start_event_v5(context, &eHandle, &descr);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_NE(eHandle, nullptr);

    otelEventHandle_t* event = (otelEventHandle_t*)eHandle;
    EXPECT_EQ(event->type, ncclProfileColl);
    EXPECT_STREQ(event->coll.func, "AllReduce");
    EXPECT_EQ(event->coll.bytes, 4096u);  // 1024 * sizeof(int32)
    EXPECT_EQ(event->coll.nChannels, 2);
}

TEST_F(ProfilerEventTest, StartP2PEventBasic)
{
    auto descr = makeP2pDescr();

    void* eHandle       = nullptr;
    ncclResult_t result = profiler_otel_start_event_v5(context, &eHandle, &descr);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_NE(eHandle, nullptr);

    otelEventHandle_t* event = (otelEventHandle_t*)eHandle;
    EXPECT_EQ(event->type, ncclProfileP2p);
    EXPECT_STREQ(event->p2p.func, "Send");
    EXPECT_EQ(event->p2p.bytes, 2048u);  // 512 * sizeof(float32)
    EXPECT_EQ(event->p2p.peer, 3);
    EXPECT_EQ(event->p2p.nChannels, 1);
}

TEST_F(ProfilerEventTest, StartProxyOpEventBasic)
{
    auto descr = makeProxyOpDescr(0, 2, 256, 1, (void*)0x9ABC);

    void* eHandle       = nullptr;
    ncclResult_t result = profiler_otel_start_event_v5(context, &eHandle, &descr);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_NE(eHandle, nullptr);

    otelEventHandle_t* event = (otelEventHandle_t*)eHandle;
    EXPECT_EQ(event->type, ncclProfileProxyOp);
    EXPECT_EQ(event->proxyOp.channelId, 0);
    EXPECT_EQ(event->proxyOp.peer, 2);
    EXPECT_EQ(event->proxyOp.chunkSize, 256);
}

TEST_F(ProfilerEventTest, StartProxyOpEventReceiveSkipped)
{
    auto descr = makeProxyOpDescr(0, 2, 256, 0);  // isSend = 0

    void* eHandle       = nullptr;
    ncclResult_t result = profiler_otel_start_event_v5(context, &eHandle, &descr);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(eHandle, nullptr);
}

TEST_F(ProfilerEventTest, StartProxyOpEventWrongPid)
{
    auto descr        = makeProxyOpDescr();
    descr.proxyOp.pid = getpid() + 1000;

    void* eHandle       = nullptr;
    ncclResult_t result = profiler_otel_start_event_v5(context, &eHandle, &descr);

    EXPECT_EQ(result, ncclSuccess);
}

// =============================================================================
// Stop Events
// =============================================================================

TEST_F(ProfilerEventTest, StopEventBasic)
{
    auto descr    = makeCollDescr();
    void* eHandle = nullptr;
    profiler_otel_start_event_v5(context, &eHandle, &descr);
    ASSERT_NE(eHandle, nullptr);

    otelEventHandle_t* event = (otelEventHandle_t*)eHandle;
    double startTs           = event->startTs;
    EXPECT_EQ(event->endTs, 0.0);

    ncclResult_t result = profiler_otel_stop_event_v5(eHandle);
    EXPECT_EQ(result, ncclSuccess);
    EXPECT_GT(event->endTs, 0.0);
    EXPECT_GE(event->endTs, startTs);
}

TEST_F(ProfilerEventTest, StopEventWithNullHandle)
{
    ncclResult_t result = profiler_otel_stop_event_v5(nullptr);
    EXPECT_EQ(result, ncclSuccess);
}

// =============================================================================
// Multiple Events
// =============================================================================

TEST_F(ProfilerEventTest, MultipleCollectiveEvents)
{
    std::vector<void*> handles;

    for (int i = 0; i < 10; i++)
    {
        auto descr    = makeCollDescr("AllReduce", 1024 + i * 100);
        void* eHandle = nullptr;
        profiler_otel_start_event_v5(context, &eHandle, &descr);
        EXPECT_NE(eHandle, nullptr);
        handles.push_back(eHandle);
    }

    for (auto handle : handles)
    {
        ncclResult_t result = profiler_otel_stop_event_v5(handle);
        EXPECT_EQ(result, ncclSuccess);
    }
}

// =============================================================================
// Datatype Size Calculation
// =============================================================================

TEST_F(ProfilerEventTest, DatatypeSizeCalculation)
{
    struct TestCase
    {
        const char* datatype;
        size_t count;
        size_t expectedBytes;
    };

    TestCase cases[] = {
        {"ncclInt8",     1000, 1000},
        {"ncclUint8",    1000, 1000},
        {"ncclFloat16",  1000, 2000},
        {"ncclBfloat16", 1000, 2000},
        {"ncclInt32",    1000, 4000},
        {"ncclUint32",   1000, 4000},
        {"ncclFloat32",  1000, 4000},
        {"ncclInt64",    1000, 8000},
        {"ncclUint64",   1000, 8000},
        {"ncclFloat64",  1000, 8000},
        {"unknown",      1000, 0   },
        {nullptr,        1000, 0   },
    };

    for (const auto& testCase : cases)
    {
        auto descr    = makeCollDescr("AllReduce", testCase.count, testCase.datatype);
        void* eHandle = nullptr;
        profiler_otel_start_event_v5(context, &eHandle, &descr);

        if (testCase.expectedBytes > 0)
        {
            ASSERT_NE(eHandle, nullptr);
            otelEventHandle_t* event = (otelEventHandle_t*)eHandle;
            EXPECT_EQ(event->coll.bytes, testCase.expectedBytes)
                << "Failed for datatype: " << (testCase.datatype ? testCase.datatype : "nullptr");
        }
    }
}

// =============================================================================
// Buffer Overflow Handling
// =============================================================================

TEST_F(ProfilerEventTest, BufferOverflowHandling)
{
    std::vector<void*> handles;
    int successCount = 0;

    for (int i = 0; i < 25000; i++)
    {
        auto descr          = makeCollDescr();
        void* eHandle       = nullptr;
        ncclResult_t result = profiler_otel_start_event_v5(context, &eHandle, &descr);
        EXPECT_EQ(result, ncclSuccess);

        if (eHandle != nullptr)
        {
            successCount++;
            handles.push_back(eHandle);
        }
    }

    EXPECT_GE(successCount, 10000);
}

// =============================================================================
// Record Event State
// =============================================================================

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

// =============================================================================
// Finalize
// =============================================================================

TEST_F(ProfilerEventTest, FinalizeWithNullContext)
{
    ncclResult_t result = profiler_otel_finalize_v5(nullptr);
    EXPECT_EQ(result, ncclSuccess);
}

// =============================================================================
// Window Triggering
// =============================================================================

TEST_F(ProfilerEventTest, WindowTriggeringAfterTriggerCount)
{
    for (int i = 0; i < WINDOW_TRIGGER_COUNT; i++)
    {
        auto descr    = makeCollDescr("AllReduce", 100);
        void* eHandle = nullptr;
        profiler_otel_start_event_v5(context, &eHandle, &descr);
    }

    // Buffer should switch after WINDOW_TRIGGER_COUNT events
}

// =============================================================================
// ProxyStep Events
// =============================================================================

TEST_F(ProfilerEventTest, StartProxyStepEventBasic)
{
    auto proxyOpDescr   = makeProxyOpDescr(0, 1, 1024, 1, (void*)0x1234);
    void* proxyOpHandle = nullptr;
    profiler_otel_start_event_v5(context, &proxyOpHandle, &proxyOpDescr);
    ASSERT_NE(proxyOpHandle, nullptr);

    auto proxyStepDescr   = makeProxyStepDescr(0, proxyOpHandle);
    void* proxyStepHandle = nullptr;
    ncclResult_t result   = profiler_otel_start_event_v5(context, &proxyStepHandle, &proxyStepDescr);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_NE(proxyStepHandle, nullptr);

    otelEventHandle_t* event = (otelEventHandle_t*)proxyStepHandle;
    EXPECT_EQ(event->type, ncclProfileProxyStep);
    EXPECT_EQ(event->proxyStep.step, 0);
}

TEST_F(ProfilerEventTest, ProxyStepEventWithNullParent)
{
    auto descr          = makeProxyStepDescr(0, nullptr);
    void* eHandle       = nullptr;
    ncclResult_t result = profiler_otel_start_event_v5(context, &eHandle, &descr);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(eHandle, nullptr);
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

    otelEventHandle_t* event = (otelEventHandle_t*)proxyStepHandle;
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

    otelEventHandle_t* event = (otelEventHandle_t*)proxyStepHandle;
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
            otelEventHandle_t* event = (otelEventHandle_t*)proxyStepHandle;
            EXPECT_EQ(event->proxyStep.step, step);
            stepHandles.push_back(proxyStepHandle);
        }
    }

    EXPECT_GT(stepHandles.size(), 0u);
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

    otelEventHandle_t* event = (otelEventHandle_t*)proxyStepHandle;
    double startTs           = event->startTs;

    ncclProfilerEventStateArgs_v5_t args = {};
    args.proxyStep.transSize             = 4096;
    profiler_otel_record_event_state_v5(proxyStepHandle, ncclProfilerProxyStepSendWait, &args);

    double sendWaitTs = event->proxyStep.sendWaitTs;
    EXPECT_GE(sendWaitTs, startTs);

    ncclResult_t result = profiler_otel_stop_event_v5(proxyStepHandle);
    EXPECT_EQ(result, ncclSuccess);

    EXPECT_GT(event->endTs, 0.0);
    EXPECT_GE(event->endTs, sendWaitTs);
}

// =============================================================================
// Group Events
// =============================================================================

TEST_F(ProfilerEventTest, StartGroupEventBasic)
{
    auto descr          = makeGroupDescr();
    void* eHandle       = nullptr;
    ncclResult_t result = profiler_otel_start_event_v5(context, &eHandle, &descr);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_NE(eHandle, nullptr);

    otelEventHandle_t* event = (otelEventHandle_t*)eHandle;
    EXPECT_EQ(event->type, ncclProfileGroup);
}

TEST_F(ProfilerEventTest, StopGroupEvent)
{
    auto descr    = makeGroupDescr();
    void* eHandle = nullptr;
    profiler_otel_start_event_v5(context, &eHandle, &descr);
    ASSERT_NE(eHandle, nullptr);

    otelEventHandle_t* event = (otelEventHandle_t*)eHandle;
    double startTs           = event->startTs;
    EXPECT_EQ(event->endTs, 0.0);

    ncclResult_t result = profiler_otel_stop_event_v5(eHandle);
    EXPECT_EQ(result, ncclSuccess);

    EXPECT_GT(event->endTs, 0.0);
    EXPECT_GE(event->endTs, startTs);
}

TEST_F(ProfilerEventTest, GroupEventAsParent)
{
    auto groupDescr   = makeGroupDescr();
    void* groupHandle = nullptr;
    profiler_otel_start_event_v5(context, &groupHandle, &groupDescr);
    ASSERT_NE(groupHandle, nullptr);

    auto collDescr      = makeCollDescr("AllReduce", 1024, "ncclInt32", 2, "Ring", "Simple", groupHandle);
    void* collHandle    = nullptr;
    ncclResult_t result = profiler_otel_start_event_v5(context, &collHandle, &collDescr);
    EXPECT_EQ(result, ncclSuccess);
    EXPECT_NE(collHandle, nullptr);

    otelEventHandle_t* collEvent = (otelEventHandle_t*)collHandle;
    EXPECT_EQ(collEvent->parentObj, groupHandle);

    profiler_otel_stop_event_v5(collHandle);
    profiler_otel_stop_event_v5(groupHandle);
}

TEST_F(ProfilerEventTest, MultipleGroupEvents)
{
    std::vector<void*> handles;

    for (int i = 0; i < 5; i++)
    {
        auto descr    = makeGroupDescr();
        void* eHandle = nullptr;
        profiler_otel_start_event_v5(context, &eHandle, &descr);
        EXPECT_NE(eHandle, nullptr);
        handles.push_back(eHandle);
    }

    for (auto handle : handles)
    {
        ncclResult_t result = profiler_otel_stop_event_v5(handle);
        EXPECT_EQ(result, ncclSuccess);
    }
}

// =============================================================================
// KernelCh Events
// =============================================================================

TEST_F(ProfilerEventTest, StartKernelChEvent)
{
    auto collDescr   = makeCollDescr();
    void* collHandle = nullptr;
    profiler_otel_start_event_v5(context, &collHandle, &collDescr);
    ASSERT_NE(collHandle, nullptr);

    auto kcDescr        = makeKernelChDescr(0, collHandle);
    void* kcHandle      = nullptr;
    ncclResult_t result = profiler_otel_start_event_v5(context, &kcHandle, &kcDescr);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_NE(kcHandle, nullptr);

    otelEventHandle_t* event = (otelEventHandle_t*)kcHandle;
    EXPECT_EQ(event->type, ncclProfileKernelCh);
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

    otelEventHandle_t* event = (otelEventHandle_t*)kcHandle;
    EXPECT_EQ(event->endTs, 0.0);

    ncclResult_t result = profiler_otel_stop_event_v5(kcHandle);
    EXPECT_EQ(result, ncclSuccess);
    EXPECT_GT(event->endTs, 0.0);
}

TEST_F(ProfilerEventTest, KernelChTracksInProgressCount)
{
    struct eventContext* ctx     = (struct eventContext*)context;
    CommunicatorState* commState = ctx->commState;
    uint8_t bufIdx               = commState->get_active_buffer_idx();

    auto collDescr   = makeCollDescr();
    void* collHandle = nullptr;
    profiler_otel_start_event_v5(context, &collHandle, &collDescr);
    ASSERT_NE(collHandle, nullptr);

    WindowMetadata* window = commState->get_window_metadata(bufIdx);
    uint32_t before        = window->kernel_ch_in_progress.load();

    auto kcDescr   = makeKernelChDescr(0, collHandle);
    void* kcHandle = nullptr;
    profiler_otel_start_event_v5(context, &kcHandle, &kcDescr);
    ASSERT_NE(kcHandle, nullptr);

    EXPECT_EQ(window->kernel_ch_in_progress.load(), before + 1);

    profiler_otel_stop_event_v5(kcHandle);
    EXPECT_EQ(window->kernel_ch_in_progress.load(), before);
}

// KernelCh events with NULL parentObj must be filtered.
// These arise from P2P Recv sub-operations: our plugin returns NULL eHandle for Recv
// events, so NCCL stores NULL as the task event handle, and uses it as parentObj for
// the KernelCh.  If we allocate a slot and increment kernel_ch_in_progress for these,
// the window may process before their stop_event arrives, producing an endTs=0 warning.
TEST_F(ProfilerEventTest, KernelChWithNullParentIsSkipped)
{
    struct eventContext* ctx     = (struct eventContext*)context;
    CommunicatorState* commState = ctx->commState;
    uint8_t bufIdx               = commState->get_active_buffer_idx();
    WindowMetadata* window       = commState->get_window_metadata(bufIdx);

    uint32_t kernel_ch_before = window->kernel_ch_in_progress.load();
    uint32_t count_before     = window->element_count.load();

    // Attempt to start a KernelCh with NULL parentObj (simulates a Recv sub-op KernelCh).
    auto kcDescr        = makeKernelChDescr(0, nullptr);  // parentObj = NULL
    void* kcHandle      = nullptr;
    ncclResult_t result = profiler_otel_start_event_v5(context, &kcHandle, &kcDescr);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(kcHandle, nullptr);  // Must be filtered (no slot allocated)

    // kernel_ch_in_progress and element_count must be unchanged.
    EXPECT_EQ(window->kernel_ch_in_progress.load(), kernel_ch_before);
    EXPECT_EQ(window->element_count.load(), count_before);
}
