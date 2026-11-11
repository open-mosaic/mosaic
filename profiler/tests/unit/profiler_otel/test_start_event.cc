// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <unistd.h>

#include <vector>

#include "../../support/profiler_event_fixture.h"

using testsupport::makeCollApiDescr;
using testsupport::makeCollDescr;
using testsupport::makeGroupApiDescr;
using testsupport::makeGroupDescr;
using testsupport::makeKernelChDescr;
using testsupport::makeP2pApiDescr;
using testsupport::makeP2pDescr;
using testsupport::makeProxyOpDescr;
using testsupport::makeProxyStepDescr;

TEST_F(ProfilerEventTest, StartCollectiveEventBasic)
{
    auto descr = makeCollDescr();

    void* eHandle       = nullptr;
    ncclResult_t result = profiler_otel_start_event_v5(context, &eHandle, &descr);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_NE(eHandle, nullptr);

    auto* event = static_cast<otelEventHandle_t*>(eHandle);
    EXPECT_EQ(event->type, ncclProfileColl);
    EXPECT_STREQ(event->coll.func, "AllReduce");
    EXPECT_EQ(event->coll.bytes, 4096u);
    EXPECT_EQ(event->coll.nChannels, 2);
}

TEST_F(ProfilerEventTest, StartP2PEventBasic)
{
    auto descr = makeP2pDescr();

    void* eHandle       = nullptr;
    ncclResult_t result = profiler_otel_start_event_v5(context, &eHandle, &descr);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_NE(eHandle, nullptr);

    auto* event = static_cast<otelEventHandle_t*>(eHandle);
    EXPECT_EQ(event->type, ncclProfileP2p);
    EXPECT_STREQ(event->p2p.func, "Send");
    EXPECT_EQ(event->p2p.bytes, 2048u);
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

    auto* event = static_cast<otelEventHandle_t*>(eHandle);
    EXPECT_EQ(event->type, ncclProfileProxyOp);
    EXPECT_EQ(event->proxyOp.channelId, 0);
    EXPECT_EQ(event->proxyOp.peer, 2);
    EXPECT_EQ(event->proxyOp.chunkSize, 256);
}

TEST_F(ProfilerEventTest, StartProxyOpEventReceiveSkipped)
{
    auto descr = makeProxyOpDescr(0, 2, 256, 0);

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
        {nullptr,        1000, 0   }
    };

    for (const auto& testCase : cases)
    {
        auto descr    = makeCollDescr("AllReduce", testCase.count, testCase.datatype);
        void* eHandle = nullptr;
        profiler_otel_start_event_v5(context, &eHandle, &descr);

        if (testCase.expectedBytes > 0)
        {
            ASSERT_NE(eHandle, nullptr);
            auto* event = static_cast<otelEventHandle_t*>(eHandle);
            EXPECT_EQ(event->coll.bytes, testCase.expectedBytes)
                << "Failed for datatype: " << (testCase.datatype ? testCase.datatype : "nullptr");
        }
    }
}

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

    auto* event = static_cast<otelEventHandle_t*>(proxyStepHandle);
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

TEST_F(ProfilerEventTest, StartGroupEventBasic)
{
    auto descr          = makeGroupDescr();
    void* eHandle       = nullptr;
    ncclResult_t result = profiler_otel_start_event_v5(context, &eHandle, &descr);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_NE(eHandle, nullptr);

    auto* event = static_cast<otelEventHandle_t*>(eHandle);
    EXPECT_EQ(event->type, ncclProfileGroup);
}

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

    auto* event = static_cast<otelEventHandle_t*>(kcHandle);
    EXPECT_EQ(event->type, ncclProfileKernelCh);
}

// API events carry graphCaptured. NCCL only emits them at capture time, so the plugin must latch the
// flag before it drops GroupApi/CollApi (null handle), and must not allocate a slot for them.
TEST_F(ProfilerEventTest, GraphCapturedCollApiSetsCommFlagAndIsStillDropped)
{
    auto* ctx = static_cast<eventContext*>(context);
    ASSERT_FALSE(ctx->commState->sawGraphCapture.load());

    auto descr          = makeCollApiDescr("AllReduce", true);
    void* eHandle       = nullptr;
    ncclResult_t result = profiler_otel_start_event_v5(context, &eHandle, &descr);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(eHandle, nullptr);
    EXPECT_TRUE(ctx->commState->sawGraphCapture.load());
}

TEST_F(ProfilerEventTest, GraphCapturedGroupApiSetsCommFlag)
{
    auto* ctx = static_cast<eventContext*>(context);

    auto descr    = makeGroupApiDescr(true);
    void* eHandle = nullptr;
    EXPECT_EQ(profiler_otel_start_event_v5(context, &eHandle, &descr), ncclSuccess);

    EXPECT_EQ(eHandle, nullptr);
    EXPECT_TRUE(ctx->commState->sawGraphCapture.load());
}

TEST_F(ProfilerEventTest, GraphCapturedP2pApiSetsCommFlagAndKeepsMarker)
{
    auto* ctx = static_cast<eventContext*>(context);

    auto descr    = makeP2pApiDescr("Send", nullptr, true);
    void* eHandle = nullptr;
    EXPECT_EQ(profiler_otel_start_event_v5(context, &eHandle, &descr), ncclSuccess);

    EXPECT_NE(eHandle, nullptr);
    EXPECT_TRUE(ctx->commState->sawGraphCapture.load());
}

TEST_F(ProfilerEventTest, EagerApiEventsDoNotSetGraphCaptureFlag)
{
    auto* ctx = static_cast<eventContext*>(context);

    auto collApi  = makeCollApiDescr("AllReduce", false);
    auto groupApi = makeGroupApiDescr(false);
    auto p2pApi   = makeP2pApiDescr("Send", nullptr, false);
    void* eHandle = nullptr;
    EXPECT_EQ(profiler_otel_start_event_v5(context, &eHandle, &collApi), ncclSuccess);
    EXPECT_EQ(profiler_otel_start_event_v5(context, &eHandle, &groupApi), ncclSuccess);
    EXPECT_EQ(profiler_otel_start_event_v5(context, &eHandle, &p2pApi), ncclSuccess);

    EXPECT_FALSE(ctx->commState->sawGraphCapture.load());
}

// On NCCL >= 2.31 a replayed Coll carries the capture-time CollApi ref as parentObj. The plugin drops
// CollApi, so that ref is null and the replayed Coll must still be tracked normally.
TEST_F(ProfilerEventTest, ReplayedCollWithNullApiParentIsTracked)
{
    auto collApi    = makeCollApiDescr("AllReduce", true);
    void* apiHandle = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v5(context, &apiHandle, &collApi), ncclSuccess);
    ASSERT_EQ(apiHandle, nullptr);

    auto descr    = makeCollDescr("AllReduce", 1024, "ncclInt32", 2, "Ring", "Simple", apiHandle);
    void* eHandle = nullptr;
    EXPECT_EQ(profiler_otel_start_event_v5(context, &eHandle, &descr), ncclSuccess);
    ASSERT_NE(eHandle, nullptr);
    EXPECT_EQ(static_cast<otelEventHandle_t*>(eHandle)->type, ncclProfileColl);
    EXPECT_EQ(profiler_otel_stop_event_v5(eHandle), ncclSuccess);
}
