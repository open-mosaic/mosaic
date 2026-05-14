// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "../../communicator_state.h"
#include "../../events.h"
#include "../../profiler_otel.h"
#include "../../profiler_v4_compat.h"
#include "../test_helpers.h"

static void mock_logger(ncclDebugLogLevel level, unsigned long flags, const char* file, int line, const char* fmt, ...)
{
    (void)level;
    (void)flags;
    (void)file;
    (void)line;
    (void)fmt;
}

// ============================================================================
// Init tests
// ============================================================================

class V4CompatInitTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        unsetenv("NCCL_PROFILER_OTEL_ENABLE");
        unsetenv("NCCL_PROFILE_EVENT_MASK");
        resetProfilerState();
    }
};

TEST_F(V4CompatInitTest, SuccessfulInit)
{
    void* context = nullptr;
    int mask      = 0;

    ncclResult_t ret = profiler_otel_init_v4(&context, &mask, "v4_comm", 12345, 2, 8, 0, mock_logger);

    EXPECT_EQ(ret, ncclSuccess);
    EXPECT_NE(context, nullptr);

    auto* ctx = static_cast<eventContext*>(context);
    EXPECT_STREQ(ctx->commName, "v4_comm");
    EXPECT_EQ(ctx->commHash, 12345u);
    EXPECT_EQ(ctx->nNodes, 2);
    EXPECT_EQ(ctx->nranks, 8);
    EXPECT_EQ(ctx->rank, 0);

    profiler_otel_finalize_v5(context);
}

TEST_F(V4CompatInitTest, MaskStripsHighBits)
{
    void* context = nullptr;
    int mask      = 0;

    ncclResult_t ret = profiler_otel_init_v4(&context, &mask, "v4_comm", 1, 1, 4, 0, mock_logger);
    EXPECT_EQ(ret, ncclSuccess);

    // Default v5 mask is 0x85E (includes KernelLaunch = 1<<11).
    // v4 adapter must strip bits > 7, leaving only bits 0-7.
    EXPECT_EQ(mask & 0xFF, mask);
    EXPECT_EQ(mask & ~0xFF, 0);

    // Coll(1<<1), P2p(1<<2), ProxyOp(1<<3), ProxyStep(1<<4), KernelCh(1<<6)
    // should survive the mask.
    EXPECT_NE(mask & ncclProfileColl, 0);
    EXPECT_NE(mask & ncclProfileP2p, 0);
    EXPECT_NE(mask & ncclProfileProxyOp, 0);
    EXPECT_NE(mask & ncclProfileProxyStep, 0);
    EXPECT_NE(mask & ncclProfileKernelCh, 0);

    // KernelLaunch (1<<11) must NOT be present.
    EXPECT_EQ(mask & ncclProfileKernelLaunch, 0);

    profiler_otel_finalize_v5(context);
}

TEST_F(V4CompatInitTest, DisabledByEnv)
{
    setenv("NCCL_PROFILER_OTEL_ENABLE", "0", 1);

    void* context = nullptr;
    int mask      = 0;

    ncclResult_t ret = profiler_otel_init_v4(&context, &mask, "v4_comm", 1, 1, 4, 0, mock_logger);
    EXPECT_EQ(ret, ncclSuccess);
    EXPECT_EQ(context, nullptr);

    unsetenv("NCCL_PROFILER_OTEL_ENABLE");
}

TEST_F(V4CompatInitTest, CommHashPassedAsCommId)
{
    void* context = nullptr;
    int mask      = 0;
    uint64_t hash = 0xDEAD'BEEF'CAFE'BABEull;

    ncclResult_t ret = profiler_otel_init_v4(&context, &mask, "v4_comm", hash, 1, 4, 0, mock_logger);
    EXPECT_EQ(ret, ncclSuccess);

    auto* ctx = static_cast<eventContext*>(context);
    EXPECT_EQ(ctx->commHash, hash);

    profiler_otel_finalize_v5(context);
}

// ============================================================================
// Event lifecycle tests (v4 descriptor → v5 forwarding)
// ============================================================================

class V4CompatEventTest : public ::testing::Test
{
protected:
    void* context = nullptr;
    int mask      = 0;

    void SetUp() override
    {
        unsetenv("NCCL_PROFILER_OTEL_ENABLE");
        unsetenv("NCCL_PROFILE_EVENT_MASK");
        resetProfilerState();

        ncclResult_t ret = profiler_otel_init_v4(&context, &mask, "v4_comm", 42, 1, 8, 0, mock_logger);
        ASSERT_EQ(ret, ncclSuccess);
        ASSERT_NE(context, nullptr);
    }

    void TearDown() override
    {
        if (context) profiler_otel_finalize_v5(context);
    }

    // ---- v4 descriptor helpers ----

    ncclProfilerEventDescr_v4_t makeCollV4(const char* func = "AllReduce", size_t count = 1024,
                                           const char* datatype = "ncclFloat32", uint8_t nChannels = 4,
                                           const char* algo = "Ring", const char* proto = "Simple",
                                           void* parent = nullptr)
    {
        ncclProfilerEventDescr_v4_t d = {};
        d.type                        = ncclProfileColl;
        d.parentObj                   = parent;
        d.coll.func                   = func;
        d.coll.count                  = count;
        d.coll.datatype               = datatype;
        d.coll.nChannels              = nChannels;
        d.coll.algo                   = algo;
        d.coll.proto                  = proto;
        return d;
    }

    ncclProfilerEventDescr_v4_t makeP2pV4(const char* func = "Send", size_t count = 256,
                                          const char* datatype = "ncclFloat32", int peer = 1, uint8_t nChannels = 2,
                                          void* parent = nullptr)
    {
        ncclProfilerEventDescr_v4_t d = {};
        d.type                        = ncclProfileP2p;
        d.parentObj                   = parent;
        d.p2p.func                    = func;
        d.p2p.count                   = count;
        d.p2p.datatype                = datatype;
        d.p2p.peer                    = peer;
        d.p2p.nChannels               = nChannels;
        return d;
    }

    ncclProfilerEventDescr_v4_t makeProxyOpV4(uint8_t channelId = 0, int peer = 1, int chunkSize = 512, int isSend = 1,
                                              void* parent = nullptr)
    {
        ncclProfilerEventDescr_v4_t d = {};
        d.type                        = ncclProfileProxyOp;
        d.parentObj                   = parent;
        d.proxyOp.pid                 = getpid();
        d.proxyOp.channelId           = channelId;
        d.proxyOp.peer                = peer;
        d.proxyOp.nSteps              = 8;
        d.proxyOp.chunkSize           = chunkSize;
        d.proxyOp.isSend              = isSend;
        return d;
    }

    ncclProfilerEventDescr_v4_t makeProxyStepV4(int step = 0, void* parent = nullptr)
    {
        ncclProfilerEventDescr_v4_t d = {};
        d.type                        = ncclProfileProxyStep;
        d.parentObj                   = parent;
        d.proxyStep.step              = step;
        return d;
    }

    ncclProfilerEventDescr_v4_t makeGroupV4()
    {
        ncclProfilerEventDescr_v4_t d = {};
        d.type                        = ncclProfileGroup;
        return d;
    }

    ncclProfilerEventDescr_v4_t makeKernelChV4(uint8_t channelId = 0, uint64_t pTimer = 100000, void* parent = nullptr)
    {
        ncclProfilerEventDescr_v4_t d = {};
        d.type                        = ncclProfileKernelCh;
        d.parentObj                   = parent;
        d.kernelCh.channelId          = channelId;
        d.kernelCh.pTimer             = pTimer;
        return d;
    }
};

// ---------- Collective ----------

TEST_F(V4CompatEventTest, CollectiveRoundTrip)
{
    auto descr   = makeCollV4("AllReduce", 2048, "ncclFloat32", 4, "Ring", "Simple");
    void* handle = nullptr;

    ASSERT_EQ(profiler_otel_start_event_v4(context, &handle, &descr), ncclSuccess);
    ASSERT_NE(handle, nullptr);

    auto* ev = static_cast<otelEventHandle_t*>(handle);
    EXPECT_EQ(ev->type, (uint64_t)ncclProfileColl);
    EXPECT_STREQ(ev->coll.func, "AllReduce");
    EXPECT_EQ(ev->coll.bytes, 2048u * 4);  // count * sizeof(float32)
    EXPECT_EQ(ev->coll.nChannels, 4);
    EXPECT_STREQ(ev->coll.algo, "Ring");
    EXPECT_STREQ(ev->coll.proto, "Simple");

    EXPECT_EQ(profiler_otel_stop_event_v5(handle), ncclSuccess);
    EXPECT_GT(ev->endTs, 0.0);
}

// ---------- P2P ----------

TEST_F(V4CompatEventTest, P2pSendRoundTrip)
{
    auto descr   = makeP2pV4("Send", 512, "ncclFloat16", 3, 2);
    void* handle = nullptr;

    ASSERT_EQ(profiler_otel_start_event_v4(context, &handle, &descr), ncclSuccess);
    ASSERT_NE(handle, nullptr);

    auto* ev = static_cast<otelEventHandle_t*>(handle);
    EXPECT_EQ(ev->type, (uint64_t)ncclProfileP2p);
    EXPECT_STREQ(ev->p2p.func, "Send");
    EXPECT_EQ(ev->p2p.bytes, 512u * 2);  // count * sizeof(float16)
    EXPECT_EQ(ev->p2p.peer, 3);
    EXPECT_EQ(ev->p2p.nChannels, 2);

    EXPECT_EQ(profiler_otel_stop_event_v5(handle), ncclSuccess);
}

TEST_F(V4CompatEventTest, P2pRecvIsSkipped)
{
    auto descr   = makeP2pV4("Recv", 512, "ncclFloat32", 1, 1);
    void* handle = nullptr;

    ASSERT_EQ(profiler_otel_start_event_v4(context, &handle, &descr), ncclSuccess);
    EXPECT_EQ(handle, nullptr);
}

// ---------- ProxyOp ----------

TEST_F(V4CompatEventTest, ProxyOpRoundTrip)
{
    auto descr   = makeProxyOpV4(3, 5, 1024, 1, (void*)0xABCD);
    void* handle = nullptr;

    ASSERT_EQ(profiler_otel_start_event_v4(context, &handle, &descr), ncclSuccess);
    ASSERT_NE(handle, nullptr);

    auto* ev = static_cast<otelEventHandle_t*>(handle);
    EXPECT_EQ(ev->type, (uint64_t)ncclProfileProxyOp);
    EXPECT_EQ(ev->proxyOp.channelId, 3);
    EXPECT_EQ(ev->proxyOp.peer, 5);
    EXPECT_EQ(ev->proxyOp.chunkSize, 1024);

    EXPECT_EQ(profiler_otel_stop_event_v5(handle), ncclSuccess);
}

TEST_F(V4CompatEventTest, ProxyOpRecvIsSkipped)
{
    auto descr   = makeProxyOpV4(0, 1, 512, 0);  // isSend = 0
    void* handle = nullptr;

    ASSERT_EQ(profiler_otel_start_event_v4(context, &handle, &descr), ncclSuccess);
    EXPECT_EQ(handle, nullptr);
}

// ---------- ProxyStep + recordEventState ----------

TEST_F(V4CompatEventTest, ProxyStepWithSendWait)
{
    auto opDescr   = makeProxyOpV4(0, 1, 1024, 1, (void*)0x1234);
    void* opHandle = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v4(context, &opHandle, &opDescr), ncclSuccess);
    ASSERT_NE(opHandle, nullptr);

    auto stepDescr   = makeProxyStepV4(0, opHandle);
    void* stepHandle = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v4(context, &stepHandle, &stepDescr), ncclSuccess);
    ASSERT_NE(stepHandle, nullptr);

    auto* ev = static_cast<otelEventHandle_t*>(stepHandle);
    EXPECT_FALSE(ev->proxyStep.hasSendWait);

    ncclProfilerEventStateArgs_v4_t args = {};
    args.proxyStep.transSize             = 8192;

    ASSERT_EQ(profiler_otel_record_event_state_v4(stepHandle, ncclProfilerProxyStepSendWait, &args), ncclSuccess);

    EXPECT_TRUE(ev->proxyStep.hasSendWait);
    EXPECT_EQ(ev->proxyStep.transSize, 8192u);
    EXPECT_GT(ev->proxyStep.sendWaitTs, 0.0);

    EXPECT_EQ(profiler_otel_stop_event_v5(stepHandle), ncclSuccess);
    EXPECT_GT(ev->endTs, ev->proxyStep.sendWaitTs);
}

TEST_F(V4CompatEventTest, RecordEventStateNullHandle)
{
    ncclProfilerEventStateArgs_v4_t args = {};
    EXPECT_EQ(profiler_otel_record_event_state_v4(nullptr, ncclProfilerProxyStepSendWait, &args), ncclSuccess);
}

// ---------- Group ----------

TEST_F(V4CompatEventTest, GroupRoundTrip)
{
    auto descr   = makeGroupV4();
    void* handle = nullptr;

    ASSERT_EQ(profiler_otel_start_event_v4(context, &handle, &descr), ncclSuccess);
    ASSERT_NE(handle, nullptr);

    auto* ev = static_cast<otelEventHandle_t*>(handle);
    EXPECT_EQ(ev->type, (uint64_t)ncclProfileGroup);

    EXPECT_EQ(profiler_otel_stop_event_v5(handle), ncclSuccess);
    EXPECT_GT(ev->endTs, 0.0);
}

// ---------- KernelCh ----------

TEST_F(V4CompatEventTest, KernelChRoundTrip)
{
    // KernelCh needs a parent Coll to track
    auto collDescr   = makeCollV4();
    void* collHandle = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v4(context, &collHandle, &collDescr), ncclSuccess);
    ASSERT_NE(collHandle, nullptr);

    auto kcDescr   = makeKernelChV4(2, 500000, collHandle);
    void* kcHandle = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v4(context, &kcHandle, &kcDescr), ncclSuccess);
    ASSERT_NE(kcHandle, nullptr);

    auto* ev = static_cast<otelEventHandle_t*>(kcHandle);
    EXPECT_EQ(ev->type, (uint64_t)ncclProfileKernelCh);
    EXPECT_EQ(ev->kernelCh.channelId, 2);
    EXPECT_EQ(ev->kernelCh.pTimerStart, 500000u);
    EXPECT_FALSE(ev->kernelCh.hasStop);

    // Record KernelChStop state via v4 adapter
    ncclProfilerEventStateArgs_v4_t args = {};
    args.kernelCh.pTimer                 = 600000;
    ASSERT_EQ(profiler_otel_record_event_state_v4(kcHandle, ncclProfilerKernelChStop, &args), ncclSuccess);

    EXPECT_TRUE(ev->kernelCh.hasStop);
    EXPECT_EQ(ev->kernelCh.pTimerStop, 600000u);

    EXPECT_EQ(profiler_otel_stop_event_v5(kcHandle), ncclSuccess);
    EXPECT_GT(ev->endTs, 0.0);
}

TEST_F(V4CompatEventTest, KernelChInProgressTracking)
{
    auto* ctx       = static_cast<eventContext*>(context);
    auto* commState = ctx->commState;
    uint8_t bufIdx  = commState->get_active_buffer_idx();
    auto* window    = commState->get_window_metadata(bufIdx);

    auto collDescr   = makeCollV4();
    void* collHandle = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v4(context, &collHandle, &collDescr), ncclSuccess);
    ASSERT_NE(collHandle, nullptr);

    uint32_t before = window->kernel_ch_in_progress.load();

    auto kcDescr   = makeKernelChV4(0, 0, collHandle);
    void* kcHandle = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v4(context, &kcHandle, &kcDescr), ncclSuccess);
    ASSERT_NE(kcHandle, nullptr);

    EXPECT_EQ(window->kernel_ch_in_progress.load(), before + 1);

    profiler_otel_stop_event_v5(kcHandle);
    EXPECT_EQ(window->kernel_ch_in_progress.load(), before);
}

// ---------- ProxyCtrl is dropped ----------

TEST_F(V4CompatEventTest, ProxyCtrlIsSkipped)
{
    ncclProfilerEventDescr_v4_t descr = {};
    descr.type                        = ncclProfileProxyCtrl;
    void* handle                      = nullptr;

    ASSERT_EQ(profiler_otel_start_event_v4(context, &handle, &descr), ncclSuccess);
    EXPECT_EQ(handle, nullptr);
}

// ---------- Unknown type is dropped ----------

TEST_F(V4CompatEventTest, UnknownTypeIsSkipped)
{
    ncclProfilerEventDescr_v4_t descr = {};
    descr.type                        = 0xFF;  // unknown
    void* handle                      = nullptr;

    ASSERT_EQ(profiler_otel_start_event_v4(context, &handle, &descr), ncclSuccess);
    EXPECT_EQ(handle, nullptr);
}

// ============================================================================
// End-to-end: full collective lifecycle through v4 API
// ============================================================================

TEST_F(V4CompatEventTest, FullCollectiveLifecycle)
{
    // Group → Coll → ProxyOp → ProxyStep (with SendWait) → stop all
    auto groupDescr   = makeGroupV4();
    void* groupHandle = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v4(context, &groupHandle, &groupDescr), ncclSuccess);
    ASSERT_NE(groupHandle, nullptr);

    auto collDescr   = makeCollV4("ReduceScatter", 4096, "ncclFloat32", 2, "Ring", "Simple", groupHandle);
    void* collHandle = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v4(context, &collHandle, &collDescr), ncclSuccess);
    ASSERT_NE(collHandle, nullptr);

    auto opDescr   = makeProxyOpV4(0, 3, 2048, 1, collHandle);
    void* opHandle = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v4(context, &opHandle, &opDescr), ncclSuccess);
    ASSERT_NE(opHandle, nullptr);

    auto stepDescr   = makeProxyStepV4(0, opHandle);
    void* stepHandle = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v4(context, &stepHandle, &stepDescr), ncclSuccess);
    ASSERT_NE(stepHandle, nullptr);

    ncclProfilerEventStateArgs_v4_t args = {};
    args.proxyStep.transSize             = 2048;
    ASSERT_EQ(profiler_otel_record_event_state_v4(stepHandle, ncclProfilerProxyStepSendWait, &args), ncclSuccess);

    EXPECT_EQ(profiler_otel_stop_event_v5(stepHandle), ncclSuccess);
    EXPECT_EQ(profiler_otel_stop_event_v5(opHandle), ncclSuccess);
    EXPECT_EQ(profiler_otel_stop_event_v5(collHandle), ncclSuccess);
    EXPECT_EQ(profiler_otel_stop_event_v5(groupHandle), ncclSuccess);

    // Verify timestamps are monotonically ordered
    auto* stepEv = static_cast<otelEventHandle_t*>(stepHandle);
    auto* opEv   = static_cast<otelEventHandle_t*>(opHandle);
    auto* collEv = static_cast<otelEventHandle_t*>(collHandle);

    EXPECT_GT(stepEv->endTs, stepEv->startTs);
    EXPECT_GT(opEv->endTs, opEv->startTs);
    EXPECT_GT(collEv->endTs, collEv->startTs);
    EXPECT_EQ(collEv->coll.bytes, 4096u * 4);
}

// ============================================================================
// v4 and v5 produce same event data for identical inputs
// ============================================================================

TEST_F(V4CompatEventTest, V4MatchesV5ForColl)
{
    // Start same collective through v4 and v5, compare event data.
    auto v4descr = makeCollV4("AllGather", 512, "ncclInt64", 8, "Tree", "LL");
    void* v4h    = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v4(context, &v4h, &v4descr), ncclSuccess);
    ASSERT_NE(v4h, nullptr);

    ncclProfilerEventDescr_v5_t v5descr = {};
    v5descr.type                        = ncclProfileColl;
    v5descr.coll.func                   = "AllGather";
    v5descr.coll.count                  = 512;
    v5descr.coll.datatype               = "ncclInt64";
    v5descr.coll.nChannels              = 8;
    v5descr.coll.algo                   = "Tree";
    v5descr.coll.proto                  = "LL";
    void* v5h                           = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v5(context, &v5h, &v5descr), ncclSuccess);
    ASSERT_NE(v5h, nullptr);

    auto* e4 = static_cast<otelEventHandle_t*>(v4h);
    auto* e5 = static_cast<otelEventHandle_t*>(v5h);

    EXPECT_EQ(e4->type, e5->type);
    EXPECT_STREQ(e4->coll.func, e5->coll.func);
    EXPECT_EQ(e4->coll.bytes, e5->coll.bytes);
    EXPECT_EQ(e4->coll.nChannels, e5->coll.nChannels);
    EXPECT_STREQ(e4->coll.algo, e5->coll.algo);
    EXPECT_STREQ(e4->coll.proto, e5->coll.proto);

    profiler_otel_stop_event_v5(v4h);
    profiler_otel_stop_event_v5(v5h);
}

TEST_F(V4CompatEventTest, V4MatchesV5ForP2p)
{
    auto v4descr = makeP2pV4("Send", 1024, "ncclBfloat16", 5, 3);
    void* v4h    = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v4(context, &v4h, &v4descr), ncclSuccess);
    ASSERT_NE(v4h, nullptr);

    ncclProfilerEventDescr_v5_t v5descr = {};
    v5descr.type                        = ncclProfileP2p;
    v5descr.p2p.func                    = "Send";
    v5descr.p2p.count                   = 1024;
    v5descr.p2p.datatype                = "ncclBfloat16";
    v5descr.p2p.peer                    = 5;
    v5descr.p2p.nChannels               = 3;
    void* v5h                           = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v5(context, &v5h, &v5descr), ncclSuccess);
    ASSERT_NE(v5h, nullptr);

    auto* e4 = static_cast<otelEventHandle_t*>(v4h);
    auto* e5 = static_cast<otelEventHandle_t*>(v5h);

    EXPECT_EQ(e4->type, e5->type);
    EXPECT_STREQ(e4->p2p.func, e5->p2p.func);
    EXPECT_EQ(e4->p2p.bytes, e5->p2p.bytes);
    EXPECT_EQ(e4->p2p.peer, e5->p2p.peer);
    EXPECT_EQ(e4->p2p.nChannels, e5->p2p.nChannels);

    profiler_otel_stop_event_v5(v4h);
    profiler_otel_stop_event_v5(v5h);
}
