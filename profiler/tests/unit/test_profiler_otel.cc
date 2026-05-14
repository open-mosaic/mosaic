// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>

#include "../../events.h"
#include "../../profiler_otel.h"
#include "../test_helpers.h"

class OtelProfilerInitTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        unsetenv("NCCL_PROFILER_OTEL_ENABLE");
        unsetenv("NCCL_PROFILE_EVENT_MASK");

        extern ncclDebugLogger_t otel_log_func;
        setInitialized(0);
        otel_log_func = nullptr;
        setPid(0);
        reset_unregister_communicator_tracking();
    }

    static void mock_logger(ncclDebugLogLevel level, unsigned long flags, const char* file, int line, const char* fmt,
                            ...)
    {
        (void)level;
        (void)flags;
        (void)file;
        (void)line;
        (void)fmt;
    }

    static constexpr uint64_t kDefaultCommId = 12345;
    static constexpr int kDefaultNNodes      = 2;
    static constexpr int kDefaultNranks      = 4;
    static constexpr int kDefaultRank        = 0;

    ncclResult_t initProfiler(void** ctx, int* mask, const char* name = "test_comm",
                              ncclDebugLogger_t logger = mock_logger)
    {
        return profiler_otel_init_v5(ctx, kDefaultCommId, mask, name, kDefaultNNodes, kDefaultNranks, kDefaultRank,
                                     logger);
    }
};

TEST_F(OtelProfilerInitTest, SuccessfulInitialization)
{
    void* context       = nullptr;
    int eActivationMask = 0;

    ncclResult_t result = initProfiler(&context, &eActivationMask);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_NE(context, nullptr);

    struct eventContext* ctx = static_cast<struct eventContext*>(context);
    EXPECT_STREQ(ctx->commName, "test_comm");
    EXPECT_EQ(ctx->commHash, kDefaultCommId);
    EXPECT_EQ(ctx->nNodes, kDefaultNNodes);
    EXPECT_EQ(ctx->nranks, kDefaultNranks);
    EXPECT_EQ(ctx->rank, kDefaultRank);

    extern ncclDebugLogger_t otel_log_func;
    EXPECT_EQ(otel_log_func, mock_logger);
}

TEST_F(OtelProfilerInitTest, DisabledByEnvironmentVariable)
{
    setenv("NCCL_PROFILER_OTEL_ENABLE", "0", 1);
    void* context       = nullptr;
    int eActivationMask = 0;

    ncclResult_t result = initProfiler(&context, &eActivationMask);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(context, nullptr);

    unsetenv("NCCL_PROFILER_OTEL_ENABLE");
}

TEST_F(OtelProfilerInitTest, NullLoggerUsesFallback)
{
    void* context       = nullptr;
    int eActivationMask = 0;

    ncclResult_t result = initProfiler(&context, &eActivationMask, "test_comm", nullptr);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_NE(context, nullptr);

    extern ncclDebugLogger_t otel_log_func;
    EXPECT_NE(otel_log_func, nullptr);
}

TEST_F(OtelProfilerInitTest, EventMaskFromEnvironment)
{
    setenv("NCCL_PROFILE_EVENT_MASK", "42", 1);
    void* context       = nullptr;
    int eActivationMask = 0;

    ncclResult_t result = initProfiler(&context, &eActivationMask);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(eActivationMask, 42);
}

TEST_F(OtelProfilerInitTest, EventMaskDefaultIncludesKernelEvents)
{
    void* context       = nullptr;
    int eActivationMask = 0;

    ncclResult_t result = initProfiler(&context, &eActivationMask);

    EXPECT_EQ(result, ncclSuccess);
    // 0x85E = Coll | P2p | ProxyOp | ProxyStep | KernelCh | KernelLaunch
    EXPECT_EQ(eActivationMask, 0x85E);
}

TEST_F(OtelProfilerInitTest, ProcessIdIsSet)
{
    void* context       = nullptr;
    int eActivationMask = 0;
    pid_t expectedPid   = getpid();

    ncclResult_t result = initProfiler(&context, &eActivationMask);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(getPid(), expectedPid);
}

TEST_F(OtelProfilerInitTest, MultipleInitializationCalls)
{
    void* context1       = nullptr;
    void* context2       = nullptr;
    int eActivationMask1 = 0;
    int eActivationMask2 = 0;

    ncclResult_t result1 = initProfiler(&context1, &eActivationMask1);
    ncclResult_t result2 =
        profiler_otel_init_v5(&context2, 67890, &eActivationMask2, "test_comm2", 2, 4, 1, mock_logger);

    EXPECT_EQ(result1, ncclSuccess);
    EXPECT_EQ(result2, ncclSuccess);
    EXPECT_NE(context1, nullptr);
    EXPECT_NE(context2, nullptr);

    struct eventContext* ctx1 = static_cast<struct eventContext*>(context1);
    struct eventContext* ctx2 = static_cast<struct eventContext*>(context2);
    EXPECT_STREQ(ctx1->commName, "test_comm");
    EXPECT_STREQ(ctx2->commName, "test_comm2");
}

TEST_F(OtelProfilerInitTest, DifferentRankValues)
{
    void* context       = nullptr;
    int eActivationMask = 0;
    int rank            = 3;

    ncclResult_t result =
        profiler_otel_init_v5(&context, kDefaultCommId, &eActivationMask, "test_comm", 2, 4, rank, mock_logger);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_NE(context, nullptr);

    struct eventContext* ctx = static_cast<struct eventContext*>(context);
    EXPECT_EQ(ctx->rank, rank);
}

TEST_F(OtelProfilerInitTest, DifferentCommIdValues)
{
    void* context       = nullptr;
    int eActivationMask = 0;
    uint64_t commId     = 98765;

    ncclResult_t result = profiler_otel_init_v5(&context, commId, &eActivationMask, "test_comm", 2, 4, 0, mock_logger);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_NE(context, nullptr);

    struct eventContext* ctx = static_cast<struct eventContext*>(context);
    EXPECT_EQ(ctx->commHash, commId);
}

TEST_F(OtelProfilerInitTest, DifferentNodeAndRankCounts)
{
    void* context       = nullptr;
    int eActivationMask = 0;
    int nNodes          = 8;
    int nranks          = 16;

    ncclResult_t result =
        profiler_otel_init_v5(&context, kDefaultCommId, &eActivationMask, "test_comm", nNodes, nranks, 0, mock_logger);

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_NE(context, nullptr);

    struct eventContext* ctx = static_cast<struct eventContext*>(context);
    EXPECT_EQ(ctx->nNodes, nNodes);
    EXPECT_EQ(ctx->nranks, nranks);
}

TEST_F(OtelProfilerInitTest, StartTimeIsInitialized)
{
    void* context       = nullptr;
    int eActivationMask = 0;

    ncclResult_t result = initProfiler(&context, &eActivationMask);

    EXPECT_EQ(result, ncclSuccess);
    double baseTime = 1234567890.0 * 1e6;
    EXPECT_GT(getStartTime(), baseTime - 1);
    EXPECT_LT(getStartTime(), baseTime + 1e9);
    EXPECT_GT(getStartTime(), 0.0);
}

TEST_F(OtelProfilerInitTest, FinalizeUnregistersCommunicatorBeforeDestroy)
{
    void* context       = nullptr;
    int eActivationMask = 0;

    ncclResult_t result = initProfiler(&context, &eActivationMask);
    ASSERT_EQ(result, ncclSuccess);
    ASSERT_NE(context, nullptr);

    auto* ctx                    = static_cast<eventContext*>(context);
    CommunicatorState* commState = ctx->commState;

    EXPECT_EQ(0, get_unregister_communicator_call_count());
    EXPECT_EQ(profiler_otel_finalize_v5(context), ncclSuccess);
    EXPECT_EQ(1, get_unregister_communicator_call_count());
    EXPECT_EQ(commState, get_last_unregistered_communicator());
}
