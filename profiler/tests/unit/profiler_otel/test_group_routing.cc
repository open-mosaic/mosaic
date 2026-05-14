// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "../../support/profiler_event_fixture.h"

using testsupport::makeCollDescr;
using testsupport::makeGroupDescr;
using testsupport::makeP2pApiDescr;
using testsupport::makeP2pDescr;

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

    auto* collEvent = static_cast<otelEventHandle_t*>(collHandle);
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

TEST_F(ProfilerEventTest, P2PInsideClosingGroupRoutesToGroupWindow)
{
    auto* ctx                = static_cast<eventContext*>(context);
    CommunicatorState* state = ctx->commState;

    auto groupDescr   = makeGroupDescr();
    void* groupHandle = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v5(context, &groupHandle, &groupDescr), ncclSuccess);
    ASSERT_NE(groupHandle, nullptr);

    auto* groupEvent = static_cast<otelEventHandle_t*>(groupHandle);
    uint8_t groupBuf = groupEvent->buffer_idx;
    EXPECT_EQ(groupBuf, 0);

    state->trigger_window_closing(groupBuf);
    EXPECT_EQ(state->get_window_metadata(groupBuf)->state.load(), WINDOW_CLOSING);
    EXPECT_EQ(state->get_active_buffer_idx(), 1);

    auto p2pApiDescr   = makeP2pApiDescr("Send");
    void* p2pApiHandle = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v5(context, &p2pApiHandle, &p2pApiDescr), ncclSuccess);
    ASSERT_NE(p2pApiHandle, nullptr);

    auto* p2pApiEvent = static_cast<otelEventHandle_t*>(p2pApiHandle);
    EXPECT_EQ(p2pApiEvent->buffer_idx, 1);

    auto p2pDescr   = makeP2pDescr("Send", 512, "ncclFloat32", 0, 1, p2pApiHandle);
    void* p2pHandle = nullptr;
    ASSERT_EQ(profiler_otel_start_event_v5(context, &p2pHandle, &p2pDescr), ncclSuccess);
    ASSERT_NE(p2pHandle, nullptr);

    auto* p2pEvent = static_cast<otelEventHandle_t*>(p2pHandle);
    EXPECT_EQ(p2pEvent->buffer_idx, groupBuf);
    EXPECT_EQ(p2pEvent->parentObj, p2pApiHandle);

    EXPECT_EQ(profiler_otel_stop_event_v5(p2pHandle), ncclSuccess);
    EXPECT_EQ(profiler_otel_stop_event_v5(p2pApiHandle), ncclSuccess);
    EXPECT_EQ(profiler_otel_stop_event_v5(groupHandle), ncclSuccess);
}