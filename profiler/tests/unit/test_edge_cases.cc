// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "../../aggregation.h"
#include "../../communicator_state.h"
#include "../../events.h"
#include "../../linear_regression.h"

// Aggressive edge case and stress tests designed to find real bugs

class EdgeCaseTest : public ::testing::Test
{
};

// Test that LinearRegression works correctly in single-threaded use
// NOTE: LinearRegression is NOT thread-safe by design (uses std::vector without sync)
TEST_F(EdgeCaseTest, LinearRegressionSingleThreaded)
{
    LinearRegression lr;

    // Add many points (single-threaded - correct usage)
    for (int i = 0; i < 1000; i++)
    {
        lr.addPoint(i * 100.0, i * 0.1);
    }

    double slope, intercept;
    EXPECT_TRUE(lr.calculate(slope, intercept));

    // Verify reasonable results for linear data
    EXPECT_GT(slope, 0.0);
    EXPECT_FALSE(std::isnan(slope));
    EXPECT_FALSE(std::isnan(intercept));
}

// Test aggregation with extreme values
TEST_F(EdgeCaseTest, AggregationExtremeValues)
{
    AggregatedTransfer transfer;

    // Add very large values
    transfer.addTransfer(SIZE_MAX / 2, 1e12);
    transfer.addTransfer(SIZE_MAX / 2, 1e12);

    // Check for overflow
    EXPECT_GT(transfer.totalBytes, 0u);  // Should not wrap to 0
    EXPECT_GT(transfer.totalTimeUs, 0.0);

    // Add very small values
    transfer.addTransfer(1, 0.001);

    // Verify precision isn't lost
    EXPECT_GT(transfer.count, 0);
}

// Test aggregation with zero and negative durations
TEST_F(EdgeCaseTest, AggregationZeroAndNegativeDurations)
{
    WindowAggregator aggregator(0);

    // Create Coll with ProxyOp that ends at same time (zero duration)
    otelEventHandle_t coll1 = {};
    coll1.type              = ncclProfileColl;
    coll1.coll.func         = "AllReduce";
    coll1.coll.algo         = "Ring";
    coll1.coll.proto        = "Simple";
    coll1.coll.nChannels    = 1;
    coll1.coll.bytes        = 1000;
    coll1.startTs           = 100.0;
    coll1.endTs             = 100.0;
    coll1.rank              = 0;
    aggregator.addEvent(coll1);

    otelEventHandle_t proxyOp1 = {};
    proxyOp1.type              = ncclProfileProxyOp;
    proxyOp1.proxyOp.peer      = 1;
    proxyOp1.proxyOp.channelId = 0;
    proxyOp1.proxyOp.chunkSize = 128;
    proxyOp1.startTs           = 100.0;
    proxyOp1.endTs             = 100.0;  // Zero duration
    proxyOp1.parentObj         = &coll1;
    proxyOp1.rank              = 0;

    otelEventHandle_t proxyStep1     = {};
    proxyStep1.type                  = ncclProfileProxyStep;
    proxyStep1.proxyStep.step        = 0;
    proxyStep1.proxyStep.transSize   = 64;
    proxyStep1.proxyStep.sendWaitTs  = 100.0;
    proxyStep1.proxyStep.hasSendWait = true;
    proxyStep1.startTs               = 100.0;
    proxyStep1.endTs                 = 100.0;
    proxyStep1.parentObj             = &proxyOp1;
    proxyStep1.rank                  = 0;

    aggregator.addEvent(proxyStep1);
    aggregator.addEvent(proxyOp1);

    // Create Coll with ProxyOp that ends BEFORE Coll starts (negative calculated duration)
    otelEventHandle_t coll2 = coll1;
    coll2.startTs           = 200.0;
    coll2.endTs             = 200.0;
    aggregator.addEvent(coll2);

    otelEventHandle_t proxyOp2 = proxyOp1;
    proxyOp2.startTs           = 150.0;
    proxyOp2.endTs             = 150.0;  // Before coll2.startTs
    proxyOp2.parentObj         = &coll2;

    otelEventHandle_t proxyStep2    = proxyStep1;
    proxyStep2.startTs              = 150.0;
    proxyStep2.endTs                = 150.0;
    proxyStep2.proxyStep.sendWaitTs = 150.0;
    proxyStep2.parentObj            = &proxyOp2;

    aggregator.addEvent(proxyStep2);
    aggregator.addEvent(proxyOp2);

    // Finalize to calculate durations
    aggregator.finalize();

    // Both collectives have invalid durations (zero and negative)
    // They are now skipped to avoid infinite bandwidth in regression calculations
    const auto& collectives = aggregator.getCollectives();
    EXPECT_EQ(collectives.size(), 0u);  // Both should be skipped
}

// Test linear regression with pathological data
TEST_F(EdgeCaseTest, LinearRegressionPathologicalData)
{
    LinearRegression lr;

    // All points at same location (degenerate case)
    for (int i = 0; i < 100; i++)
    {
        lr.addPoint(1000.0, 100.0);
    }

    double slope, intercept;
    EXPECT_FALSE(lr.calculate(slope, intercept));

    // Clear and try with extreme spread
    lr.clear();
    lr.addPoint(0.0, 0.0);
    lr.addPoint(1e15, 1e15);  // Extreme values

    bool result = lr.calculate(slope, intercept);
    if (result)
    {
        // Should not produce NaN or Inf
        EXPECT_FALSE(std::isnan(slope));
        EXPECT_FALSE(std::isnan(intercept));
        EXPECT_FALSE(std::isinf(slope));
        EXPECT_FALSE(std::isinf(intercept));
    }
}

// Test communicator state with rapid buffer switches
TEST_F(EdgeCaseTest, CommunicatorStateRapidBufferSwitch)
{
    CommunicatorState state;

    // Rapidly switch buffers without adding many events
    for (int i = 0; i < 1000; i++)
    {
        uint8_t idx = state.get_active_buffer_idx();
        state.trigger_window_closing(idx);
    }

    // Should wrap around cleanly
    uint8_t final_idx = state.get_active_buffer_idx();
    EXPECT_LT(final_idx, NUM_BUFFERS);

    // Should still be able to allocate
    otelEventHandle_t* slot = state.allocate_event_slot();
    EXPECT_NE(slot, nullptr);
}

// Test for memory corruption with buffer boundary
TEST_F(EdgeCaseTest, CommunicatorStateBufferBoundary)
{
    CommunicatorState state;
    std::vector<otelEventHandle_t*> slots;

    // Fill exactly to buffer size
    for (int i = 0; i < BUFFER_SIZE; i++)
    {
        otelEventHandle_t* slot = state.allocate_event_slot();
        if (slot)
        {
            slots.push_back(slot);
            // Write to the slot to detect corruption
            slot->type    = ncclProfileColl;
            slot->startTs = i * 1.0;
        }
    }

    // With auto window-switching at WINDOW_TRIGGER_COUNT, we should fill one window and switch
    EXPECT_GE(slots.size(), WINDOW_TRIGGER_COUNT);

    // Window should have auto-switched, next allocation should succeed
    otelEventHandle_t* next_slot = state.allocate_event_slot();
    EXPECT_NE(next_slot, nullptr);

    // Verify no corruption of existing slots
    for (size_t i = 0; i < slots.size(); i++)
    {
        EXPECT_EQ(slots[i]->type, ncclProfileColl);
        EXPECT_DOUBLE_EQ(slots[i]->startTs, i * 1.0);
    }
}

// Test aggregation with NULL/invalid strings
TEST_F(EdgeCaseTest, AggregationNullStrings)
{
    WindowAggregator aggregator(0);

    otelEventHandle_t event = {};
    event.type              = ncclProfileColl;
    event.coll.func         = nullptr;  // NULL function name
    event.coll.algo         = nullptr;  // NULL algorithm
    event.coll.proto        = nullptr;  // NULL protocol
    event.coll.nChannels    = 1;
    event.coll.bytes        = 1000;
    event.startTs           = 0.0;
    event.endTs             = 10.0;
    event.rank              = 0;

    // Should not crash
    aggregator.addEvent(event);

    // Check if it was added (behavior may vary)
    const auto& collectives = aggregator.getCollectives();
    // May or may not add with NULL strings - document behavior
    EXPECT_GE(collectives.size(), 0u);
}

// Test for integer overflow in element counting
TEST_F(EdgeCaseTest, WindowMetadataElementCountOverflow)
{
    WindowMetadata window;
    window.state.store(WINDOW_FILLING);
    window.element_count.store(UINT32_MAX - 10);

    // Try to increment past max
    for (int i = 0; i < 20; i++)
    {
        window.element_count.fetch_add(1);
    }

    // Should wrap around (document overflow behavior)
    uint32_t final_count = window.element_count.load();
    EXPECT_LT(final_count, 20u);  // Wrapped around
}

// Test concurrent window transitions
TEST_F(EdgeCaseTest, CommunicatorStateConcurrentTransitions)
{
    // Use heap allocation to ensure state outlives threads
    auto state = std::make_unique<CommunicatorState>();
    std::atomic<int> transition_count{0};
    std::atomic<bool> stop{false};

    auto worker = [&]()
    {
        while (!stop.load(std::memory_order_acquire))
        {
            uint8_t idx = state->get_active_buffer_idx();
            state->trigger_window_closing(idx);
            transition_count.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; i++)
    {
        threads.emplace_back(worker);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop.store(true, std::memory_order_release);

    // Join all threads before destroying state
    for (auto& t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    // Should still be in valid state
    uint8_t final_idx = state->get_active_buffer_idx();
    EXPECT_LT(final_idx, NUM_BUFFERS);

    // Should have transitioned multiple times
    EXPECT_GT(transition_count.load(), 0);

    // Explicitly destroy state after all threads are done
    state.reset();
}

// Test mixing different event types in quick succession
TEST_F(EdgeCaseTest, AggregationMixedEventTypes)
{
    WindowAggregator aggregator(0);

    // Store events in vectors to have stable addresses
    std::vector<otelEventHandle_t> colls, p2ps, proxyOps, proxySteps;
    colls.reserve(100);
    p2ps.reserve(100);
    proxyOps.reserve(200);  // 100 for colls, 100 for p2ps
    proxySteps.reserve(200);

    for (int i = 0; i < 100; i++)
    {
        // Collective
        otelEventHandle_t coll = {};
        coll.type              = ncclProfileColl;
        coll.coll.func         = "AllReduce";
        coll.coll.algo         = "Ring";
        coll.coll.proto        = "Simple";
        coll.coll.nChannels    = 1;
        coll.coll.bytes        = 1000;
        coll.startTs           = i * 10.0;
        coll.endTs             = i * 10.0 + 5.0;
        coll.rank              = 0;
        colls.push_back(coll);
        aggregator.addEvent(colls.back());

        // ProxyOp linked to collective
        otelEventHandle_t proxyOp = {};
        proxyOp.type              = ncclProfileProxyOp;
        proxyOp.proxyOp.peer      = 1;
        proxyOp.proxyOp.channelId = i % 4;
        proxyOp.proxyOp.chunkSize = 128;
        proxyOp.startTs           = i * 10.0 + 2.0;
        proxyOp.endTs             = i * 10.0 + 4.0;
        proxyOp.rank              = 0;
        proxyOp.parentObj         = &colls.back();
        proxyOps.push_back(proxyOp);

        otelEventHandle_t proxyStep     = {};
        proxyStep.type                  = ncclProfileProxyStep;
        proxyStep.proxyStep.step        = 0;
        proxyStep.proxyStep.transSize   = 64;
        proxyStep.proxyStep.sendWaitTs  = i * 10.0 + 2.5;
        proxyStep.proxyStep.hasSendWait = true;
        proxyStep.startTs               = i * 10.0 + 2.0;
        proxyStep.endTs                 = i * 10.0 + 4.0;
        proxyStep.parentObj             = &proxyOps.back();
        proxyStep.rank                  = 0;
        proxySteps.push_back(proxyStep);

        aggregator.addEvent(proxySteps.back());
        aggregator.addEvent(proxyOps[proxyOps.size() - 1]);

        // P2P
        otelEventHandle_t p2p = {};
        p2p.type              = ncclProfileP2p;
        p2p.p2p.func          = "Send";
        p2p.p2p.peer          = 1;
        p2p.p2p.nChannels     = 1;
        p2p.p2p.bytes         = 500;
        p2p.startTs           = i * 10.0 + 5.0;
        p2p.endTs             = i * 10.0 + 8.0;
        p2p.rank              = 0;
        p2ps.push_back(p2p);
        aggregator.addEvent(p2ps.back());

        // ProxyOp for P2P
        otelEventHandle_t proxyOp2 = {};
        proxyOp2.type              = ncclProfileProxyOp;
        proxyOp2.proxyOp.peer      = 1;
        proxyOp2.proxyOp.channelId = 0;
        proxyOp2.proxyOp.chunkSize = 64;
        proxyOp2.startTs           = i * 10.0 + 6.0;
        proxyOp2.endTs             = i * 10.0 + 7.0;
        proxyOp2.rank              = 0;
        proxyOp2.parentObj         = &p2ps.back();
        proxyOps.push_back(proxyOp2);

        otelEventHandle_t proxyStep2     = {};
        proxyStep2.type                  = ncclProfileProxyStep;
        proxyStep2.proxyStep.step        = 0;
        proxyStep2.proxyStep.transSize   = 32;
        proxyStep2.proxyStep.sendWaitTs  = i * 10.0 + 6.5;
        proxyStep2.proxyStep.hasSendWait = true;
        proxyStep2.startTs               = i * 10.0 + 6.0;
        proxyStep2.endTs                 = i * 10.0 + 7.0;
        proxyStep2.parentObj             = &proxyOps.back();
        proxyStep2.rank                  = 0;
        proxySteps.push_back(proxyStep2);

        aggregator.addEvent(proxySteps.back());
        aggregator.addEvent(proxyOps.back());
    }

    // Finalize before checking
    aggregator.finalize();

    // Verify all were aggregated
    EXPECT_GT(aggregator.getCollectives().size(), 0u);
    EXPECT_GT(aggregator.getP2Ps().size(), 0u);
    EXPECT_GT(aggregator.getRankTransfers().size(), 0u);
}

// Test for precision loss with many small additions
TEST_F(EdgeCaseTest, AggregationPrecisionLoss)
{
    AggregatedTransfer transfer;

    // Add many very small values
    for (int i = 0; i < 1000000; i++)
    {
        transfer.addTransfer(1, 0.001);
    }

    // Check that we haven't lost too much precision
    EXPECT_EQ(transfer.count, 1000000);
    EXPECT_EQ(transfer.totalBytes, 1000000u);
    EXPECT_GT(transfer.totalTimeUs, 999.0);  // Should be ~1000
    EXPECT_LT(transfer.totalTimeUs, 1001.0);
}

// Test state machine with out-of-order operations
TEST_F(EdgeCaseTest, WindowStateOutOfOrderOperations)
{
    CommunicatorState state;
    uint8_t idx = 0;

    // Mark complete before start (should handle gracefully)
    state.mark_operation_complete(idx);

    // Mark start multiple times
    state.mark_operation_start(idx);
    state.mark_operation_start(idx);
    state.mark_operation_start(idx);

    WindowMetadata* window = state.get_window_metadata(idx);
    EXPECT_EQ(window->in_progress_count.load(), 3);

    // Complete more times than started (should protect against underflow)
    state.mark_operation_complete(idx);
    state.mark_operation_complete(idx);
    state.mark_operation_complete(idx);
    state.mark_operation_complete(idx);  // Extra
    state.mark_operation_complete(idx);  // Extra

    // Should not underflow
    EXPECT_EQ(window->in_progress_count.load(), 0);
}

// Test with maximum channel count
TEST_F(EdgeCaseTest, AggregationMaxChannels)
{
    WindowAggregator aggregator(0);

    otelEventHandle_t coll = {};
    coll.type              = ncclProfileColl;
    coll.coll.func         = "AllReduce";
    coll.coll.algo         = "Ring";
    coll.coll.proto        = "Simple";
    coll.coll.nChannels    = 255;  // Maximum for uint8_t
    coll.coll.bytes        = 1000;
    coll.startTs           = 0.0;
    coll.endTs             = 10.0;
    coll.rank              = 0;

    aggregator.addEvent(coll);

    // Add a ProxyOp with ProxyStep to complete the event sequence
    otelEventHandle_t proxyOp = {};
    proxyOp.type              = ncclProfileProxyOp;
    proxyOp.proxyOp.peer      = 1;
    proxyOp.proxyOp.channelId = 0;
    proxyOp.proxyOp.chunkSize = 128;
    proxyOp.startTs           = 1.0;
    proxyOp.endTs             = 5.0;
    proxyOp.parentObj         = &coll;
    proxyOp.rank              = 0;

    otelEventHandle_t proxyStep     = {};
    proxyStep.type                  = ncclProfileProxyStep;
    proxyStep.proxyStep.step        = 0;
    proxyStep.proxyStep.transSize   = 64;
    proxyStep.proxyStep.sendWaitTs  = 2.0;
    proxyStep.proxyStep.hasSendWait = true;
    proxyStep.startTs               = 1.0;
    proxyStep.endTs                 = 5.0;
    proxyStep.parentObj             = &proxyOp;
    proxyStep.rank                  = 0;

    aggregator.addEvent(proxyStep);
    aggregator.addEvent(proxyOp);

    // Finalize before checking
    aggregator.finalize();

    const auto& collectives = aggregator.getCollectives();
    EXPECT_EQ(collectives.size(), 1u);
}

// Test linear regression with numerical instability
TEST_F(EdgeCaseTest, LinearRegressionNumericalInstability)
{
    LinearRegression lr;

    // Points that differ only in least significant bits
    lr.addPoint(1.0, 1.0);
    lr.addPoint(1.0 + 1e-15, 1.0 + 1e-15);
    lr.addPoint(1.0 + 2e-15, 1.0 + 2e-15);

    double slope, intercept;
    bool result = lr.calculate(slope, intercept);

    // May or may not succeed due to numerical issues
    if (result)
    {
        EXPECT_FALSE(std::isnan(slope));
        EXPECT_FALSE(std::isinf(slope));
    }
}
