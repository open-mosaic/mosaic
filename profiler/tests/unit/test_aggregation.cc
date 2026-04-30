// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

#include "../../aggregation.h"
#include "../../communicator_state.h"
#include "../../events.h"

// Test fixture for AggregatedTransfer
class AggregatedTransferTest : public ::testing::Test
{
protected:
    AggregatedTransfer transfer;
};

TEST_F(AggregatedTransferTest, InitialState)
{
    EXPECT_EQ(transfer.totalBytes, 0u);
    EXPECT_EQ(transfer.totalTimeUs, 0.0);
    EXPECT_EQ(transfer.count, 0);
}

TEST_F(AggregatedTransferTest, SingleTransfer)
{
    transfer.addTransfer(1024, 10.5);

    EXPECT_EQ(transfer.totalBytes, 1024u);
    EXPECT_DOUBLE_EQ(transfer.totalTimeUs, 10.5);
    EXPECT_EQ(transfer.count, 1);
    EXPECT_DOUBLE_EQ(transfer.getAverageSize(), 1024.0);
    EXPECT_DOUBLE_EQ(transfer.getAverageTime(), 10.5);
    EXPECT_DOUBLE_EQ(transfer.getAverageRateMBps(), 1024.0 / 10.5);
}

TEST_F(AggregatedTransferTest, MultipleTransfers)
{
    transfer.addTransfer(1000, 10.0);
    transfer.addTransfer(2000, 20.0);
    transfer.addTransfer(3000, 30.0);

    EXPECT_EQ(transfer.totalBytes, 6000u);
    EXPECT_DOUBLE_EQ(transfer.totalTimeUs, 60.0);
    EXPECT_EQ(transfer.count, 3);
    EXPECT_DOUBLE_EQ(transfer.getAverageSize(), 2000.0);
    EXPECT_DOUBLE_EQ(transfer.getAverageTime(), 20.0);
}

TEST_F(AggregatedTransferTest, ZeroBytes)
{
    transfer.addTransfer(0, 10.0);

    EXPECT_EQ(transfer.totalBytes, 0u);
    EXPECT_DOUBLE_EQ(transfer.totalTimeUs, 10.0);
    EXPECT_EQ(transfer.count, 1);
    EXPECT_DOUBLE_EQ(transfer.getAverageSize(), 0.0);
}

TEST_F(AggregatedTransferTest, ZeroTime)
{
    transfer.addTransfer(1000, 0.0);

    EXPECT_EQ(transfer.totalBytes, 1000u);
    EXPECT_DOUBLE_EQ(transfer.totalTimeUs, 0.0);
    EXPECT_DOUBLE_EQ(transfer.getAverageRateMBps(), 0.0);  // Should handle division by zero
}

// =============================================================================
// Tests for Linear Regression Latency Calculation
// Rate is calculated using getRateFromActiveTime() instead of 1/slope.
// Use getLatencyFromLinearRegression() for latency.
// =============================================================================

TEST_F(AggregatedTransferTest, LinearRegressionWithTwoPoints)
{
    // Only 2 different sizes - should fail (need at least 3)
    transfer.addTransfer(0, 10.0);
    transfer.addTransfer(1000, 11.0);

    double latency;
    EXPECT_FALSE(transfer.getLatencyFromLinearRegression(latency));
    EXPECT_DOUBLE_EQ(latency, 0.0);
}

TEST_F(AggregatedTransferTest, LinearRegressionWithOnePoint)
{
    transfer.addTransfer(1000, 10.0);

    double latency;
    EXPECT_FALSE(transfer.getLatencyFromLinearRegression(latency));
    EXPECT_DOUBLE_EQ(latency, 0.0);
}

TEST_F(AggregatedTransferTest, LinearRegressionWithNoPoints)
{
    double latency;
    EXPECT_FALSE(transfer.getLatencyFromLinearRegression(latency));
}

TEST_F(AggregatedTransferTest, LinearRegressionWithIdenticalSizes)
{
    // All same size - should fail (vertical line)
    transfer.addTransfer(1000, 10.0);
    transfer.addTransfer(1000, 11.0);
    transfer.addTransfer(1000, 12.0);

    double latency;
    EXPECT_FALSE(transfer.getLatencyFromLinearRegression(latency));
}

TEST_F(AggregatedTransferTest, LinearRegressionWithZeroSlope)
{
    // Constant time - slope = 0 (infinite bandwidth) - should be rejected as not physically meaningful
    transfer.addTransfer(0, 10.0);
    transfer.addTransfer(1000, 10.0);
    transfer.addTransfer(2000, 10.0);
    transfer.addTransfer(3000, 10.0);

    double latency;
    EXPECT_FALSE(transfer.getLatencyFromLinearRegression(latency));
    EXPECT_DOUBLE_EQ(latency, 0.0);
}

TEST_F(AggregatedTransferTest, LinearRegressionWithNegativeSlope)
{
    // Negative slope (time decreases with size) - physically meaningless for bandwidth
    transfer.addTransfer(0, 100.0);
    transfer.addTransfer(1000, 90.0);
    transfer.addTransfer(2000, 80.0);

    double latency;
    EXPECT_FALSE(transfer.getLatencyFromLinearRegression(latency));
    EXPECT_DOUBLE_EQ(latency, 0.0);
}

TEST_F(AggregatedTransferTest, LinearRegressionRealisticCase)
{
    // Realistic case: 10us latency, 0.01 us/byte (100 MB/s)
    transfer.addTransfer(0, 10.0);
    transfer.addTransfer(1000, 20.0);
    transfer.addTransfer(2000, 30.0);
    transfer.addTransfer(5000, 60.0);

    double latency;
    EXPECT_TRUE(transfer.getLatencyFromLinearRegression(latency));
    EXPECT_NEAR(latency, 10.0, 0.5);
}

// =============================================================================
// Tests for New Latency Method (getLatencyFromLinearRegression)
// This is the recommended method for latency calculation.
// =============================================================================

TEST_F(AggregatedTransferTest, LatencyFromLinearRegressionBasic)
{
    // Realistic case: 10us latency
    transfer.addTransfer(0, 10.0);
    transfer.addTransfer(1000, 20.0);
    transfer.addTransfer(2000, 30.0);
    transfer.addTransfer(5000, 60.0);

    double latency;
    EXPECT_TRUE(transfer.getLatencyFromLinearRegression(latency));
    EXPECT_NEAR(latency, 10.0, 0.5);
}

TEST_F(AggregatedTransferTest, LatencyFromLinearRegressionInsufficientData)
{
    // Only 2 points - should fail
    transfer.addTransfer(0, 10.0);
    transfer.addTransfer(1000, 20.0);

    double latency;
    EXPECT_FALSE(transfer.getLatencyFromLinearRegression(latency));
    EXPECT_DOUBLE_EQ(latency, 0.0);
}

TEST_F(AggregatedTransferTest, LatencyFromLinearRegressionNegativeSlope)
{
    // Negative slope (time decreases with size) - should fail
    transfer.addTransfer(0, 100.0);
    transfer.addTransfer(1000, 90.0);
    transfer.addTransfer(2000, 80.0);

    double latency;
    EXPECT_FALSE(transfer.getLatencyFromLinearRegression(latency));
    EXPECT_DOUBLE_EQ(latency, 0.0);
}

// Test fixture for AggregatedOperationBase
class AggregatedOperationBaseTest : public ::testing::Test
{
protected:
    AggregatedOperationBase operation;
};

TEST_F(AggregatedOperationBaseTest, InitialState)
{
    EXPECT_EQ(operation.totalBytes, 0u);
    EXPECT_EQ(operation.totalTimeUs, 0.0);
    EXPECT_EQ(operation.count, 0);
    EXPECT_EQ(operation.cachedTotalTransferCount, 0);
    EXPECT_EQ(operation.cachedTotalTransferBytes, 0u);
    EXPECT_DOUBLE_EQ(operation.cachedTotalTransferTimeUs, 0.0);
}

TEST_F(AggregatedOperationBaseTest, AddOperation)
{
    operation.addOperation(2048, 15.5);

    EXPECT_EQ(operation.totalBytes, 2048u);
    EXPECT_DOUBLE_EQ(operation.totalTimeUs, 15.5);
    EXPECT_EQ(operation.count, 1);
    EXPECT_DOUBLE_EQ(operation.getAverageSize(), 2048.0);
    EXPECT_DOUBLE_EQ(operation.getAverageTime(), 15.5);
}

TEST_F(AggregatedOperationBaseTest, MultipleOperations)
{
    operation.addOperation(1000, 10.0);
    operation.addOperation(2000, 20.0);
    operation.addOperation(3000, 30.0);

    EXPECT_EQ(operation.count, 3);
    EXPECT_EQ(operation.totalBytes, 6000u);
    EXPECT_DOUBLE_EQ(operation.totalTimeUs, 60.0);
    EXPECT_DOUBLE_EQ(operation.getAverageSize(), 2000.0);
    EXPECT_DOUBLE_EQ(operation.getAverageTime(), 20.0);
}

TEST_F(AggregatedOperationBaseTest, AddTransferToCache)
{
    operation.addTransferToCache(512, 5.0);
    operation.addTransferToCache(1024, 10.0);

    EXPECT_EQ(operation.cachedTotalTransferCount, 2);
    EXPECT_EQ(operation.cachedTotalTransferBytes, 1536u);
    EXPECT_DOUBLE_EQ(operation.cachedTotalTransferTimeUs, 15.0);
}

TEST_F(AggregatedOperationBaseTest, GetTotalTransferCount)
{
    EXPECT_EQ(operation.getTotalTransferCount(), 0);

    operation.addTransferToCache(100, 1.0);
    EXPECT_EQ(operation.getTotalTransferCount(), 1);

    operation.addTransferToCache(200, 2.0);
    EXPECT_EQ(operation.getTotalTransferCount(), 2);
}

TEST_F(AggregatedOperationBaseTest, GetAverageTransferCountWithNoOperations)
{
    operation.addTransferToCache(100, 1.0);
    EXPECT_DOUBLE_EQ(operation.getAverageTransferCount(), 0.0);  // No operations added yet
}

TEST_F(AggregatedOperationBaseTest, GetAverageTransferCountWithOperations)
{
    operation.addOperation(1000, 10.0);
    operation.addOperation(2000, 20.0);
    operation.addTransferToCache(100, 1.0);
    operation.addTransferToCache(200, 2.0);
    operation.addTransferToCache(300, 3.0);
    operation.addTransferToCache(400, 4.0);

    EXPECT_EQ(operation.count, 2);
    EXPECT_EQ(operation.cachedTotalTransferCount, 4);
    EXPECT_DOUBLE_EQ(operation.getAverageTransferCount(), 2.0);  // 4 transfers / 2 operations
}

TEST_F(AggregatedOperationBaseTest, GetAverageTransferSizeWithNoTransfers)
{
    EXPECT_DOUBLE_EQ(operation.getAverageTransferSize(), 0.0);
}

TEST_F(AggregatedOperationBaseTest, GetAverageTransferSizeWithTransfers)
{
    operation.addTransferToCache(100, 1.0);
    operation.addTransferToCache(200, 2.0);
    operation.addTransferToCache(300, 3.0);

    double avgSize = (100.0 + 200.0 + 300.0) / 3.0;
    EXPECT_DOUBLE_EQ(operation.getAverageTransferSize(), avgSize);
}

TEST_F(AggregatedOperationBaseTest, GetAverageTransferTimeWithNoTransfers)
{
    EXPECT_DOUBLE_EQ(operation.getAverageTransferTime(), 0.0);
}

TEST_F(AggregatedOperationBaseTest, GetAverageTransferTimeWithTransfers)
{
    operation.addTransferToCache(100, 1.0);
    operation.addTransferToCache(200, 3.0);
    operation.addTransferToCache(300, 5.0);

    double avgTime = (1.0 + 3.0 + 5.0) / 3.0;
    EXPECT_DOUBLE_EQ(operation.getAverageTransferTime(), avgTime);
}

// Test fixture for AggregatedP2P
class AggregatedP2PTest : public ::testing::Test
{
protected:
    AggregatedP2P p2p;
};

TEST_F(AggregatedP2PTest, InheritsFromBase)
{
    // Should have all base class functionality
    p2p.addOperation(1000, 10.0);
    EXPECT_EQ(p2p.count, 1);
    EXPECT_EQ(p2p.totalBytes, 1000u);
}

TEST_F(AggregatedP2PTest, ConvenienceMethodDelegates)
{
    p2p.addP2P(2000, 20.0);
    EXPECT_EQ(p2p.count, 1);
    EXPECT_EQ(p2p.totalBytes, 2000u);
    EXPECT_DOUBLE_EQ(p2p.totalTimeUs, 20.0);
}

// Test fixture for AggregatedCollective
class AggregatedCollectiveTest : public ::testing::Test
{
protected:
    AggregatedCollective collective;
};

TEST_F(AggregatedCollectiveTest, InheritsFromBase)
{
    collective.addOperation(3000, 30.0);
    EXPECT_EQ(collective.count, 1);
    EXPECT_EQ(collective.totalBytes, 3000u);
}

TEST_F(AggregatedCollectiveTest, ConvenienceMethodDelegates)
{
    collective.addCollective(4000, 40.0);
    EXPECT_EQ(collective.count, 1);
    EXPECT_EQ(collective.totalBytes, 4000u);
    EXPECT_DOUBLE_EQ(collective.totalTimeUs, 40.0);
}

// Test fixture for WindowAggregator
class WindowAggregatorTest : public ::testing::Test
{
protected:
    WindowAggregator* aggregator;

    void SetUp() override
    {
        aggregator = new WindowAggregator(0);  // rank 0
    }

    void TearDown() override
    {
        delete aggregator;
    }

    otelEventHandle_t createCollectiveEvent(const char* func, const char* algo, const char* proto, uint8_t channels,
                                            size_t bytes, double startTs, double endTs)
    {
        otelEventHandle_t event = {};
        event.type              = ncclProfileColl;
        event.coll.func         = func;
        event.coll.algo         = algo;
        event.coll.proto        = proto;
        event.coll.nChannels    = channels;
        event.coll.bytes        = bytes;
        event.startTs           = startTs;
        event.endTs             = endTs;
        event.parentObj         = (void*)0x1234;
        event.rank              = 0;
        return event;
    }

    otelEventHandle_t createP2PEvent(const char* func, int peer, uint8_t channels, size_t bytes, double startTs,
                                     double endTs)
    {
        otelEventHandle_t event = {};
        event.type              = ncclProfileP2p;
        event.p2p.func          = func;
        event.p2p.peer          = peer;
        event.p2p.nChannels     = channels;
        event.p2p.bytes         = bytes;
        event.startTs           = startTs;
        event.endTs             = endTs;
        event.parentObj         = (void*)0x5678;
        event.rank              = 0;
        return event;
    }

    otelEventHandle_t createProxyOpEvent(int peer, uint8_t channelId, int chunkSize, double startTs, double endTs,
                                         void* parentObj = nullptr)
    {
        otelEventHandle_t event = {};
        event.type              = ncclProfileProxyOp;
        event.proxyOp.peer      = peer;
        event.proxyOp.channelId = channelId;
        event.proxyOp.chunkSize = chunkSize;
        event.startTs           = startTs;
        event.endTs             = endTs;
        event.parentObj         = parentObj;
        event.rank              = 0;
        return event;
    }

    otelEventHandle_t createProxyStepEvent(int step, size_t transSize, double startTs, double sendWaitTs, double endTs,
                                           void* parentObj = nullptr)
    {
        otelEventHandle_t event     = {};
        event.type                  = ncclProfileProxyStep;
        event.proxyStep.step        = step;
        event.proxyStep.transSize   = transSize;
        event.proxyStep.sendWaitTs  = sendWaitTs;
        event.proxyStep.hasSendWait = (transSize > 0);
        event.startTs               = startTs;
        event.endTs                 = endTs;
        event.parentObj             = parentObj;
        event.rank                  = 0;
        return event;
    }

    otelEventHandle_t createKernelChEvent(uint8_t channelId, uint64_t pTimerStart, uint64_t pTimerStop, double startTs,
                                          double endTs, void* parentObj = nullptr)
    {
        otelEventHandle_t event    = {};
        event.type                 = ncclProfileKernelCh;
        event.kernelCh.channelId   = channelId;
        event.kernelCh.pTimerStart = pTimerStart;
        event.kernelCh.pTimerStop  = pTimerStop;
        event.kernelCh.hasStop     = true;
        event.startTs              = startTs;
        event.endTs                = endTs;
        event.parentObj            = parentObj;
        event.rank                 = 0;
        return event;
    }

    // Creates a collective event with an explicit CommunicatorState so that
    // op.nRanks and op.peer are derived correctly and inferCollectiveTransfers
    // returns actual transfers (needed when testing the timingReliable path).
    otelEventHandle_t createCollectiveEventWithCommState(const char* func, const char* algo, const char* proto,
                                                         uint8_t channels, size_t bytes, double startTs, double endTs,
                                                         CommunicatorState* commState)
    {
        auto event      = createCollectiveEvent(func, algo, proto, channels, bytes, startTs, endTs);
        event.commState = commState;
        return event;
    }
};

TEST_F(WindowAggregatorTest, InitialState)
{
    EXPECT_TRUE(aggregator->getCollectives().empty());
    EXPECT_TRUE(aggregator->getP2Ps().empty());
    EXPECT_TRUE(aggregator->getRankTransfers().empty());
    EXPECT_TRUE(aggregator->getChannelTransfers().empty());
}

TEST_F(WindowAggregatorTest, AggregatesSingleCollective)
{
    // Create Coll event (tracked but not aggregated yet)
    auto coll = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 0.0, 10.0);
    aggregator->addEvent(coll);

    // Create ProxyOp events that reference this Coll
    auto proxyOp1 = createProxyOpEvent(1, 0, 262144, 10.0, 200.0, &coll);

    // Create ProxyStep with SendWait for ProxyOp1
    auto proxyStep1 = createProxyStepEvent(0, 512, 10.0, 100.0, 200.0, &proxyOp1);

    // Add in sequence: Coll, ProxyStep, ProxyOp
    aggregator->addEvent(proxyStep1);
    aggregator->addEvent(proxyOp1);

    // Finalize to calculate Coll duration
    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    EXPECT_EQ(collectives.size(), 1u);

    auto it = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(it, collectives.end());
    EXPECT_EQ(it->second.count, 1);
    EXPECT_EQ(it->second.totalBytes, 1024u);
    // Duration is now from Coll.start (0.0) to ProxyOp.end (200.0)
    EXPECT_DOUBLE_EQ(it->second.totalTimeUs, 200.0);
}

TEST_F(WindowAggregatorTest, AggregatesMultipleCollectivesWithSameKey)
{
    // First Coll
    auto coll1 = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 0.0, 10.0);
    aggregator->addEvent(coll1);
    auto proxyOp1   = createProxyOpEvent(1, 0, 262144, 10.0, 100.0, &coll1);
    auto proxyStep1 = createProxyStepEvent(0, 512, 10.0, 50.0, 100.0, &proxyOp1);
    aggregator->addEvent(proxyStep1);
    aggregator->addEvent(proxyOp1);

    // Second Coll (same key)
    auto coll2 = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 2048, 100.0, 110.0);
    aggregator->addEvent(coll2);
    auto proxyOp2   = createProxyOpEvent(1, 0, 262144, 110.0, 200.0, &coll2);
    auto proxyStep2 = createProxyStepEvent(0, 1024, 110.0, 150.0, 200.0, &proxyOp2);
    aggregator->addEvent(proxyStep2);
    aggregator->addEvent(proxyOp2);

    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    EXPECT_EQ(collectives.size(), 1u);

    auto it = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(it, collectives.end());
    EXPECT_EQ(it->second.count, 2);
    EXPECT_EQ(it->second.totalBytes, 3072u);
    // Durations: coll1 (0->100=100us) + coll2 (100->200=100us) = 200us
    EXPECT_DOUBLE_EQ(it->second.totalTimeUs, 200.0);
}

TEST_F(WindowAggregatorTest, AggregatesMultipleCollectivesWithDifferentKeys)
{
    // First Coll
    auto coll1 = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 0.0, 10.0);
    aggregator->addEvent(coll1);
    auto proxyOp1   = createProxyOpEvent(1, 0, 262144, 10.0, 100.0, &coll1);
    auto proxyStep1 = createProxyStepEvent(0, 512, 10.0, 50.0, 100.0, &proxyOp1);
    aggregator->addEvent(proxyStep1);
    aggregator->addEvent(proxyOp1);

    // Second Coll (different key)
    auto coll2 = createCollectiveEvent("AllGather", "Tree", "LL", 4, 2048, 100.0, 110.0);
    aggregator->addEvent(coll2);
    auto proxyOp2   = createProxyOpEvent(1, 0, 262144, 110.0, 200.0, &coll2);
    auto proxyStep2 = createProxyStepEvent(0, 1024, 110.0, 150.0, 200.0, &proxyOp2);
    aggregator->addEvent(proxyStep2);
    aggregator->addEvent(proxyOp2);

    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    EXPECT_EQ(collectives.size(), 2u);
    EXPECT_TRUE(collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl") != collectives.end());
    EXPECT_TRUE(collectives.find("Comm0_AllGather_Tree_LL_4Chnl") != collectives.end());
}

TEST_F(WindowAggregatorTest, AggregatesSingleP2P)
{
    // Create P2P event
    auto p2p = createP2PEvent("Send", 3, 2, 512, 0.0, 5.0);
    aggregator->addEvent(p2p);

    // Create ProxyOp that references this P2P
    auto proxyOp   = createProxyOpEvent(3, 0, 262144, 5.0, 50.0, &p2p);
    auto proxyStep = createProxyStepEvent(0, 256, 5.0, 25.0, 50.0, &proxyOp);
    aggregator->addEvent(proxyStep);
    aggregator->addEvent(proxyOp);

    aggregator->finalize();

    const auto& p2ps = aggregator->getP2Ps();
    EXPECT_EQ(p2ps.size(), 1u);

    // New P2P key format: Comm<hash>_(<hostname>)_<func>_Pipeline<src>ToPipeline<peer>_<nChannels>Chnl
    // When commState is NULL: hash=0, hostname="unknown", src=rank=0
    auto it = p2ps.find("Comm0_(unknown)_Send_Pipeline0ToPipeline3_2Chnl");
    ASSERT_NE(it, p2ps.end());
    EXPECT_EQ(it->second.count, 1);
    EXPECT_EQ(it->second.totalBytes, 512u);
    // Duration from P2P.start (0.0) to ProxyOp.end (50.0)
    EXPECT_DOUBLE_EQ(it->second.totalTimeUs, 50.0);
}

TEST_F(WindowAggregatorTest, AggregatesSingleProxyOp)
{
    // Create ProxyOp with ProxyStep transfer data
    auto proxyOp   = createProxyOpEvent(2, 1, 256, 0.0, 3.0);
    auto proxyStep = createProxyStepEvent(0, 128, 0.0, 1.0, 3.0, &proxyOp);
    aggregator->addEvent(proxyStep);
    aggregator->addEvent(proxyOp);
    aggregator->finalize();  // Must finalize to link ProxyOps to transfers

    const auto& rankTransfers = aggregator->getRankTransfers();
    EXPECT_EQ(rankTransfers.size(), 1u);

    // New key format: Comm<hash>_Rank<X>_ToPeer<peer> for COLLECTIVE (commState NULL = not P2P)
    auto it = rankTransfers.find("Comm0_Rank0_ToPeer2");
    ASSERT_NE(it, rankTransfers.end());
    EXPECT_EQ(it->second.count, 1);
    // Uses ProxyStep.transSize (128), not ProxyOp.chunkSize (256)
    EXPECT_EQ(it->second.totalBytes, 128u);
}

TEST_F(WindowAggregatorTest, LinksProxyOpToCollective)
{
    // First add a collective
    auto collEvent = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 0.0, 10.0);
    aggregator->addEvent(collEvent);

    // Then add a proxy op with parentObj pointing to the Coll event
    auto proxyOp   = createProxyOpEvent(1, 0, 128, 1.0, 2.0, &collEvent);
    auto proxyStep = createProxyStepEvent(0, 64, 1.0, 1.5, 2.0, &proxyOp);
    aggregator->addEvent(proxyStep);
    aggregator->addEvent(proxyOp);

    aggregator->finalize();

    // Check that the collective has the transfer cached
    const auto& collectives = aggregator->getCollectives();
    auto it                 = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(it, collectives.end());
    EXPECT_EQ(it->second.cachedTotalTransferCount, 1);
    // Uses ProxyStep.transSize (64), not ProxyOp.chunkSize (128)
    EXPECT_EQ(it->second.cachedTotalTransferBytes, 64u);
    EXPECT_DOUBLE_EQ(it->second.cachedTotalTransferTimeUs, 0.5);  // 2.0 - 1.5 = 0.5
}

TEST_F(WindowAggregatorTest, LinksMultipleProxyOpsToCollective)
{
    auto collEvent = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 0.0, 10.0);
    aggregator->addEvent(collEvent);

    // Add multiple proxy ops with ProxySteps
    auto proxyOp1   = createProxyOpEvent(1, 0, 128, 1.0, 2.0, &collEvent);
    auto proxyStep1 = createProxyStepEvent(0, 100, 1.0, 1.5, 2.0, &proxyOp1);  // transfer time = 0.5
    aggregator->addEvent(proxyStep1);
    aggregator->addEvent(proxyOp1);

    auto proxyOp2   = createProxyOpEvent(1, 1, 256, 2.0, 4.0, &collEvent);
    auto proxyStep2 = createProxyStepEvent(0, 200, 2.0, 3.0, 4.0, &proxyOp2);  // transfer time = 1.0
    aggregator->addEvent(proxyStep2);
    aggregator->addEvent(proxyOp2);

    auto proxyOp3   = createProxyOpEvent(2, 0, 512, 3.0, 6.0, &collEvent);
    auto proxyStep3 = createProxyStepEvent(0, 300, 3.0, 4.5, 6.0, &proxyOp3);  // transfer time = 1.5
    aggregator->addEvent(proxyStep3);
    aggregator->addEvent(proxyOp3);

    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    auto it                 = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(it, collectives.end());
    EXPECT_EQ(it->second.cachedTotalTransferCount, 3);
    // Uses ProxyStep.transSize: 100 + 200 + 300 = 600
    EXPECT_EQ(it->second.cachedTotalTransferBytes, 600u);
    // Transfer times: 0.5 + 1.0 + 1.5 = 3.0
    EXPECT_DOUBLE_EQ(it->second.cachedTotalTransferTimeUs, 3.0);
    EXPECT_DOUBLE_EQ(it->second.getAverageTransferCount(), 3.0);
}

TEST_F(WindowAggregatorTest, LinksProxyOpToP2P)
{
    auto p2pEvent = createP2PEvent("Send", 3, 2, 512, 0.0, 5.0);
    aggregator->addEvent(p2pEvent);

    auto proxyOp   = createProxyOpEvent(3, 0, 64, 1.0, 1.5, &p2pEvent);
    auto proxyStep = createProxyStepEvent(0, 32, 1.0, 1.2, 1.5, &proxyOp);
    aggregator->addEvent(proxyStep);
    aggregator->addEvent(proxyOp);

    aggregator->finalize();

    const auto& p2ps = aggregator->getP2Ps();
    // New P2P key format
    auto it = p2ps.find("Comm0_(unknown)_Send_Pipeline0ToPipeline3_2Chnl");
    ASSERT_NE(it, p2ps.end());
    EXPECT_EQ(it->second.cachedTotalTransferCount, 1);
    // Uses ProxyStep.transSize (32), not ProxyOp.chunkSize (64)
    EXPECT_EQ(it->second.cachedTotalTransferBytes, 32u);
}

TEST_F(WindowAggregatorTest, HandlesNegativeDuration)
{
    // ProxyOp.endTs < Coll.startTs should result in 0 duration
    auto coll = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 10.0, 15.0);
    aggregator->addEvent(coll);

    // ProxyOp ends BEFORE Coll starts (negative duration)
    auto proxyOp   = createProxyOpEvent(1, 0, 128, 0.0, 5.0, &coll);
    auto proxyStep = createProxyStepEvent(0, 64, 0.0, 2.0, 5.0, &proxyOp);
    aggregator->addEvent(proxyStep);
    aggregator->addEvent(proxyOp);

    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    auto it                 = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(it, collectives.end());
    EXPECT_DOUBLE_EQ(it->second.totalTimeUs, 0.0);  // Should be clamped to 0
}

TEST_F(WindowAggregatorTest, HandlesZeroDuration)
{
    auto coll = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 10.0, 15.0);
    aggregator->addEvent(coll);

    // ProxyOp ends at same time as Coll starts (zero duration)
    auto proxyOp   = createProxyOpEvent(1, 0, 128, 5.0, 10.0, &coll);
    auto proxyStep = createProxyStepEvent(0, 64, 5.0, 7.0, 10.0, &proxyOp);
    aggregator->addEvent(proxyStep);
    aggregator->addEvent(proxyOp);

    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    auto it                 = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(it, collectives.end());
    EXPECT_DOUBLE_EQ(it->second.totalTimeUs, 0.0);
}

TEST_F(WindowAggregatorTest, HandlesProxyOpWithoutParent)
{
    // Proxy op without parentObj should still be aggregated in rankTransfers
    auto proxyOp   = createProxyOpEvent(1, 0, 128, 0.0, 1.0, nullptr);
    auto proxyStep = createProxyStepEvent(0, 64, 0.0, 0.5, 1.0, &proxyOp);
    aggregator->addEvent(proxyStep);
    aggregator->addEvent(proxyOp);
    aggregator->finalize();  // Must finalize to link ProxyOps to transfers

    const auto& rankTransfers = aggregator->getRankTransfers();
    EXPECT_EQ(rankTransfers.size(), 1u);
    // New key format: Comm<hash>_Rank<X>_ToPeer<peer> for COLLECTIVE (commState NULL = not P2P)
    auto it = rankTransfers.find("Comm0_Rank0_ToPeer1");
    ASSERT_NE(it, rankTransfers.end());
    EXPECT_EQ(it->second.count, 1);
}

TEST_F(WindowAggregatorTest, ComplexScenarioWithMultipleEventTypes)
{
    // Collective 1
    auto coll1 = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 2048, 0.0, 20.0);
    aggregator->addEvent(coll1);

    // Proxy ops for collective 1 (with ProxySteps)
    auto proxyOp1   = createProxyOpEvent(1, 0, 256, 1.0, 2.0, &coll1);
    auto proxyStep1 = createProxyStepEvent(0, 128, 1.0, 1.5, 2.0, &proxyOp1);
    aggregator->addEvent(proxyStep1);
    aggregator->addEvent(proxyOp1);

    auto proxyOp2   = createProxyOpEvent(1, 1, 256, 2.0, 3.0, &coll1);
    auto proxyStep2 = createProxyStepEvent(0, 128, 2.0, 2.5, 3.0, &proxyOp2);
    aggregator->addEvent(proxyStep2);
    aggregator->addEvent(proxyOp2);

    // P2P
    auto p2p1 = createP2PEvent("Send", 2, 1, 1024, 20.0, 30.0);
    aggregator->addEvent(p2p1);

    // Proxy op for P2P (with ProxyStep)
    auto proxyOp3   = createProxyOpEvent(2, 0, 128, 21.0, 22.0, &p2p1);
    auto proxyStep3 = createProxyStepEvent(0, 64, 21.0, 21.5, 22.0, &proxyOp3);
    aggregator->addEvent(proxyStep3);
    aggregator->addEvent(proxyOp3);

    // Standalone proxy op (with ProxyStep)
    auto proxyOp4   = createProxyOpEvent(3, 0, 512, 30.0, 35.0, nullptr);
    auto proxyStep4 = createProxyStepEvent(0, 256, 30.0, 32.0, 35.0, &proxyOp4);
    aggregator->addEvent(proxyStep4);
    aggregator->addEvent(proxyOp4);

    // Finalize to calculate durations
    aggregator->finalize();

    // Verify aggregation
    EXPECT_EQ(aggregator->getCollectives().size(), 1u);
    EXPECT_EQ(aggregator->getP2Ps().size(), 1u);
    EXPECT_EQ(aggregator->getRankTransfers().size(), 3u);  // Rank0->1, Rank0->2, Rank0->3

    // Check collective has transfers
    auto collIt = aggregator->getCollectives().find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(collIt, aggregator->getCollectives().end());
    EXPECT_EQ(collIt->second.cachedTotalTransferCount, 2);

    // Check P2P has transfer (new key format)
    auto p2pIt = aggregator->getP2Ps().find("Comm0_(unknown)_Send_Pipeline0ToPipeline2_1Chnl");
    ASSERT_NE(p2pIt, aggregator->getP2Ps().end());
    EXPECT_EQ(p2pIt->second.cachedTotalTransferCount, 1);
}

TEST_F(WindowAggregatorTest, StressTestManyEvents)
{
    // Stress test: add thousands of events
    // For simplicity, create complete event sequences (Coll/P2P with ProxyOp+ProxyStep)
    const int NUM_COLLS              = 1000;
    const int NUM_P2PS               = 500;
    const int NUM_STANDALONE_PROXIES = 500;

    // Create collectives with ProxyOps
    for (int i = 0; i < NUM_COLLS; i++)
    {
        auto coll = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, i * 10.0, i * 10.0 + 5.0);
        aggregator->addEvent(coll);

        // Add 2 ProxyOps for this Coll
        for (int ch = 0; ch < 2; ch++)
        {
            auto proxyOp = createProxyOpEvent(i % 4, ch, 128, i * 10.0 + ch, i * 10.0 + ch + 1.0, &coll);
            auto proxyStep =
                createProxyStepEvent(0, 64, i * 10.0 + ch, i * 10.0 + ch + 0.5, i * 10.0 + ch + 1.0, &proxyOp);
            aggregator->addEvent(proxyStep);
            aggregator->addEvent(proxyOp);
        }
    }

    // Create P2Ps with ProxyOps
    for (int i = 0; i < NUM_P2PS; i++)
    {
        auto p2p = createP2PEvent("Send", i % 4, 1, 512, i * 5.0, i * 5.0 + 3.0);
        aggregator->addEvent(p2p);

        auto proxyOp   = createProxyOpEvent(i % 4, 0, 64, i * 5.0, i * 5.0 + 2.0, &p2p);
        auto proxyStep = createProxyStepEvent(0, 32, i * 5.0, i * 5.0 + 1.0, i * 5.0 + 2.0, &proxyOp);
        aggregator->addEvent(proxyStep);
        aggregator->addEvent(proxyOp);
    }

    // Create standalone ProxyOps (orphaned)
    for (int i = 0; i < NUM_STANDALONE_PROXIES; i++)
    {
        auto proxyOp   = createProxyOpEvent(i % 4, i % 8, 128, i * 2.0, i * 2.0 + 1.0, nullptr);
        auto proxyStep = createProxyStepEvent(0, 64, i * 2.0, i * 2.0 + 0.5, i * 2.0 + 1.0, &proxyOp);
        aggregator->addEvent(proxyStep);
        aggregator->addEvent(proxyOp);
    }

    // Finalize before checking
    aggregator->finalize();

    // Should have aggregated many events
    EXPECT_GT(aggregator->getCollectives().size(), 0u);
    EXPECT_GT(aggregator->getP2Ps().size(), 0u);
    EXPECT_GT(aggregator->getRankTransfers().size(), 0u);
}

TEST_F(WindowAggregatorTest, StressTestDeepProxyOpNesting)
{
    // Create one collective with many proxy ops
    auto coll = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 0.0, 100.0);
    aggregator->addEvent(coll);

    // Add 1000 proxy ops (with ProxySteps) to the same collective
    for (int i = 0; i < 1000; i++)
    {
        auto proxy     = createProxyOpEvent(i % 4, i % 8, 128 + i, i * 0.1, i * 0.1 + 0.05, &coll);
        auto proxyStep = createProxyStepEvent(0, 64 + i, i * 0.1, i * 0.1 + 0.02, i * 0.1 + 0.05, &proxy);
        aggregator->addEvent(proxyStep);
        aggregator->addEvent(proxy);
    }

    // Finalize before checking
    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    auto it                 = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(it, collectives.end());

    // Check cache was properly updated
    EXPECT_EQ(it->second.cachedTotalTransferCount, 1000);
    EXPECT_GT(it->second.cachedTotalTransferBytes, 0u);
    EXPECT_GT(it->second.cachedTotalTransferTimeUs, 0.0);

    // Verify O(1) access works correctly
    EXPECT_DOUBLE_EQ(it->second.getAverageTransferCount(), 1000.0);
}

TEST_F(WindowAggregatorTest, OrphanedProxyOps)
{
    // Add proxy ops before their parent collectives (orphaned)
    // These ProxyOps point to a fake parent that doesn't exist yet
    auto proxy1     = createProxyOpEvent(1, 0, 128, 1.0, 2.0, (void*)0x1234);
    auto proxyStep1 = createProxyStepEvent(0, 64, 1.0, 1.5, 2.0, &proxy1);
    aggregator->addEvent(proxyStep1);
    aggregator->addEvent(proxy1);

    auto proxy2     = createProxyOpEvent(1, 1, 256, 2.0, 4.0, (void*)0x1234);
    auto proxyStep2 = createProxyStepEvent(0, 128, 2.0, 3.0, 4.0, &proxy2);
    aggregator->addEvent(proxyStep2);
    aggregator->addEvent(proxy2);

    // Add parent collective later (but with different identity, so still orphaned)
    // Since no ProxyOps are linked, the collective uses endTs-startTs for duration
    // (internal links fallback behavior)
    auto coll = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 0.0, 10.0);
    aggregator->addEvent(coll);

    // Finalize to calculate durations
    aggregator->finalize();

    // Orphaned proxy ops should still be in rankTransfers
    EXPECT_GT(aggregator->getRankTransfers().size(), 0u);

    // Collective without linked ProxyOps now uses endTs-startTs for duration (internal links fallback)
    // So it should be found with duration = 10.0 - 0.0 = 10.0 us
    const auto& collectives = aggregator->getCollectives();
    auto it                 = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    EXPECT_NE(it, collectives.end());  // Should be found (uses endTs for duration)
    if (it != collectives.end())
    {
        EXPECT_EQ(it->second.count, 1u);
    }
}

// =============================================================================
// Tests for Interval-Based Rate Calculation
// =============================================================================

// Test fixture for interval merging and rate calculation
class AggregatedTransferIntervalTest : public ::testing::Test
{
protected:
    AggregatedTransfer transfer;
};

TEST_F(AggregatedTransferIntervalTest, EmptyIntervalsReturnsZeroActiveTime)
{
    // No intervals added
    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 0.0);

    double rate;
    EXPECT_FALSE(transfer.getRateFromActiveTime(rate));
    EXPECT_DOUBLE_EQ(rate, 0.0);
}

TEST_F(AggregatedTransferIntervalTest, SingleIntervalActiveTime)
{
    // Single transfer from t=10 to t=20 (10 us duration)
    transfer.addTransferWithTimestamps(1000, 10.0, 10.0, 20.0);

    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 10.0);
    EXPECT_EQ(transfer.count, 1);
    EXPECT_EQ(transfer.totalBytes, 1000u);

    double rate;
    EXPECT_TRUE(transfer.getRateFromActiveTime(rate));
    // 1000 bytes / 10 us = 100 MB/s
    EXPECT_DOUBLE_EQ(rate, 100.0);
}

TEST_F(AggregatedTransferIntervalTest, NonOverlappingIntervals)
{
    // Two non-overlapping transfers with a gap:
    // Transfer A: t=0 to t=10 (1000 bytes)
    // Transfer B: t=20 to t=30 (2000 bytes)
    // Gap between t=10 and t=20
    transfer.addTransferWithTimestamps(1000, 10.0, 0.0, 10.0);
    transfer.addTransferWithTimestamps(2000, 10.0, 20.0, 30.0);

    // Active time = (10 - 0) + (30 - 20) = 20 us
    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 20.0);
    EXPECT_EQ(transfer.totalBytes, 3000u);

    double rate;
    EXPECT_TRUE(transfer.getRateFromActiveTime(rate));
    // 3000 bytes / 20 us = 150 MB/s
    EXPECT_DOUBLE_EQ(rate, 150.0);
}

TEST_F(AggregatedTransferIntervalTest, OverlappingIntervalsSimple)
{
    // Two overlapping transfers:
    // Transfer A: t=0 to t=20 (1000 bytes)
    // Transfer B: t=10 to t=30 (2000 bytes)
    // Overlap between t=10 and t=20
    transfer.addTransferWithTimestamps(1000, 20.0, 0.0, 20.0);
    transfer.addTransferWithTimestamps(2000, 20.0, 10.0, 30.0);

    // Merged: [0, 30], Active time = 30 us
    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 30.0);
    EXPECT_EQ(transfer.totalBytes, 3000u);

    double rate;
    EXPECT_TRUE(transfer.getRateFromActiveTime(rate));
    // 3000 bytes / 30 us = 100 MB/s
    EXPECT_DOUBLE_EQ(rate, 100.0);
}

TEST_F(AggregatedTransferIntervalTest, FullyContainedInterval)
{
    // Transfer B is fully contained within Transfer A:
    // Transfer A: t=0 to t=100 (5000 bytes)
    // Transfer B: t=20 to t=50 (2000 bytes)
    transfer.addTransferWithTimestamps(5000, 100.0, 0.0, 100.0);
    transfer.addTransferWithTimestamps(2000, 30.0, 20.0, 50.0);

    // Merged: [0, 100], Active time = 100 us
    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 100.0);
    EXPECT_EQ(transfer.totalBytes, 7000u);

    double rate;
    EXPECT_TRUE(transfer.getRateFromActiveTime(rate));
    // 7000 bytes / 100 us = 70 MB/s
    EXPECT_DOUBLE_EQ(rate, 70.0);
}

TEST_F(AggregatedTransferIntervalTest, ComplexOverlappingScenario)
{
    // Complex scenario from the user's example:
    // Transfer A: t0=0 to t2=30 (starts first, ends third)
    // Transfer B: t1=10 to t3=50 (starts second, ends fourth)
    // Transfer C: t4=60 to t5=80 (separate, no overlap)
    //
    // Timeline:
    // |----A----|
    //      |------B------|
    //                         |--C--|
    // 0   10   20   30   40   50   60   70   80
    //
    // Merged intervals: [0, 50] and [60, 80]
    // Active time = 50 + 20 = 70 us

    transfer.addTransferWithTimestamps(1000, 30.0, 0.0, 30.0);   // A
    transfer.addTransferWithTimestamps(2000, 40.0, 10.0, 50.0);  // B
    transfer.addTransferWithTimestamps(1500, 20.0, 60.0, 80.0);  // C

    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 70.0);
    EXPECT_EQ(transfer.totalBytes, 4500u);

    double rate;
    EXPECT_TRUE(transfer.getRateFromActiveTime(rate));
    // 4500 bytes / 70 us ≈ 64.286 MB/s
    EXPECT_NEAR(rate, 64.2857, 0.001);
}

TEST_F(AggregatedTransferIntervalTest, AdjacentIntervals)
{
    // Two adjacent intervals (end of one equals start of next):
    // Transfer A: t=0 to t=10
    // Transfer B: t=10 to t=20
    transfer.addTransferWithTimestamps(1000, 10.0, 0.0, 10.0);
    transfer.addTransferWithTimestamps(1000, 10.0, 10.0, 20.0);

    // Should merge to [0, 20], Active time = 20 us
    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 20.0);

    double rate;
    EXPECT_TRUE(transfer.getRateFromActiveTime(rate));
    // 2000 bytes / 20 us = 100 MB/s
    EXPECT_DOUBLE_EQ(rate, 100.0);
}

TEST_F(AggregatedTransferIntervalTest, ManyOverlappingIntervals)
{
    // Many small overlapping transfers:
    // Each transfer is 5us long, starts 2us after the previous
    // Creates a chain of overlapping intervals
    for (int i = 0; i < 10; i++)
    {
        double start = i * 2.0;
        double end   = start + 5.0;
        transfer.addTransferWithTimestamps(100, 5.0, start, end);
    }

    // First interval: [0, 5]
    // Last interval: [18, 23]
    // All overlap, so merged = [0, 23]
    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 23.0);
    EXPECT_EQ(transfer.totalBytes, 1000u);

    double rate;
    EXPECT_TRUE(transfer.getRateFromActiveTime(rate));
    // 1000 bytes / 23 us ≈ 43.478 MB/s
    EXPECT_NEAR(rate, 43.478, 0.01);
}

TEST_F(AggregatedTransferIntervalTest, UnorderedIntervalAddition)
{
    // Add intervals out of order - should still merge correctly
    transfer.addTransferWithTimestamps(1000, 10.0, 50.0, 60.0);  // Third
    transfer.addTransferWithTimestamps(1000, 10.0, 0.0, 10.0);   // First
    transfer.addTransferWithTimestamps(1000, 10.0, 20.0, 35.0);  // Second (overlaps with third)
    transfer.addTransferWithTimestamps(1000, 10.0, 30.0, 55.0);  // Overlaps second and third

    // Timeline:
    // [0, 10]           [20, 35]    [50, 60]
    //                   [30,    55]
    // Merged: [0, 10] and [20, 60]
    // Active time = 10 + 40 = 50 us
    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 50.0);
}

TEST_F(AggregatedTransferIntervalTest, ZeroDurationInterval)
{
    // Zero duration interval (start == end) should not be added
    transfer.addTransferWithTimestamps(1000, 0.0, 10.0, 10.0);

    // No valid interval added
    EXPECT_TRUE(transfer.intervals.empty());
    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 0.0);

    // But bytes and count should still be added
    EXPECT_EQ(transfer.totalBytes, 1000u);
    EXPECT_EQ(transfer.count, 1);
}

TEST_F(AggregatedTransferIntervalTest, NegativeDurationInterval)
{
    // Negative duration interval (start > end) should not be added
    transfer.addTransferWithTimestamps(1000, -5.0, 20.0, 10.0);

    // No valid interval added
    EXPECT_TRUE(transfer.intervals.empty());
    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 0.0);
}

TEST_F(AggregatedTransferIntervalTest, MergeIntervalsFromAnotherTransfer)
{
    // Test merging intervals from another AggregatedTransfer
    transfer.addTransferWithTimestamps(1000, 10.0, 0.0, 10.0);
    transfer.addTransferWithTimestamps(1000, 10.0, 20.0, 30.0);

    AggregatedTransfer other;
    other.addTransferWithTimestamps(500, 5.0, 5.0, 15.0);   // Overlaps with first
    other.addTransferWithTimestamps(500, 5.0, 40.0, 45.0);  // New gap

    transfer.mergeIntervals(other);

    // Intervals: [0, 10], [20, 30], [5, 15], [40, 45]
    // After sorting: [0, 10], [5, 15], [20, 30], [40, 45]
    // Merged: [0, 15], [20, 30], [40, 45]
    // Active time = 15 + 10 + 5 = 30 us
    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 30.0);

    // Note: totalBytes is NOT merged by mergeIntervals - only intervals
    // This matches the intended design where bytes/time are merged separately
    EXPECT_EQ(transfer.totalBytes, 2000u);  // Only original transfer's bytes
}

TEST_F(AggregatedTransferIntervalTest, RateCalculationWithZeroBytes)
{
    // Add interval but with zero bytes
    transfer.intervals.push_back({0.0, 10.0});

    double rate;
    EXPECT_FALSE(transfer.getRateFromActiveTime(rate));
    EXPECT_DOUBLE_EQ(rate, 0.0);
}

TEST_F(AggregatedTransferIntervalTest, LatencyStillWorksWithIntervals)
{
    // Verify that latency calculation (linear regression) still works
    // when intervals are also being tracked
    transfer.addTransferWithTimestamps(0, 10.0, 0.0, 10.0);
    transfer.addTransferWithTimestamps(1000, 20.0, 10.0, 30.0);
    transfer.addTransferWithTimestamps(2000, 30.0, 30.0, 60.0);
    transfer.addTransferWithTimestamps(5000, 60.0, 60.0, 120.0);

    // Latency from linear regression
    double latency;
    bool hasLatency = transfer.getLatencyFromLinearRegression(latency);
    EXPECT_TRUE(hasLatency);
    EXPECT_NEAR(latency, 10.0, 0.5);

    // Rate from active time
    // Intervals: [0,10], [10,30], [30,60], [60,120]
    // All adjacent, merged: [0, 120]
    // Total bytes: 0 + 1000 + 2000 + 5000 = 8000
    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 120.0);

    double rate;
    EXPECT_TRUE(transfer.getRateFromActiveTime(rate));
    // 8000 bytes / 120 us ≈ 66.67 MB/s
    EXPECT_NEAR(rate, 66.67, 0.1);
}

// =============================================================================
// Tests for CUDA Graph classification in scale-up aggregation
//
// Policy:
// - A communicator is classified once as CUDA-Graph-driven or not.
// - For CUDA Graph communicators, scale-up aggregation exports count/time only and
//   suppresses bandwidth/latency (no inferred transfer cache, no rank/channel intervals).
//
// Detection heuristic: two or more distinct Coll/P2P handles share the same
// KernelCh pTimerStart value within a window.
// =============================================================================

TEST_F(WindowAggregatorTest, ScaleUpCudaGraphCommunicatorCountTimeOnlyAndSuppressesTransfers)
{
    const uint64_t SHARED_PTIMER_START = 5000000;
    const uint64_t PTIMER_STOP_OP1     = 5000000 + 12084;
    const uint64_t PTIMER_STOP_OP2     = 5000000 + 13000;

    std::unique_ptr<CommunicatorState> commState(new CommunicatorState());
    commState->nranks     = 8;
    commState->rank       = 0;
    commState->comm_hash  = 0;
    commState->local_rank = 0;
    commState->hostname   = "test";

    const size_t BYTES = 1024 * 1024;

    auto coll1 =
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, BYTES, 100.0, 200.0, commState.get());
    aggregator->addEvent(coll1);
    auto kch1 = createKernelChEvent(0, SHARED_PTIMER_START, PTIMER_STOP_OP1, 200.0, 220.0, &coll1);
    aggregator->addEvent(kch1);

    auto coll2 =
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, BYTES, 300.0, 400.0, commState.get());
    aggregator->addEvent(coll2);
    auto kch2 = createKernelChEvent(0, SHARED_PTIMER_START, PTIMER_STOP_OP2, 400.0, 420.0, &coll2);
    aggregator->addEvent(kch2);

    aggregator->finalize();

    // Communicator classified as CUDA Graph driven.
    EXPECT_STREQ(commState->getScaleUpExecModeString(), "cuda_graph");

    // Collective count/time still exported (CPU KernelCh timing).
    const auto& collectives = aggregator->getCollectives();
    auto it                 = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(it, collectives.end());
    EXPECT_EQ(it->second.count, 2);
    EXPECT_NEAR(it->second.totalTimeUs, 240.0, 0.1);  // 2 × (220-100)

    // In CUDA Graph mode we still infer transfer count/size (volume), but we do not
    // emit timing-derived data (intervals) required for bandwidth/latency.
    EXPECT_GT(it->second.getTotalTransferCount(), 0);

    const auto& rankTransfers    = aggregator->getRankTransfers();
    const auto& channelTransfers = aggregator->getChannelTransfers();
    EXPECT_FALSE(rankTransfers.empty());
    EXPECT_FALSE(channelTransfers.empty());

    for (const auto& rt : rankTransfers)
    {
        EXPECT_GT(rt.second.totalBytes, 0u);
        EXPECT_GT(rt.second.count, 0);
        EXPECT_TRUE(rt.second.intervals.empty());
        double rate;
        EXPECT_FALSE(rt.second.getRateFromActiveTime(rate));
        double lat;
        EXPECT_FALSE(rt.second.getLatencyFromLinearRegression(lat));
    }

    for (const auto& ct : channelTransfers)
    {
        EXPECT_GT(ct.second.totalBytes, 0u);
        EXPECT_GT(ct.second.count, 0);
        EXPECT_TRUE(ct.second.intervals.empty());
        double rate;
        EXPECT_FALSE(ct.second.getRateFromActiveTime(rate));
        double lat;
        EXPECT_FALSE(ct.second.getLatencyFromLinearRegression(lat));
    }
}

TEST_F(WindowAggregatorTest, CudaGraphCommunicatorStillKeepsProxyTransferTiming)
{
    const uint64_t SHARED_PTIMER_START = 7000000;

    std::unique_ptr<CommunicatorState> commState(new CommunicatorState());
    commState->nranks     = 8;
    commState->rank       = 0;
    commState->comm_hash  = 0;
    commState->local_rank = 0;
    commState->hostname   = "test";
    commState->comm_type  = CommunicatorState::CommType::COLLECTIVE;

    const size_t BYTES = 1024 * 1024;

    // First, create CUDA-graph evidence from kernel-only collectives.
    auto coll1 =
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, BYTES, 100.0, 200.0, commState.get());
    aggregator->addEvent(coll1);
    auto kch1 = createKernelChEvent(0, SHARED_PTIMER_START, SHARED_PTIMER_START + 1000, 200.0, 220.0, &coll1);
    aggregator->addEvent(kch1);

    auto coll2 =
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, BYTES, 300.0, 400.0, commState.get());
    aggregator->addEvent(coll2);
    auto kch2 = createKernelChEvent(0, SHARED_PTIMER_START, SHARED_PTIMER_START + 2000, 400.0, 420.0, &coll2);
    aggregator->addEvent(kch2);

    // Then add a scale-out style collective that has real ProxyOp/ProxyStep timing.
    auto coll3 =
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, BYTES, 500.0, 550.0, commState.get());
    aggregator->addEvent(coll3);
    auto proxyOp      = createProxyOpEvent(1, 0, 262144, 505.0, 560.0, &coll3);
    proxyOp.commState = commState.get();
    aggregator->addEvent(proxyOp);
    auto proxyStep      = createProxyStepEvent(0, 262144, 505.0, 520.0, 560.0, &proxyOp);
    proxyStep.commState = commState.get();
    aggregator->addEvent(proxyStep);

    aggregator->finalize();

    EXPECT_STREQ(commState->getScaleUpExecModeString(), "cuda_graph");

    const auto& collectives = aggregator->getCollectives();
    auto collIt             = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(collIt, collectives.end());
    EXPECT_EQ(collIt->second.count, 3);
    EXPECT_GT(collIt->second.getTotalTransferCount(), 0);
    EXPECT_GT(collIt->second.cachedTotalTransferTimeUs, 0.0);

    const auto& rankTransfers = aggregator->getRankTransfers();
    ASSERT_FALSE(rankTransfers.empty());
    auto rankIt = rankTransfers.find("Comm0_Rank0_ToPeer1");
    ASSERT_NE(rankIt, rankTransfers.end());
    EXPECT_GT(rankIt->second.totalBytes, 0u);
    EXPECT_GT(rankIt->second.totalTimeUs, 0.0);
    EXPECT_FALSE(rankIt->second.intervals.empty());
    double rate = 0.0;
    EXPECT_TRUE(rankIt->second.getRateFromActiveTime(rate));

    const auto& channelTransfers = aggregator->getChannelTransfers();
    ASSERT_FALSE(channelTransfers.empty());
    auto channelIt = channelTransfers.find("Comm0_Rank0_ToPeer1_Chnl0");
    ASSERT_NE(channelIt, channelTransfers.end());
    EXPECT_GT(channelIt->second.totalBytes, 0u);
    EXPECT_GT(channelIt->second.totalTimeUs, 0.0);
    EXPECT_FALSE(channelIt->second.intervals.empty());
}

// ---------------------------------------------------------------------------
// AlltoAll collective synthesis (Phase 4)
// Verify that P2P Send events that share a P2pApi parent are aggregated
// into a single Collective entry by finalize().
// ---------------------------------------------------------------------------
TEST_F(WindowAggregatorTest, AlltoAllCollectiveSynthesis_BasicGrouping)
{
    // --- Communicator state for a 4-rank COLLECTIVE communicator ---
    auto commState       = std::make_unique<CommunicatorState>();
    commState->comm_hash = 0xABCD;
    commState->nranks    = 4;
    commState->rank      = 0;
    commState->comm_type = CommunicatorState::CommType::COLLECTIVE;

    // --- P2pApi marker event (AlltoAll anchor) ---
    otelEventHandle_t p2pApiEvent = {};
    p2pApiEvent.type              = ncclProfileP2pApi;
    p2pApiEvent.p2pApi.func       = "AlltoAll";
    p2pApiEvent.startTs           = 0.0;
    p2pApiEvent.endTs             = 1.0;
    p2pApiEvent.commState         = commState.get();

    // --- P2P Send to peer=1 (non-self, parentObj = &p2pApiEvent) ---
    otelEventHandle_t p2pSend1 = {};
    p2pSend1.type              = ncclProfileP2p;
    p2pSend1.p2p.func          = "Send";
    p2pSend1.p2p.peer          = 1;
    p2pSend1.p2p.nChannels     = 1;
    p2pSend1.p2p.bytes         = 2048;
    p2pSend1.startTs           = 1.0;
    p2pSend1.endTs             = 50.0;
    p2pSend1.parentObj         = &p2pApiEvent;
    p2pSend1.commState         = commState.get();
    p2pSend1.rank              = 0;

    // --- P2P Send to peer=2 (non-self) ---
    otelEventHandle_t p2pSend2 = {};
    p2pSend2.type              = ncclProfileP2p;
    p2pSend2.p2p.func          = "Send";
    p2pSend2.p2p.peer          = 2;
    p2pSend2.p2p.nChannels     = 1;
    p2pSend2.p2p.bytes         = 2048;
    p2pSend2.startTs           = 1.5;
    p2pSend2.endTs             = 55.0;
    p2pSend2.parentObj         = &p2pApiEvent;
    p2pSend2.commState         = commState.get();
    p2pSend2.rank              = 0;

    // --- ProxyOp for p2pSend1 ---
    otelEventHandle_t proxyOp1 = {};
    proxyOp1.type              = ncclProfileProxyOp;
    proxyOp1.proxyOp.peer      = 1;
    proxyOp1.proxyOp.channelId = 0;
    proxyOp1.proxyOp.chunkSize = 2048;
    proxyOp1.startTs           = 2.0;
    proxyOp1.endTs             = 80.0;  // last proxy op end for send1
    proxyOp1.parentObj         = &p2pSend1;
    proxyOp1.commState         = commState.get();
    proxyOp1.rank              = 0;

    // --- ProxyOp for p2pSend2 ---
    otelEventHandle_t proxyOp2 = {};
    proxyOp2.type              = ncclProfileProxyOp;
    proxyOp2.proxyOp.peer      = 2;
    proxyOp2.proxyOp.channelId = 0;
    proxyOp2.proxyOp.chunkSize = 2048;
    proxyOp2.startTs           = 3.0;
    proxyOp2.endTs             = 100.0;  // last proxy op end for send2 — controls AlltoAll duration
    proxyOp2.parentObj         = &p2pSend2;
    proxyOp2.commState         = commState.get();
    proxyOp2.rank              = 0;

    // Feed events in buffer order: P2pApi → P2P sends → ProxyOps
    aggregator->addEvent(p2pApiEvent);
    aggregator->addEvent(p2pSend1);
    aggregator->addEvent(p2pSend2);
    aggregator->addEvent(proxyOp1);
    aggregator->addEvent(proxyOp2);
    aggregator->finalize();

    // Expect one AlltoAll collective entry under the synthesised key
    const auto& collectives = aggregator->getCollectives();
    std::string expectedKey = "Comm43981_AlltoAll_4Ranks";  // comm_hash=0xABCD=43981, nranks=4
    auto it                 = collectives.find(expectedKey);
    ASSERT_NE(it, collectives.end()) << "AlltoAll collective key not found: " << expectedKey;

    // Total bytes = 2048 (send1) + 2048 (send2) = 4096
    EXPECT_EQ(it->second.totalBytes, 4096u);
    // Duration = max(proxyOp2.endTs=100) - min(p2pSend1.startTs=1.0) = 99 us
    EXPECT_NEAR(it->second.getAverageTime(), 99.0, 1e-3);
    EXPECT_EQ(it->second.count, 1u);
}

TEST_F(WindowAggregatorTest, AlltoAllCollectiveSynthesis_NoPeerEventsYieldsNothing)
{
    // A P2pApi event with no P2P children should not produce any collective entry.
    otelEventHandle_t p2pApiEvent = {};
    p2pApiEvent.type              = ncclProfileP2pApi;
    p2pApiEvent.p2pApi.func       = "AlltoAll";
    p2pApiEvent.startTs           = 0.0;
    p2pApiEvent.endTs             = 1.0;

    aggregator->addEvent(p2pApiEvent);
    aggregator->finalize();

    EXPECT_TRUE(aggregator->getCollectives().empty());
}

TEST_F(WindowAggregatorTest, AlltoAllCollectiveSynthesis_NullFuncIsIgnored)
{
    // A P2pApi event with a NULL func should not be tracked (no collective key can be built).
    otelEventHandle_t p2pApiEvent = {};
    p2pApiEvent.type              = ncclProfileP2pApi;
    p2pApiEvent.p2pApi.func       = nullptr;
    p2pApiEvent.startTs           = 0.0;
    p2pApiEvent.endTs             = 1.0;

    otelEventHandle_t p2pSend = {};
    p2pSend.type              = ncclProfileP2p;
    p2pSend.p2p.func          = "Send";
    p2pSend.p2p.peer          = 1;
    p2pSend.p2p.nChannels     = 1;
    p2pSend.p2p.bytes         = 1024;
    p2pSend.startTs           = 1.0;
    p2pSend.endTs             = 10.0;
    p2pSend.parentObj         = &p2pApiEvent;

    aggregator->addEvent(p2pApiEvent);
    aggregator->addEvent(p2pSend);
    aggregator->finalize();

    EXPECT_TRUE(aggregator->getCollectives().empty());
}

// Explicit ncclSend/ncclRecv on a pipeline-parallel communicator (nranks==2, CommType::P2P)
// also produces P2pApi events (the same NCCL code path as AlltoAll).  These must NOT be
// synthesized as Collective entries — they belong to the P2P section of the dashboard.
TEST_F(WindowAggregatorTest, AlltoAllCollectiveSynthesis_PipelineParallelStaysInP2P)
{
    auto commState       = std::make_unique<CommunicatorState>();
    commState->comm_hash = 0x1234;
    commState->nranks    = 2;
    commState->rank      = 0;
    commState->comm_type = CommunicatorState::CommType::P2P;  // pipeline-parallel

    otelEventHandle_t p2pApiEvent = {};
    p2pApiEvent.type              = ncclProfileP2pApi;
    p2pApiEvent.p2pApi.func       = "Send";
    p2pApiEvent.startTs           = 0.0;
    p2pApiEvent.endTs             = 1.0;
    p2pApiEvent.commState         = commState.get();

    otelEventHandle_t p2pSend = {};
    p2pSend.type              = ncclProfileP2p;
    p2pSend.p2p.func          = "Send";
    p2pSend.p2p.peer          = 1;
    p2pSend.p2p.nChannels     = 1;
    p2pSend.p2p.bytes         = 65536;
    p2pSend.startTs           = 1.0;
    p2pSend.endTs             = 20.0;
    p2pSend.parentObj         = &p2pApiEvent;
    p2pSend.commState         = commState.get();
    p2pSend.rank              = 0;

    aggregator->addEvent(p2pApiEvent);
    aggregator->addEvent(p2pSend);
    aggregator->finalize();

    // Must NOT appear in Collective section.
    EXPECT_TRUE(aggregator->getCollectives().empty());

    // Must still appear in P2P section (the normal P2P tracking is unaffected).
    EXPECT_FALSE(aggregator->getP2Ps().empty());
}
