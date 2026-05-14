// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "../../support/window_aggregator_fixture.h"

TEST_F(WindowAggregatorTest, ScaleUpCudaGraphCommunicatorCountTimeOnlyAndSuppressesTransfers)
{
    const uint64_t sharedPTimerStart = 5000000;
    const uint64_t pTimerStopOp1     = 5000000 + 12084;
    const uint64_t pTimerStopOp2     = 5000000 + 13000;

    std::unique_ptr<CommunicatorState> commState(new CommunicatorState());
    commState->nranks     = 8;
    commState->rank       = 0;
    commState->comm_hash  = 0;
    commState->local_rank = 0;
    commState->hostname   = "test";

    const size_t bytes = 1024 * 1024;

    auto coll1 =
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, bytes, 100.0, 200.0, commState.get());
    aggregator->addEvent(coll1);
    auto kch1 = createKernelChEvent(0, sharedPTimerStart, pTimerStopOp1, 200.0, 220.0, &coll1);
    aggregator->addEvent(kch1);

    auto coll2 =
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, bytes, 300.0, 400.0, commState.get());
    aggregator->addEvent(coll2);
    auto kch2 = createKernelChEvent(0, sharedPTimerStart, pTimerStopOp2, 400.0, 420.0, &coll2);
    aggregator->addEvent(kch2);

    aggregator->finalize();

    EXPECT_STREQ(commState->getScaleUpExecModeString(), "cuda_graph");

    const auto& collectives = aggregator->getCollectives();
    auto it                 = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(it, collectives.end());
    EXPECT_EQ(it->second.count, 2);
    EXPECT_NEAR(it->second.totalTimeUs, 240.0, 0.1);
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
    const uint64_t sharedPTimerStart = 7000000;

    std::unique_ptr<CommunicatorState> commState(new CommunicatorState());
    commState->nranks     = 8;
    commState->rank       = 0;
    commState->comm_hash  = 0;
    commState->local_rank = 0;
    commState->hostname   = "test";
    commState->comm_type  = CommunicatorState::CommType::COLLECTIVE;

    const size_t bytes = 1024 * 1024;

    auto coll1 =
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, bytes, 100.0, 200.0, commState.get());
    aggregator->addEvent(coll1);
    auto kch1 = createKernelChEvent(0, sharedPTimerStart, sharedPTimerStart + 1000, 200.0, 220.0, &coll1);
    aggregator->addEvent(kch1);

    auto coll2 =
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, bytes, 300.0, 400.0, commState.get());
    aggregator->addEvent(coll2);
    auto kch2 = createKernelChEvent(0, sharedPTimerStart, sharedPTimerStart + 2000, 400.0, 420.0, &coll2);
    aggregator->addEvent(kch2);

    auto coll3 =
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, bytes, 500.0, 550.0, commState.get());
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

TEST_F(WindowAggregatorTest, TinyScaleUpCollectivePreservesExactInferredByteTotals)
{
    std::unique_ptr<CommunicatorState> commState(new CommunicatorState());
    commState->nranks     = 8;
    commState->rank       = 0;
    commState->comm_hash  = 0;
    commState->local_rank = 0;
    commState->hostname   = "test";
    commState->comm_type  = CommunicatorState::CommType::COLLECTIVE;
    commState->scaleUpExecMode.store(static_cast<uint8_t>(CommunicatorState::ScaleUpExecMode::NON_CUDA_GRAPH),
                                     std::memory_order_release);

    auto coll = createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, 4, 100.0, 200.0, commState.get());
    aggregator->addEvent(coll);

    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    auto collIt             = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(collIt, collectives.end());
    EXPECT_EQ(28, collIt->second.getTotalTransferCount());
    EXPECT_EQ(7u, collIt->second.cachedTotalTransferBytes);
    EXPECT_DOUBLE_EQ(7.0 / 28.0, collIt->second.getAverageTransferSize());

    const auto& rankTransfers = aggregator->getRankTransfers();
    auto rankIt               = rankTransfers.find("Comm0_Rank0_ToPeer1");
    ASSERT_NE(rankIt, rankTransfers.end());
    EXPECT_EQ(28, rankIt->second.count);
    EXPECT_EQ(7u, rankIt->second.totalBytes);

    const auto& channelTransfers = aggregator->getChannelTransfers();
    size_t totalChannelBytes     = 0;
    int totalChannelTransfers    = 0;
    for (const auto& channelPair : channelTransfers)
    {
        totalChannelBytes += channelPair.second.totalBytes;
        totalChannelTransfers += channelPair.second.count;
    }
    EXPECT_EQ(7u, totalChannelBytes);
    EXPECT_EQ(28, totalChannelTransfers);
}

TEST_F(WindowAggregatorTest, MultiChannelScaleUpUsesChannelLocalTransferTime)
{
    std::unique_ptr<CommunicatorState> commState(new CommunicatorState());
    commState->nranks     = 4;
    commState->rank       = 0;
    commState->comm_hash  = 0;
    commState->local_rank = 0;
    commState->hostname   = "test";
    commState->comm_type  = CommunicatorState::CommType::COLLECTIVE;
    commState->scaleUpExecMode.store(static_cast<uint8_t>(CommunicatorState::ScaleUpExecMode::NON_CUDA_GRAPH),
                                     std::memory_order_release);

    auto coll =
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, 4096, 100.0, 160.0, commState.get());
    aggregator->addEvent(coll);
    aggregator->addEvent(createKernelChEvent(0, 1111, 2222, 100.0, 160.0, &coll));
    aggregator->addEvent(createKernelChEvent(1, 1111, 2222, 100.0, 160.0, &coll));

    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    auto collIt             = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(collIt, collectives.end());
    EXPECT_EQ(12, collIt->second.getTotalTransferCount());
    EXPECT_NEAR(10.0, collIt->second.getAverageTransferTime(), 1e-9);

    const auto& channelTransfers = aggregator->getChannelTransfers();
    ASSERT_EQ(2u, channelTransfers.size());
    for (const auto& channelPair : channelTransfers)
    {
        EXPECT_EQ(6, channelPair.second.count);
        EXPECT_NEAR(10.0, channelPair.second.getAverageTime(), 1e-9);
    }
}