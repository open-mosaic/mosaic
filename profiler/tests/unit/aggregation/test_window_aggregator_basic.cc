// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "../../support/window_aggregator_fixture.h"

TEST_F(WindowAggregatorTest, InitialState)
{
    EXPECT_TRUE(aggregator->getCollectives().empty());
    EXPECT_TRUE(aggregator->getP2Ps().empty());
    EXPECT_TRUE(aggregator->getRankTransfers().empty());
    EXPECT_TRUE(aggregator->getChannelTransfers().empty());
}

TEST_F(WindowAggregatorTest, AggregatesSingleCollective)
{
    auto coll = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 0.0, 10.0);
    aggregator->addEvent(coll);

    auto proxyOp1   = createProxyOpEvent(1, 0, 262144, 10.0, 200.0, &coll);
    auto proxyStep1 = createProxyStepEvent(0, 512, 10.0, 100.0, 200.0, &proxyOp1);
    aggregator->addEvent(proxyStep1);
    aggregator->addEvent(proxyOp1);
    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    EXPECT_EQ(collectives.size(), 1u);

    auto it = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(it, collectives.end());
    EXPECT_EQ(it->second.count, 1);
    EXPECT_EQ(it->second.totalBytes, 1024u);
    EXPECT_DOUBLE_EQ(it->second.totalTimeUs, 200.0);
}

TEST_F(WindowAggregatorTest, AggregatesMultipleCollectivesWithSameKey)
{
    auto coll1 = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 0.0, 10.0);
    aggregator->addEvent(coll1);
    auto proxyOp1   = createProxyOpEvent(1, 0, 262144, 10.0, 100.0, &coll1);
    auto proxyStep1 = createProxyStepEvent(0, 512, 10.0, 50.0, 100.0, &proxyOp1);
    aggregator->addEvent(proxyStep1);
    aggregator->addEvent(proxyOp1);

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
    EXPECT_DOUBLE_EQ(it->second.totalTimeUs, 200.0);
}

TEST_F(WindowAggregatorTest, AggregatesMultipleCollectivesWithDifferentKeys)
{
    auto coll1 = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 0.0, 10.0);
    aggregator->addEvent(coll1);
    auto proxyOp1   = createProxyOpEvent(1, 0, 262144, 10.0, 100.0, &coll1);
    auto proxyStep1 = createProxyStepEvent(0, 512, 10.0, 50.0, 100.0, &proxyOp1);
    aggregator->addEvent(proxyStep1);
    aggregator->addEvent(proxyOp1);

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
    auto p2p = createP2PEvent("Send", 3, 2, 512, 0.0, 5.0);
    aggregator->addEvent(p2p);

    auto proxyOp   = createProxyOpEvent(3, 0, 262144, 5.0, 50.0, &p2p);
    auto proxyStep = createProxyStepEvent(0, 256, 5.0, 25.0, 50.0, &proxyOp);
    aggregator->addEvent(proxyStep);
    aggregator->addEvent(proxyOp);
    aggregator->finalize();

    const auto& p2ps = aggregator->getP2Ps();
    EXPECT_EQ(p2ps.size(), 1u);

    auto it = p2ps.find("Comm0_(unknown)_Send_Pipeline0ToPipeline3_2Chnl");
    ASSERT_NE(it, p2ps.end());
    EXPECT_EQ(it->second.count, 1);
    EXPECT_EQ(it->second.totalBytes, 512u);
    EXPECT_DOUBLE_EQ(it->second.totalTimeUs, 50.0);
}

TEST_F(WindowAggregatorTest, AggregatesSingleProxyOp)
{
    auto proxyOp   = createProxyOpEvent(2, 1, 256, 0.0, 3.0);
    auto proxyStep = createProxyStepEvent(0, 128, 0.0, 1.0, 3.0, &proxyOp);
    aggregator->addEvent(proxyStep);
    aggregator->addEvent(proxyOp);
    aggregator->finalize();

    const auto& rankTransfers = aggregator->getRankTransfers();
    EXPECT_EQ(rankTransfers.size(), 1u);

    auto it = rankTransfers.find("Comm0_Rank0_ToPeer2");
    ASSERT_NE(it, rankTransfers.end());
    EXPECT_EQ(it->second.count, 1);
    EXPECT_EQ(it->second.totalBytes, 128u);
}

TEST_F(WindowAggregatorTest, HandlesNegativeDuration)
{
    auto coll = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 10.0, 15.0);
    aggregator->addEvent(coll);

    auto proxyOp   = createProxyOpEvent(1, 0, 128, 0.0, 5.0, &coll);
    auto proxyStep = createProxyStepEvent(0, 64, 0.0, 2.0, 5.0, &proxyOp);
    aggregator->addEvent(proxyStep);
    aggregator->addEvent(proxyOp);
    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    auto it                 = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(it, collectives.end());
    EXPECT_DOUBLE_EQ(it->second.totalTimeUs, 0.0);
}

TEST_F(WindowAggregatorTest, HandlesZeroDuration)
{
    auto coll = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 10.0, 15.0);
    aggregator->addEvent(coll);

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
    auto proxyOp   = createProxyOpEvent(1, 0, 128, 0.0, 1.0, nullptr);
    auto proxyStep = createProxyStepEvent(0, 64, 0.0, 0.5, 1.0, &proxyOp);
    aggregator->addEvent(proxyStep);
    aggregator->addEvent(proxyOp);
    aggregator->finalize();

    const auto& rankTransfers = aggregator->getRankTransfers();
    EXPECT_EQ(rankTransfers.size(), 1u);
    auto it = rankTransfers.find("Comm0_Rank0_ToPeer1");
    ASSERT_NE(it, rankTransfers.end());
    EXPECT_EQ(it->second.count, 1);
}

TEST_F(WindowAggregatorTest, ComplexScenarioWithMultipleEventTypes)
{
    auto coll1 = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 2048, 0.0, 20.0);
    aggregator->addEvent(coll1);

    auto proxyOp1   = createProxyOpEvent(1, 0, 256, 1.0, 2.0, &coll1);
    auto proxyStep1 = createProxyStepEvent(0, 128, 1.0, 1.5, 2.0, &proxyOp1);
    aggregator->addEvent(proxyStep1);
    aggregator->addEvent(proxyOp1);

    auto proxyOp2   = createProxyOpEvent(1, 1, 256, 2.0, 3.0, &coll1);
    auto proxyStep2 = createProxyStepEvent(0, 128, 2.0, 2.5, 3.0, &proxyOp2);
    aggregator->addEvent(proxyStep2);
    aggregator->addEvent(proxyOp2);

    auto p2p1 = createP2PEvent("Send", 2, 1, 1024, 20.0, 30.0);
    aggregator->addEvent(p2p1);

    auto proxyOp3   = createProxyOpEvent(2, 0, 128, 21.0, 22.0, &p2p1);
    auto proxyStep3 = createProxyStepEvent(0, 64, 21.0, 21.5, 22.0, &proxyOp3);
    aggregator->addEvent(proxyStep3);
    aggregator->addEvent(proxyOp3);

    auto proxyOp4   = createProxyOpEvent(3, 0, 512, 30.0, 35.0, nullptr);
    auto proxyStep4 = createProxyStepEvent(0, 256, 30.0, 32.0, 35.0, &proxyOp4);
    aggregator->addEvent(proxyStep4);
    aggregator->addEvent(proxyOp4);

    aggregator->finalize();

    EXPECT_EQ(aggregator->getCollectives().size(), 1u);
    EXPECT_EQ(aggregator->getP2Ps().size(), 1u);
    EXPECT_EQ(aggregator->getRankTransfers().size(), 3u);

    auto collIt = aggregator->getCollectives().find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(collIt, aggregator->getCollectives().end());
    EXPECT_EQ(collIt->second.cachedTotalTransferCount, 2);

    auto p2pIt = aggregator->getP2Ps().find("Comm0_(unknown)_Send_Pipeline0ToPipeline2_1Chnl");
    ASSERT_NE(p2pIt, aggregator->getP2Ps().end());
    EXPECT_EQ(p2pIt->second.cachedTotalTransferCount, 1);
}

TEST_F(WindowAggregatorTest, StressTestManyEvents)
{
    const int NUM_COLLS              = 1000;
    const int NUM_P2PS               = 500;
    const int NUM_STANDALONE_PROXIES = 500;

    for (int i = 0; i < NUM_COLLS; i++)
    {
        auto coll = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, i * 10.0, i * 10.0 + 5.0);
        aggregator->addEvent(coll);

        for (int ch = 0; ch < 2; ch++)
        {
            auto proxyOp = createProxyOpEvent(i % 4, ch, 128, i * 10.0 + ch, i * 10.0 + ch + 1.0, &coll);
            auto proxyStep =
                createProxyStepEvent(0, 64, i * 10.0 + ch, i * 10.0 + ch + 0.5, i * 10.0 + ch + 1.0, &proxyOp);
            aggregator->addEvent(proxyStep);
            aggregator->addEvent(proxyOp);
        }
    }

    for (int i = 0; i < NUM_P2PS; i++)
    {
        auto p2p = createP2PEvent("Send", i % 4, 1, 512, i * 5.0, i * 5.0 + 3.0);
        aggregator->addEvent(p2p);

        auto proxyOp   = createProxyOpEvent(i % 4, 0, 64, i * 5.0, i * 5.0 + 2.0, &p2p);
        auto proxyStep = createProxyStepEvent(0, 32, i * 5.0, i * 5.0 + 1.0, i * 5.0 + 2.0, &proxyOp);
        aggregator->addEvent(proxyStep);
        aggregator->addEvent(proxyOp);
    }

    for (int i = 0; i < NUM_STANDALONE_PROXIES; i++)
    {
        auto proxyOp   = createProxyOpEvent(i % 4, i % 8, 128, i * 2.0, i * 2.0 + 1.0, nullptr);
        auto proxyStep = createProxyStepEvent(0, 64, i * 2.0, i * 2.0 + 0.5, i * 2.0 + 1.0, &proxyOp);
        aggregator->addEvent(proxyStep);
        aggregator->addEvent(proxyOp);
    }

    aggregator->finalize();

    EXPECT_GT(aggregator->getCollectives().size(), 0u);
    EXPECT_GT(aggregator->getP2Ps().size(), 0u);
    EXPECT_GT(aggregator->getRankTransfers().size(), 0u);
}

TEST_F(WindowAggregatorTest, StressTestDeepProxyOpNesting)
{
    auto coll = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 0.0, 100.0);
    aggregator->addEvent(coll);

    for (int i = 0; i < 1000; i++)
    {
        auto proxy     = createProxyOpEvent(i % 4, i % 8, 128 + i, i * 0.1, i * 0.1 + 0.05, &coll);
        auto proxyStep = createProxyStepEvent(0, 64 + i, i * 0.1, i * 0.1 + 0.02, i * 0.1 + 0.05, &proxy);
        aggregator->addEvent(proxyStep);
        aggregator->addEvent(proxy);
    }

    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    auto it                 = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(it, collectives.end());
    EXPECT_EQ(it->second.cachedTotalTransferCount, 1000);
    EXPECT_GT(it->second.cachedTotalTransferBytes, 0u);
    EXPECT_GT(it->second.cachedTotalTransferTimeUs, 0.0);
    EXPECT_DOUBLE_EQ(it->second.getAverageTransferCount(), 1000.0);
}

TEST_F(WindowAggregatorTest, OrphanedProxyOps)
{
    auto proxy1     = createProxyOpEvent(1, 0, 128, 1.0, 2.0, (void*)0x1234);
    auto proxyStep1 = createProxyStepEvent(0, 64, 1.0, 1.5, 2.0, &proxy1);
    aggregator->addEvent(proxyStep1);
    aggregator->addEvent(proxy1);

    auto proxy2     = createProxyOpEvent(1, 1, 256, 2.0, 4.0, (void*)0x1234);
    auto proxyStep2 = createProxyStepEvent(0, 128, 2.0, 3.0, 4.0, &proxy2);
    aggregator->addEvent(proxyStep2);
    aggregator->addEvent(proxy2);

    auto coll = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 0.0, 10.0);
    aggregator->addEvent(coll);
    aggregator->finalize();

    EXPECT_GT(aggregator->getRankTransfers().size(), 0u);

    const auto& collectives = aggregator->getCollectives();
    auto it                 = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    EXPECT_NE(it, collectives.end());
    if (it != collectives.end())
    {
        EXPECT_EQ(it->second.count, 1u);
    }
}