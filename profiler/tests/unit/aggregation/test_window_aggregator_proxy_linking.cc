// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "../../support/window_aggregator_fixture.h"

TEST_F(WindowAggregatorTest, TwoRankCollectiveTransfersUseCollectiveKeys)
{
    auto commState       = std::make_unique<CommunicatorState>();
    commState->comm_hash = 0x2222;
    commState->nranks    = 2;
    commState->rank      = 0;
    commState->hostname  = "romeo";
    commState->comm_type = CommunicatorState::CommType::P2P;

    auto collEvent =
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 1, 4096, 0.0, 10.0, commState.get());
    aggregator->addEvent(collEvent);

    auto proxyOp        = createProxyOpEvent(1, 0, 1024, 1.0, 5.0, &collEvent);
    proxyOp.commState   = commState.get();
    auto proxyStep      = createProxyStepEvent(0, 1024, 1.0, 2.0, 5.0, &proxyOp);
    proxyStep.commState = commState.get();

    aggregator->addEvent(proxyStep);
    aggregator->addEvent(proxyOp);
    aggregator->finalize();

    const auto& rankTransfers = aggregator->getRankTransfers();
    EXPECT_EQ(rankTransfers.size(), 1u);
    EXPECT_NE(rankTransfers.find("Comm8738_Rank0_ToPeer1"), rankTransfers.end());
    EXPECT_EQ(rankTransfers.find("Comm8738_romeo_Pipeline0_ToPipeline1"), rankTransfers.end());

    const auto& channelTransfers = aggregator->getChannelTransfers();
    EXPECT_EQ(channelTransfers.size(), 1u);
    EXPECT_NE(channelTransfers.find("Comm8738_Rank0_ToPeer1_Chnl0"), channelTransfers.end());
    EXPECT_EQ(channelTransfers.find("Comm8738_romeo_Pipeline0_ToPipeline1_Chnl0"), channelTransfers.end());
}

TEST_F(WindowAggregatorTest, LinksProxyOpToCollective)
{
    auto collEvent = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 0.0, 10.0);
    aggregator->addEvent(collEvent);

    auto proxyOp   = createProxyOpEvent(1, 0, 128, 1.0, 2.0, &collEvent);
    auto proxyStep = createProxyStepEvent(0, 64, 1.0, 1.5, 2.0, &proxyOp);
    aggregator->addEvent(proxyStep);
    aggregator->addEvent(proxyOp);
    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    auto it                 = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(it, collectives.end());
    EXPECT_EQ(it->second.cachedTotalTransferCount, 1);
    EXPECT_EQ(it->second.cachedTotalTransferBytes, 64u);
    EXPECT_DOUBLE_EQ(it->second.cachedTotalTransferTimeUs, 0.5);
}

TEST_F(WindowAggregatorTest, LinksMultipleProxyOpsToCollective)
{
    auto collEvent = createCollectiveEvent("AllReduce", "Ring", "Simple", 2, 1024, 0.0, 10.0);
    aggregator->addEvent(collEvent);

    auto proxyOp1   = createProxyOpEvent(1, 0, 128, 1.0, 2.0, &collEvent);
    auto proxyStep1 = createProxyStepEvent(0, 100, 1.0, 1.5, 2.0, &proxyOp1);
    aggregator->addEvent(proxyStep1);
    aggregator->addEvent(proxyOp1);

    auto proxyOp2   = createProxyOpEvent(1, 1, 256, 2.0, 4.0, &collEvent);
    auto proxyStep2 = createProxyStepEvent(0, 200, 2.0, 3.0, 4.0, &proxyOp2);
    aggregator->addEvent(proxyStep2);
    aggregator->addEvent(proxyOp2);

    auto proxyOp3   = createProxyOpEvent(2, 0, 512, 3.0, 6.0, &collEvent);
    auto proxyStep3 = createProxyStepEvent(0, 300, 3.0, 4.5, 6.0, &proxyOp3);
    aggregator->addEvent(proxyStep3);
    aggregator->addEvent(proxyOp3);

    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    auto it                 = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(it, collectives.end());
    EXPECT_EQ(it->second.cachedTotalTransferCount, 3);
    EXPECT_EQ(it->second.cachedTotalTransferBytes, 600u);
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
    auto it          = p2ps.find("Comm0_(unknown)_Send_Pipeline0ToPipeline3_2Chnl");
    ASSERT_NE(it, p2ps.end());
    EXPECT_EQ(it->second.cachedTotalTransferCount, 1);
    EXPECT_EQ(it->second.cachedTotalTransferBytes, 32u);
}