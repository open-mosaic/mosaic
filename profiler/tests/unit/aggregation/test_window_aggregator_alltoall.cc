// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <memory>

#include "../../support/window_aggregator_fixture.h"

TEST_F(WindowAggregatorTest, AlltoAllCollectiveSynthesis_BasicGrouping)
{
    auto commState       = std::make_unique<CommunicatorState>();
    commState->comm_hash = 0xABCD;
    commState->nranks    = 4;
    commState->rank      = 0;
    commState->comm_type = CommunicatorState::CommType::COLLECTIVE;

    auto groupEvent = createGroupEvent(0.5, 10.0, commState.get());
    auto selfApi    = createP2pApiEvent("Send", 0.0, 0.1, commState.get());
    auto peer1Api   = createP2pApiEvent("Send", 0.1, 0.2, commState.get());
    auto peer2Api   = createP2pApiEvent("Send", 0.2, 0.3, commState.get());
    auto peer3Api   = createP2pApiEvent("Send", 0.3, 0.4, commState.get());

    otelEventHandle_t p2pSelf = {};
    p2pSelf.type              = ncclProfileP2p;
    p2pSelf.p2p.func          = "Send";
    p2pSelf.p2p.peer          = 0;
    p2pSelf.p2p.nChannels     = 1;
    p2pSelf.p2p.bytes         = 2048;
    p2pSelf.startTs           = 0.8;
    p2pSelf.endTs             = 20.0;
    p2pSelf.parentObj         = &selfApi;
    p2pSelf.commState         = commState.get();
    p2pSelf.rank              = 0;

    otelEventHandle_t p2pSend1 = {};
    p2pSend1.type              = ncclProfileP2p;
    p2pSend1.p2p.func          = "Send";
    p2pSend1.p2p.peer          = 1;
    p2pSend1.p2p.nChannels     = 1;
    p2pSend1.p2p.bytes         = 2048;
    p2pSend1.startTs           = 1.0;
    p2pSend1.endTs             = 50.0;
    p2pSend1.parentObj         = &peer1Api;
    p2pSend1.commState         = commState.get();
    p2pSend1.rank              = 0;

    otelEventHandle_t p2pSend2 = {};
    p2pSend2.type              = ncclProfileP2p;
    p2pSend2.p2p.func          = "Send";
    p2pSend2.p2p.peer          = 2;
    p2pSend2.p2p.nChannels     = 1;
    p2pSend2.p2p.bytes         = 2048;
    p2pSend2.startTs           = 1.5;
    p2pSend2.endTs             = 55.0;
    p2pSend2.parentObj         = &peer2Api;
    p2pSend2.commState         = commState.get();
    p2pSend2.rank              = 0;

    otelEventHandle_t p2pSend3 = {};
    p2pSend3.type              = ncclProfileP2p;
    p2pSend3.p2p.func          = "Send";
    p2pSend3.p2p.peer          = 3;
    p2pSend3.p2p.nChannels     = 1;
    p2pSend3.p2p.bytes         = 2048;
    p2pSend3.startTs           = 1.7;
    p2pSend3.endTs             = 57.0;
    p2pSend3.parentObj         = &peer3Api;
    p2pSend3.commState         = commState.get();
    p2pSend3.rank              = 0;

    otelEventHandle_t proxyOp1 = {};
    proxyOp1.type              = ncclProfileProxyOp;
    proxyOp1.proxyOp.peer      = 1;
    proxyOp1.proxyOp.channelId = 0;
    proxyOp1.proxyOp.chunkSize = 2048;
    proxyOp1.startTs           = 2.0;
    proxyOp1.endTs             = 80.0;
    proxyOp1.parentObj         = &p2pSend1;
    proxyOp1.commState         = commState.get();
    proxyOp1.rank              = 0;

    otelEventHandle_t proxyStep1     = {};
    proxyStep1.type                  = ncclProfileProxyStep;
    proxyStep1.proxyStep.step        = 0;
    proxyStep1.proxyStep.transSize   = 2048;
    proxyStep1.proxyStep.sendWaitTs  = 20.0;
    proxyStep1.proxyStep.hasSendWait = true;
    proxyStep1.startTs               = 10.0;
    proxyStep1.endTs                 = 80.0;
    proxyStep1.parentObj             = &proxyOp1;
    proxyStep1.commState             = commState.get();
    proxyStep1.rank                  = 0;

    otelEventHandle_t proxyOp2 = {};
    proxyOp2.type              = ncclProfileProxyOp;
    proxyOp2.proxyOp.peer      = 2;
    proxyOp2.proxyOp.channelId = 0;
    proxyOp2.proxyOp.chunkSize = 2048;
    proxyOp2.startTs           = 3.0;
    proxyOp2.endTs             = 100.0;
    proxyOp2.parentObj         = &p2pSend2;
    proxyOp2.commState         = commState.get();
    proxyOp2.rank              = 0;

    otelEventHandle_t proxyStep2     = {};
    proxyStep2.type                  = ncclProfileProxyStep;
    proxyStep2.proxyStep.step        = 0;
    proxyStep2.proxyStep.transSize   = 2048;
    proxyStep2.proxyStep.sendWaitTs  = 25.0;
    proxyStep2.proxyStep.hasSendWait = true;
    proxyStep2.startTs               = 12.0;
    proxyStep2.endTs                 = 100.0;
    proxyStep2.parentObj             = &proxyOp2;
    proxyStep2.commState             = commState.get();
    proxyStep2.rank                  = 0;

    otelEventHandle_t proxyOp3 = {};
    proxyOp3.type              = ncclProfileProxyOp;
    proxyOp3.proxyOp.peer      = 3;
    proxyOp3.proxyOp.channelId = 0;
    proxyOp3.proxyOp.chunkSize = 2048;
    proxyOp3.startTs           = 4.0;
    proxyOp3.endTs             = 90.0;
    proxyOp3.parentObj         = &p2pSend3;
    proxyOp3.commState         = commState.get();
    proxyOp3.rank              = 0;

    otelEventHandle_t proxyStep3     = {};
    proxyStep3.type                  = ncclProfileProxyStep;
    proxyStep3.proxyStep.step        = 0;
    proxyStep3.proxyStep.transSize   = 2048;
    proxyStep3.proxyStep.sendWaitTs  = 30.0;
    proxyStep3.proxyStep.hasSendWait = true;
    proxyStep3.startTs               = 15.0;
    proxyStep3.endTs                 = 90.0;
    proxyStep3.parentObj             = &proxyOp3;
    proxyStep3.commState             = commState.get();
    proxyStep3.rank                  = 0;

    aggregator->addEvent(selfApi);
    aggregator->addEvent(peer1Api);
    aggregator->addEvent(peer2Api);
    aggregator->addEvent(peer3Api);
    aggregator->addEvent(groupEvent);
    aggregator->addEvent(p2pSelf);
    aggregator->addEvent(p2pSend1);
    aggregator->addEvent(p2pSend2);
    aggregator->addEvent(p2pSend3);
    aggregator->addEvent(proxyStep1);
    aggregator->addEvent(proxyStep2);
    aggregator->addEvent(proxyStep3);
    aggregator->addEvent(proxyOp1);
    aggregator->addEvent(proxyOp2);
    aggregator->addEvent(proxyOp3);
    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    std::string expectedKey = "Comm43981_AlltoAll_4Ranks";
    auto it                 = collectives.find(expectedKey);
    ASSERT_NE(it, collectives.end()) << "AlltoAll collective key not found: " << expectedKey;

    EXPECT_EQ(it->second.totalBytes, 8192u);
    EXPECT_NEAR(it->second.getAverageTime(), 99.2, 1e-3);
    EXPECT_EQ(it->second.count, 1u);
    EXPECT_TRUE(aggregator->getP2Ps().empty());

    const auto& rankTransfers = aggregator->getRankTransfers();
    EXPECT_EQ(rankTransfers.size(), 3u);
    for (const auto& rankTransfer : rankTransfers)
    {
        EXPECT_EQ(rankTransfer.first.find("Pipeline"), std::string::npos);
    }
}

TEST_F(WindowAggregatorTest, AlltoAllCollectiveSynthesis_RuntimeShapeWithoutTrackedSelfSendStillStaysCollectiveOnly)
{
    auto commState       = std::make_unique<CommunicatorState>();
    commState->comm_hash = 0xBEEF;
    commState->nranks    = 4;
    commState->rank      = 0;
    commState->comm_type = CommunicatorState::CommType::COLLECTIVE;

    auto groupEvent = createGroupEvent(0.5, 10.0, commState.get());
    auto selfApi    = createP2pApiEvent("Send", 0.0, 0.1, commState.get());
    auto peer1Api   = createP2pApiEvent("Send", 0.1, 0.2, commState.get());
    auto peer2Api   = createP2pApiEvent("Send", 0.2, 0.3, commState.get());
    auto peer3Api   = createP2pApiEvent("Send", 0.3, 0.4, commState.get());

    auto p2pSend1      = createP2PEvent("Send", 1, 1, 2048, 1.0, 50.0);
    p2pSend1.parentObj = &peer1Api;
    p2pSend1.commState = commState.get();
    p2pSend1.rank      = 0;

    auto p2pSend2      = createP2PEvent("Send", 2, 1, 2048, 1.5, 55.0);
    p2pSend2.parentObj = &peer2Api;
    p2pSend2.commState = commState.get();
    p2pSend2.rank      = 0;

    auto p2pSend3      = createP2PEvent("Send", 3, 1, 2048, 1.7, 57.0);
    p2pSend3.parentObj = &peer3Api;
    p2pSend3.commState = commState.get();
    p2pSend3.rank      = 0;

    auto proxyOp1      = createProxyOpEvent(1, 0, 2048, 2.0, 80.0, &p2pSend1);
    proxyOp1.commState = commState.get();
    proxyOp1.rank      = 0;

    auto proxyOp2      = createProxyOpEvent(2, 0, 2048, 3.0, 100.0, &p2pSend2);
    proxyOp2.commState = commState.get();
    proxyOp2.rank      = 0;

    auto proxyOp3      = createProxyOpEvent(3, 0, 2048, 4.0, 90.0, &p2pSend3);
    proxyOp3.commState = commState.get();
    proxyOp3.rank      = 0;

    aggregator->addEvent(selfApi);
    aggregator->addEvent(peer1Api);
    aggregator->addEvent(peer2Api);
    aggregator->addEvent(peer3Api);
    aggregator->addEvent(groupEvent);
    aggregator->addEvent(p2pSend1);
    aggregator->addEvent(p2pSend2);
    aggregator->addEvent(p2pSend3);
    aggregator->addEvent(proxyOp1);
    aggregator->addEvent(proxyOp2);
    aggregator->addEvent(proxyOp3);
    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    auto it                 = collectives.find("Comm48879_AlltoAll_4Ranks");
    ASSERT_NE(it, collectives.end());
    EXPECT_EQ(it->second.totalBytes, 8192u);
    EXPECT_NEAR(it->second.getAverageTime(), 99.0, 1e-3);
    EXPECT_EQ(it->second.count, 1u);
    EXPECT_TRUE(aggregator->getP2Ps().empty());
}

TEST_F(WindowAggregatorTest, AlltoAllCollectiveSynthesis_TwoRankCollectiveStillSynthesizes)
{
    auto commState       = std::make_unique<CommunicatorState>();
    commState->comm_hash = 0x2222;
    commState->nranks    = 2;
    commState->rank      = 0;
    commState->comm_type = CommunicatorState::CommType::P2P;

    auto groupEvent = createGroupEvent(0.5, 10.0, commState.get());
    auto selfApi    = createP2pApiEvent("Send", 0.0, 0.1, commState.get());
    auto peerApi    = createP2pApiEvent("Send", 0.1, 0.2, commState.get());

    otelEventHandle_t p2pSelf = {};
    p2pSelf.type              = ncclProfileP2p;
    p2pSelf.p2p.func          = "Send";
    p2pSelf.p2p.peer          = 0;
    p2pSelf.p2p.nChannels     = 1;
    p2pSelf.p2p.bytes         = 2048;
    p2pSelf.startTs           = 1.0;
    p2pSelf.endTs             = 12.0;
    p2pSelf.parentObj         = &selfApi;
    p2pSelf.commState         = commState.get();
    p2pSelf.rank              = 0;

    otelEventHandle_t p2pSend = {};
    p2pSend.type              = ncclProfileP2p;
    p2pSend.p2p.func          = "Send";
    p2pSend.p2p.peer          = 1;
    p2pSend.p2p.nChannels     = 1;
    p2pSend.p2p.bytes         = 2048;
    p2pSend.startTs           = 1.2;
    p2pSend.endTs             = 15.0;
    p2pSend.parentObj         = &peerApi;
    p2pSend.commState         = commState.get();
    p2pSend.rank              = 0;

    otelEventHandle_t proxyOp = {};
    proxyOp.type              = ncclProfileProxyOp;
    proxyOp.proxyOp.peer      = 1;
    proxyOp.proxyOp.channelId = 0;
    proxyOp.proxyOp.chunkSize = 2048;
    proxyOp.startTs           = 2.0;
    proxyOp.endTs             = 25.0;
    proxyOp.parentObj         = &p2pSend;
    proxyOp.commState         = commState.get();
    proxyOp.rank              = 0;

    aggregator->addEvent(selfApi);
    aggregator->addEvent(peerApi);
    aggregator->addEvent(groupEvent);
    aggregator->addEvent(p2pSelf);
    aggregator->addEvent(p2pSend);
    aggregator->addEvent(proxyOp);
    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    auto it                 = collectives.find("Comm8738_AlltoAll_2Ranks");
    ASSERT_NE(it, collectives.end());
    EXPECT_EQ(it->second.totalBytes, 4096u);
    EXPECT_NEAR(it->second.getAverageTime(), 24.0, 1e-3);
    EXPECT_TRUE(aggregator->getP2Ps().empty());
}

TEST_F(WindowAggregatorTest, AlltoAllCollectiveSynthesis_NoPeerEventsYieldsNothing)
{
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

TEST_F(WindowAggregatorTest, AlltoAllCollectiveSynthesis_PipelineParallelStaysInP2P)
{
    auto commState       = std::make_unique<CommunicatorState>();
    commState->comm_hash = 0x1234;
    commState->nranks    = 2;
    commState->rank      = 0;
    commState->comm_type = CommunicatorState::CommType::P2P;

    auto groupEvent = createGroupEvent(0.5, 10.0, commState.get());
    auto peerApi    = createP2pApiEvent("Send", 0.0, 0.1, commState.get());

    otelEventHandle_t p2pSend = {};
    p2pSend.type              = ncclProfileP2p;
    p2pSend.p2p.func          = "Send";
    p2pSend.p2p.peer          = 1;
    p2pSend.p2p.nChannels     = 1;
    p2pSend.p2p.bytes         = 65536;
    p2pSend.startTs           = 1.0;
    p2pSend.endTs             = 20.0;
    p2pSend.parentObj         = &peerApi;
    p2pSend.commState         = commState.get();
    p2pSend.rank              = 0;

    aggregator->addEvent(peerApi);
    aggregator->addEvent(groupEvent);
    aggregator->addEvent(p2pSend);
    aggregator->finalize();

    EXPECT_TRUE(aggregator->getCollectives().empty());
    EXPECT_FALSE(aggregator->getP2Ps().empty());
}

TEST_F(WindowAggregatorTest, AlltoAllCollectiveSynthesis_GroupOrderingStillStaysCollectiveOnly)
{
    auto commState       = std::make_unique<CommunicatorState>();
    commState->comm_hash = 0xCAFE;
    commState->nranks    = 4;
    commState->rank      = 0;
    commState->comm_type = CommunicatorState::CommType::COLLECTIVE;

    auto groupEvent1 = createGroupEvent(0.5, 2.0, commState.get());
    auto selfApi1    = createP2pApiEvent("Send", 0.0, 0.1, commState.get());
    auto peer1Api1   = createP2pApiEvent("Send", 0.1, 0.2, commState.get());
    auto peer2Api1   = createP2pApiEvent("Send", 0.2, 0.3, commState.get());
    auto peer3Api1   = createP2pApiEvent("Send", 0.3, 0.4, commState.get());

    auto p2pSelf1      = createP2PEvent("Send", 0, 1, 1024, 0.8, 3.0);
    p2pSelf1.parentObj = &selfApi1;
    p2pSelf1.commState = commState.get();
    p2pSelf1.rank      = 0;

    auto p2pPeer11      = createP2PEvent("Send", 1, 1, 1024, 1.0, 3.1);
    p2pPeer11.parentObj = &peer1Api1;
    p2pPeer11.commState = commState.get();
    p2pPeer11.rank      = 0;

    auto p2pPeer21      = createP2PEvent("Send", 2, 1, 1024, 1.1, 3.2);
    p2pPeer21.parentObj = &peer2Api1;
    p2pPeer21.commState = commState.get();
    p2pPeer21.rank      = 0;

    auto p2pPeer31      = createP2PEvent("Send", 3, 1, 1024, 1.2, 3.3);
    p2pPeer31.parentObj = &peer3Api1;
    p2pPeer31.commState = commState.get();
    p2pPeer31.rank      = 0;

    auto groupEvent2 = createGroupEvent(10.5, 12.0, commState.get());
    auto selfApi2    = createP2pApiEvent("Send", 10.0, 10.1, commState.get());
    auto peer1Api2   = createP2pApiEvent("Send", 10.1, 10.2, commState.get());
    auto peer2Api2   = createP2pApiEvent("Send", 10.2, 10.3, commState.get());
    auto peer3Api2   = createP2pApiEvent("Send", 10.3, 10.4, commState.get());

    auto p2pSelf2      = createP2PEvent("Send", 0, 1, 1024, 10.8, 13.0);
    p2pSelf2.parentObj = &selfApi2;
    p2pSelf2.commState = commState.get();
    p2pSelf2.rank      = 0;

    auto p2pPeer12      = createP2PEvent("Send", 1, 1, 1024, 11.0, 13.1);
    p2pPeer12.parentObj = &peer1Api2;
    p2pPeer12.commState = commState.get();
    p2pPeer12.rank      = 0;

    auto p2pPeer22      = createP2PEvent("Send", 2, 1, 1024, 11.1, 13.2);
    p2pPeer22.parentObj = &peer2Api2;
    p2pPeer22.commState = commState.get();
    p2pPeer22.rank      = 0;

    auto p2pPeer32      = createP2PEvent("Send", 3, 1, 1024, 11.2, 13.3);
    p2pPeer32.parentObj = &peer3Api2;
    p2pPeer32.commState = commState.get();
    p2pPeer32.rank      = 0;

    aggregator->addEvent(selfApi1);
    aggregator->addEvent(peer1Api1);
    aggregator->addEvent(peer2Api1);
    aggregator->addEvent(peer3Api1);
    aggregator->addEvent(p2pSelf1);
    aggregator->addEvent(p2pPeer11);
    aggregator->addEvent(p2pPeer21);
    aggregator->addEvent(p2pPeer31);
    aggregator->addEvent(groupEvent1);

    aggregator->addEvent(selfApi2);
    aggregator->addEvent(peer1Api2);
    aggregator->addEvent(peer2Api2);
    aggregator->addEvent(peer3Api2);
    aggregator->addEvent(groupEvent2);
    aggregator->addEvent(p2pSelf2);
    aggregator->addEvent(p2pPeer12);
    aggregator->addEvent(p2pPeer22);
    aggregator->addEvent(p2pPeer32);
    aggregator->finalize();

    const auto& collectives = aggregator->getCollectives();
    auto it                 = collectives.find("Comm51966_AlltoAll_4Ranks");
    ASSERT_NE(it, collectives.end());
    EXPECT_EQ(it->second.count, 2u);
    EXPECT_EQ(it->second.totalBytes, 8192u);
    EXPECT_TRUE(aggregator->getP2Ps().empty());
}

TEST_F(WindowAggregatorTest, AlltoAllGroupedP2PsDoNotTriggerCudaGraph)
{
    auto commState       = std::make_unique<CommunicatorState>();
    commState->comm_hash = 0xABCD;
    commState->nranks    = 4;
    commState->rank      = 0;
    commState->comm_type = CommunicatorState::CommType::COLLECTIVE;

    auto groupEvent = createGroupEvent(0.5, 10.0, commState.get());
    auto selfApi    = createP2pApiEvent("Send", 0.0, 0.1, commState.get());
    auto peer1Api   = createP2pApiEvent("Send", 0.1, 0.2, commState.get());
    auto peer2Api   = createP2pApiEvent("Send", 0.2, 0.3, commState.get());
    auto peer3Api   = createP2pApiEvent("Send", 0.3, 0.4, commState.get());

    auto p2pSelf      = createP2PEvent("Send", 0, 1, 2048, 1.0, 12.0);
    p2pSelf.parentObj = &selfApi;
    p2pSelf.commState = commState.get();
    p2pSelf.rank      = 0;

    auto p2p1      = createP2PEvent("Send", 1, 1, 2048, 1.1, 20.0);
    p2p1.parentObj = &peer1Api;
    p2p1.commState = commState.get();
    p2p1.rank      = 0;

    auto p2p2      = createP2PEvent("Send", 2, 1, 2048, 1.2, 21.0);
    p2p2.parentObj = &peer2Api;
    p2p2.commState = commState.get();
    p2p2.rank      = 0;

    auto p2p3      = createP2PEvent("Send", 3, 1, 2048, 1.3, 22.0);
    p2p3.parentObj = &peer3Api;
    p2p3.commState = commState.get();
    p2p3.rank      = 0;

    constexpr uint64_t sharedPTimerStart = 0x1000;
    auto kch1      = createKernelChEvent(0, sharedPTimerStart, sharedPTimerStart + 10, 2.0, 3.0, &p2p1);
    kch1.commState = commState.get();
    auto kch2      = createKernelChEvent(0, sharedPTimerStart, sharedPTimerStart + 20, 2.1, 3.1, &p2p2);
    kch2.commState = commState.get();
    auto kch3      = createKernelChEvent(0, sharedPTimerStart, sharedPTimerStart + 30, 2.2, 3.2, &p2p3);
    kch3.commState = commState.get();

    aggregator->addEvent(selfApi);
    aggregator->addEvent(peer1Api);
    aggregator->addEvent(peer2Api);
    aggregator->addEvent(peer3Api);
    aggregator->addEvent(groupEvent);
    aggregator->addEvent(p2pSelf);
    aggregator->addEvent(p2p1);
    aggregator->addEvent(p2p2);
    aggregator->addEvent(p2p3);
    aggregator->addEvent(kch1);
    aggregator->addEvent(kch2);
    aggregator->addEvent(kch3);
    aggregator->finalize();

    EXPECT_STREQ(commState->getScaleUpExecModeString(), "non_cuda_graph");
    EXPECT_TRUE(aggregator->getP2Ps().empty());
    EXPECT_NE(aggregator->getCollectives().find("Comm43981_AlltoAll_4Ranks"), aggregator->getCollectives().end());
}