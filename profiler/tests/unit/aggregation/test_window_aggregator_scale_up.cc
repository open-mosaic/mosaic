// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "../../support/window_aggregator_fixture.h"

// NCCL < 2.31 graph replays: different Colls report the same stale pTimerStart. That classifies the
// communicator as cuda_graph and the operations fall back to count/time plus byte volume only.
TEST_F(WindowAggregatorTest, ScaleUpCudaGraphSharedPTimersCountTimeOnlyAndSuppressesTransfers)
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

namespace
{
std::unique_ptr<CommunicatorState> makeScaleUpCommState(bool sawGraphCapture)
{
    std::unique_ptr<CommunicatorState> commState(new CommunicatorState());
    commState->nranks     = 8;
    commState->rank       = 0;
    commState->comm_hash  = 0;
    commState->local_rank = 0;
    commState->hostname   = "test";
    commState->sawGraphCapture.store(sawGraphCapture);
    return commState;
}

// Shape observed on NCCL 2.32.3 graph replays: every replay has its own pTimerStart, shared only by the
// channels of that one Coll, with a real KernelChStop pTimer.
// Storage is reserved up front so parentObj pointers into the vector stay valid.
void addFreshReplayOps(WindowAggregator& aggregator, std::vector<otelEventHandle_t>& events,
                       CommunicatorState* commState)
{
    const size_t sizes[] = {1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024};
    events.reserve(12);
    double t = 100.0;
    for (int i = 0; i < 3; i++)
    {
        events.push_back(testsupport::makeCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, sizes[i], t,
                                                                       t + 5.0, commState));
        otelEventHandle_t* coll = &events.back();
        aggregator.addEvent(*coll);

        const uint64_t pTimerStart = 9000000 + (uint64_t)i * 100000;
        const double spanUs        = 50.0 * (i + 1);
        for (uint8_t ch = 0; ch < 2; ch++)
        {
            events.push_back(testsupport::makeKernelChEvent(ch, pTimerStart, pTimerStart + (uint64_t)(spanUs * 1000),
                                                            t + 1.0, t + 1.0 + spanUs, coll));
            aggregator.addEvent(events.back());
        }
        t += 1000.0;
    }
}
}  // namespace

// NCCL >= 2.31: graph capture is reported via graphCaptured and replays carry fresh KernelCh timing.
// The communicator is labelled cuda_graph, and transfer timing must match the non-graph path exactly.
TEST_F(WindowAggregatorTest, GraphCapturedCommWithFreshKernelTimersMatchesNonGraphTiming)
{
    auto graphComm = makeScaleUpCommState(true);
    std::vector<otelEventHandle_t> graphEvents;
    addFreshReplayOps(*aggregator, graphEvents, graphComm.get());
    aggregator->finalize();

    auto eagerComm = makeScaleUpCommState(false);
    WindowAggregator eagerAggregator(0);
    std::vector<otelEventHandle_t> eagerEvents;
    addFreshReplayOps(eagerAggregator, eagerEvents, eagerComm.get());
    eagerAggregator.finalize();

    EXPECT_STREQ(graphComm->getScaleUpExecModeString(), "cuda_graph");
    EXPECT_STREQ(eagerComm->getScaleUpExecModeString(), "non_cuda_graph");

    const auto& graphRanks = aggregator->getRankTransfers();
    const auto& eagerRanks = eagerAggregator.getRankTransfers();
    ASSERT_FALSE(graphRanks.empty());
    ASSERT_EQ(graphRanks.size(), eagerRanks.size());
    for (const auto& rt : graphRanks)
    {
        auto eagerIt = eagerRanks.find(rt.first);
        ASSERT_NE(eagerIt, eagerRanks.end()) << rt.first;
        EXPECT_FALSE(rt.second.intervals.empty());
        EXPECT_GT(rt.second.totalTimeUs, 0.0);
        double rate = 0.0;
        EXPECT_TRUE(rt.second.getRateFromActiveTime(rate));
        EXPECT_EQ(rt.second.count, eagerIt->second.count);
        EXPECT_EQ(rt.second.totalBytes, eagerIt->second.totalBytes);
        EXPECT_DOUBLE_EQ(rt.second.totalTimeUs, eagerIt->second.totalTimeUs);
        EXPECT_EQ(rt.second.intervals.size(), eagerIt->second.intervals.size());
    }

    const auto& graphChannels = aggregator->getChannelTransfers();
    const auto& eagerChannels = eagerAggregator.getChannelTransfers();
    ASSERT_EQ(graphChannels.size(), eagerChannels.size());
    for (const auto& ct : graphChannels)
    {
        auto eagerIt = eagerChannels.find(ct.first);
        ASSERT_NE(eagerIt, eagerChannels.end()) << ct.first;
        EXPECT_GT(ct.second.totalTimeUs, 0.0);
        EXPECT_DOUBLE_EQ(ct.second.totalTimeUs, eagerIt->second.totalTimeUs);
    }

    const auto& graphColls = aggregator->getCollectives();
    const auto& eagerColls = eagerAggregator.getCollectives();
    auto graphCollIt       = graphColls.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    auto eagerCollIt       = eagerColls.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(graphCollIt, graphColls.end());
    ASSERT_NE(eagerCollIt, eagerColls.end());
    EXPECT_EQ(graphCollIt->second.count, 3);
    EXPECT_DOUBLE_EQ(graphCollIt->second.totalTimeUs, eagerCollIt->second.totalTimeUs);
    EXPECT_DOUBLE_EQ(graphCollIt->second.cachedTotalTransferTimeUs, eagerCollIt->second.cachedTotalTransferTimeUs);
}

// A window can mix stale (shared pTimer) and fresh-looking replays. Once stale timers are seen, no KernelCh
// timing on the communicator is trusted, so even the op with its own timer stays volume-only.
TEST_F(WindowAggregatorTest, CudaGraphMixedWindowStaleTimersMakeAllOpsVolumeOnly)
{
    const uint64_t sharedPTimerStart = 5000000;
    const uint64_t freshPTimerStart  = 6000000;
    auto commState                   = makeScaleUpCommState(false);
    const size_t bytes               = 1024 * 1024;

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
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, bytes, 500.0, 505.0, commState.get());
    aggregator->addEvent(coll3);
    auto kch3a = createKernelChEvent(0, freshPTimerStart, freshPTimerStart + 40000, 501.0, 541.0, &coll3);
    aggregator->addEvent(kch3a);
    auto kch3b = createKernelChEvent(1, freshPTimerStart, freshPTimerStart + 42000, 501.0, 543.0, &coll3);
    aggregator->addEvent(kch3b);

    aggregator->finalize();

    EXPECT_STREQ(commState->getScaleUpExecModeString(), "cuda_graph");
    EXPECT_TRUE(commState->sawStaleKernelTimers.load());

    const auto& rankTransfers = aggregator->getRankTransfers();
    ASSERT_EQ(rankTransfers.size(), 1u);
    const AggregatedTransfer& rank = rankTransfers.begin()->second;
    EXPECT_GT(rank.count, 0);
    EXPECT_TRUE(rank.intervals.empty());
    EXPECT_EQ(rank.totalTimeUs, 0.0);
}

// Stale timers are sticky across windows: a later window whose single replay has no twin is still untrusted.
TEST_F(WindowAggregatorTest, StaleKernelTimersStayUntrustedInLaterWindows)
{
    const uint64_t sharedPTimerStart = 5000000;
    auto commState                   = makeScaleUpCommState(false);
    const size_t bytes               = 1024 * 1024;

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
    aggregator->finalize();
    ASSERT_TRUE(commState->sawStaleKernelTimers.load());

    WindowAggregator nextWindow(0);
    auto coll3 =
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, bytes, 500.0, 505.0, commState.get());
    nextWindow.addEvent(coll3);
    auto kch3a = createKernelChEvent(0, sharedPTimerStart, sharedPTimerStart + 40000, 501.0, 541.0, &coll3);
    nextWindow.addEvent(kch3a);
    auto kch3b = createKernelChEvent(1, sharedPTimerStart, sharedPTimerStart + 42000, 501.0, 543.0, &coll3);
    nextWindow.addEvent(kch3b);
    nextWindow.finalize();

    const auto& rankTransfers = nextWindow.getRankTransfers();
    ASSERT_FALSE(rankTransfers.empty());
    for (const auto& rt : rankTransfers)
    {
        EXPECT_GT(rt.second.totalBytes, 0u);
        EXPECT_TRUE(rt.second.intervals.empty());
    }
}

// P2P ops are not scanned for shared timers, but stale Coll timers on the same communicator make them
// volume-only too.
TEST_F(WindowAggregatorTest, StaleCollTimersMakeP2POnSameCommVolumeOnly)
{
    const uint64_t sharedPTimerStart = 5000000;
    auto commState                   = makeScaleUpCommState(false);
    const size_t bytes               = 1024 * 1024;

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

    auto p2p      = createP2PEvent("Send", 1, 1, bytes, 500.0, 505.0);
    p2p.commState = commState.get();
    aggregator->addEvent(p2p);
    auto kchP2p = createKernelChEvent(0, 7000000, 7040000, 501.0, 541.0, &p2p);
    aggregator->addEvent(kchP2p);

    aggregator->finalize();

    EXPECT_STREQ(commState->getScaleUpExecModeString(), "cuda_graph");
    const auto& p2ps = aggregator->getP2Ps();
    ASSERT_FALSE(p2ps.empty());

    bool sawP2PRank = false;
    for (const auto& rt : aggregator->getRankTransfers())
    {
        EXPECT_TRUE(rt.second.intervals.empty()) << rt.first;
        if (rt.first.find("_Pipeline") != std::string::npos) sawP2PRank = true;
    }
    EXPECT_TRUE(sawP2PRank);
}

// A graph-captured communicator without stale timers: an op with no KernelCh events takes the normal path,
// as it did before graphCaptured classification existed.
TEST_F(WindowAggregatorTest, GraphCapturedCommOpWithoutKernelChUsesNormalPath)
{
    auto commState     = makeScaleUpCommState(true);
    const size_t bytes = 1024 * 1024;

    auto coll =
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, bytes, 100.0, 150.0, commState.get());
    aggregator->addEvent(coll);
    aggregator->finalize();

    EXPECT_STREQ(commState->getScaleUpExecModeString(), "cuda_graph");
    EXPECT_FALSE(commState->sawStaleKernelTimers.load());

    const auto& rankTransfers = aggregator->getRankTransfers();
    ASSERT_FALSE(rankTransfers.empty());
    for (const auto& rt : rankTransfers)
    {
        EXPECT_FALSE(rt.second.intervals.empty());
        EXPECT_GT(rt.second.totalTimeUs, 0.0);
    }
}

// Once stale timers are seen, an op with no KernelCh is volume-only as well.
TEST_F(WindowAggregatorTest, StaleTimersCommOpWithoutKernelChIsVolumeOnly)
{
    auto commState = makeScaleUpCommState(true);
    commState->sawStaleKernelTimers.store(true);
    const size_t bytes = 1024 * 1024;

    auto coll =
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, bytes, 100.0, 150.0, commState.get());
    aggregator->addEvent(coll);
    aggregator->finalize();

    const auto& rankTransfers = aggregator->getRankTransfers();
    ASSERT_FALSE(rankTransfers.empty());
    for (const auto& rt : rankTransfers)
    {
        EXPECT_GT(rt.second.totalBytes, 0u);
        EXPECT_TRUE(rt.second.intervals.empty());
    }
}

// Graph replays without a usable KernelCh stop timer cannot be trusted and stay volume-only.
TEST_F(WindowAggregatorTest, GraphCapturedCommWithoutKernelChStopIsVolumeOnly)
{
    auto commState     = makeScaleUpCommState(true);
    const size_t bytes = 1024 * 1024;

    auto coll =
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, bytes, 100.0, 105.0, commState.get());
    aggregator->addEvent(coll);
    auto kch             = createKernelChEvent(0, 8000000, 0, 101.0, 151.0, &coll);
    kch.kernelCh.hasStop = false;
    aggregator->addEvent(kch);

    aggregator->finalize();

    EXPECT_STREQ(commState->getScaleUpExecModeString(), "cuda_graph");

    const auto& collectives = aggregator->getCollectives();
    auto it                 = collectives.find("Comm0_AllReduce_Ring_Simple_2Chnl");
    ASSERT_NE(it, collectives.end());
    EXPECT_EQ(it->second.count, 1);
    EXPECT_NEAR(it->second.totalTimeUs, 51.0, 0.1);

    const auto& rankTransfers = aggregator->getRankTransfers();
    ASSERT_FALSE(rankTransfers.empty());
    for (const auto& rt : rankTransfers)
    {
        EXPECT_GT(rt.second.totalBytes, 0u);
        EXPECT_TRUE(rt.second.intervals.empty());
        EXPECT_EQ(rt.second.totalTimeUs, 0.0);
    }
}

// graphCaptured is authoritative even when no KernelCh is present yet in the window.
TEST_F(WindowAggregatorTest, GraphCapturedFlagClassifiesWithoutKernelCh)
{
    auto commState = makeScaleUpCommState(true);

    auto coll =
        createCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, 4096, 100.0, 150.0, commState.get());
    aggregator->addEvent(coll);
    aggregator->finalize();

    EXPECT_STREQ(commState->getScaleUpExecModeString(), "cuda_graph");
}

// Without graphCaptured or shared timers, KernelCh channels sharing one Coll's start timer
// must not be mistaken for graph replay.
TEST_F(WindowAggregatorTest, ChannelsSharingOneCollPTimerStayNonCudaGraph)
{
    auto commState = makeScaleUpCommState(false);
    std::vector<otelEventHandle_t> events;
    addFreshReplayOps(*aggregator, events, commState.get());
    aggregator->finalize();

    EXPECT_STREQ(commState->getScaleUpExecModeString(), "non_cuda_graph");
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