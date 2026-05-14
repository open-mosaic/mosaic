// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "../../../aggregation.h"
#include "../../../telemetry_internal.h"

namespace
{
TEST(TelemetryEmitViewsTest, CollectiveEmitViewUsesAggregateAndTransferStats)
{
    AggregatedCollective collective;
    collective.addCollective(4096, 120.0);
    collective.addCollective(2048, 60.0);
    collective.addTransferToCache(1024, 10.0);
    collective.addTransferToCache(2048, 20.0);

    CollectiveEmitView emit                 = makeStandardCollectiveEmitView(collective);
    CollectiveExportEligibility eligibility = computeCollectiveEligibility(collective);

    EXPECT_DOUBLE_EQ(2.0, emit.count);
    EXPECT_DOUBLE_EQ(6144.0, emit.totalBytes);
    EXPECT_DOUBLE_EQ(180.0, emit.totalTimeUs);
    EXPECT_DOUBLE_EQ(3072.0, emit.avgBytes);
    EXPECT_DOUBLE_EQ(90.0, emit.avgTimeUs);
    EXPECT_DOUBLE_EQ(1.0, emit.avgNumTransfers);
    EXPECT_DOUBLE_EQ(1536.0, emit.avgTransferSize);
    EXPECT_DOUBLE_EQ(15.0, emit.avgTransferTime);
    EXPECT_TRUE(eligibility.export_core);
    EXPECT_TRUE(eligibility.export_transfers);
    EXPECT_TRUE(eligibility.export_transfer_time);
}

TEST(TelemetryEmitViewsTest, P2PEligibilityDependsOnTransferCache)
{
    AggregatedP2P p2p;
    p2p.addP2P(512, 8.0);

    P2PEmitView emit                 = makeStandardP2PEmitView(p2p);
    P2PExportEligibility eligibility = computeP2PEligibility(p2p);

    EXPECT_DOUBLE_EQ(512.0, emit.avgBytes);
    EXPECT_DOUBLE_EQ(8.0, emit.avgTimeUs);
    EXPECT_DOUBLE_EQ(0.0, emit.avgNumTransfers);
    EXPECT_DOUBLE_EQ(0.0, emit.avgTransferSize);
    EXPECT_DOUBLE_EQ(0.0, emit.avgTransferTime);
    EXPECT_TRUE(eligibility.export_core);
    EXPECT_FALSE(eligibility.export_transfers);
    EXPECT_FALSE(eligibility.export_transfer_time);
}

TEST(TelemetryEmitViewsTest, RankAndTransferEmitViewsExposeRegressionDerivedValues)
{
    AggregatedTransfer transfer;
    transfer.addTransferWithTimestamps(1000, 20.0, 0.0, 20.0);
    transfer.addTransferWithTimestamps(2000, 35.0, 20.0, 55.0);
    transfer.addTransferWithTimestamps(4000, 65.0, 60.0, 125.0);

    RankEmitView rankEmit                         = makeStandardRankEmitView(transfer);
    RankExportEligibility rankEligibility         = computeRankEligibility(transfer);
    TransferEmitView transferEmit                 = makeStandardTransferEmitView(transfer);
    TransferExportEligibility transferEligibility = computeTransferEligibility(transfer);

    EXPECT_EQ(7000u, rankEmit.totalBytes);
    EXPECT_DOUBLE_EQ(120.0, rankEmit.activeTimeUs);
    EXPECT_GT(rankEmit.latencyUs, 0.0);
    EXPECT_GT(rankEmit.rateMBps, 0.0);
    EXPECT_TRUE(rankEligibility.export_latency);
    EXPECT_TRUE(rankEligibility.export_rate);

    EXPECT_DOUBLE_EQ(transfer.getAverageSize(), transferEmit.avgSize);
    EXPECT_DOUBLE_EQ(transfer.getAverageTime(), transferEmit.avgTime);
    EXPECT_GT(transferEmit.latencyUs, 0.0);
    EXPECT_TRUE(transferEligibility.export_channel_metrics);
    EXPECT_TRUE(transferEligibility.export_avg_time);
    EXPECT_TRUE(transferEligibility.export_latency);
}
}  // namespace