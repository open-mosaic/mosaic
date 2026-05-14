// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <map>
#include <string>

#include "../../../aggregation.h"
#include "../../support/telemetry_test_fixture.h"

namespace
{
AggregatedCollective makeCollective(size_t bytes, double timeUs, size_t transferBytes, double transferTimeUs)
{
    AggregatedCollective collective;
    collective.addCollective(bytes, timeUs);
    collective.addTransferToCache(transferBytes, transferTimeUs);
    return collective;
}

AggregatedP2P makeP2P(size_t bytes, double timeUs, size_t transferBytes, double transferTimeUs)
{
    AggregatedP2P p2p;
    p2p.addP2P(bytes, timeUs);
    p2p.addTransferToCache(transferBytes, transferTimeUs);
    return p2p;
}

AggregatedTransfer makeTransfer(size_t bytes, double timeUs, double startTs, double endTs)
{
    AggregatedTransfer transfer;
    transfer.addTransferWithTimestamps(bytes, timeUs, startTs, endTs);
    return transfer;
}

AggregatedTransfer makeRichTransfer(double baseStartTs)
{
    AggregatedTransfer transfer;
    transfer.addTransferWithTimestamps(1000, 20.0, baseStartTs + 0.0, baseStartTs + 20.0);
    transfer.addTransferWithTimestamps(2000, 35.0, baseStartTs + 25.0, baseStartTs + 60.0);
    transfer.addTransferWithTimestamps(4000, 65.0, baseStartTs + 70.0, baseStartTs + 135.0);
    return transfer;
}

class TelemetryPrimerStateMachineTest : public testsupport::TelemetryFixture
{
};

TEST_F(TelemetryPrimerStateMachineTest, CollectiveCudaGraphPrimerEmitsPrimerThenStandard)
{
    setScaleUpExecMode(CommunicatorState::ScaleUpExecMode::CUDA_GRAPH);

    const std::string key = "collective_key";
    registerCollectivePrimer(&commState, key, makeCollective(1024, 20.0, 512, 4.0));

    auto handledDuringPrimer =
        processPendingCollectivePrimers(&commState, {
                                                        {key, makeCollective(2048, 40.0, 1024, 8.0)}
    });

    EXPECT_EQ(1u, handledDuringPrimer.count(key));
    ASSERT_EQ(1u, telemetrytest::getCollectiveExportCalls().size());
    const auto& primerExport = telemetrytest::getCollectiveExportCalls()[0];
    EXPECT_EQ("PRIMER", primerExport.exportTag);
    EXPECT_EQ(key, primerExport.key);
    EXPECT_EQ("cuda_graph", primerExport.scaleUpExecMode);
    EXPECT_DOUBLE_EQ(0.0, primerExport.emit.count);
    EXPECT_DOUBLE_EQ(0.0, primerExport.emit.totalBytes);
    EXPECT_FALSE(isCollectivePrimerDone(&commState, key));

    auto handledDuringStandard =
        processPendingCollectivePrimers(&commState, {
                                                        {key, makeCollective(512, 10.0, 256, 2.0)}
    });

    EXPECT_EQ(1u, handledDuringStandard.count(key));
    ASSERT_EQ(2u, telemetrytest::getCollectiveExportCalls().size());
    const auto& standardExport = telemetrytest::getCollectiveExportCalls()[1];
    EXPECT_EQ("STANDARD", standardExport.exportTag);
    EXPECT_EQ(key, standardExport.key);
    EXPECT_DOUBLE_EQ(3.0, standardExport.emit.count);
    EXPECT_DOUBLE_EQ(3584.0, standardExport.emit.totalBytes);
    EXPECT_DOUBLE_EQ(70.0, standardExport.emit.totalTimeUs);
    EXPECT_TRUE(isCollectivePrimerDone(&commState, key));
}

TEST_F(TelemetryPrimerStateMachineTest, RankPrimerWaitsWhileScaleUpModeIsUnknown)
{
    setScaleUpExecMode(CommunicatorState::ScaleUpExecMode::UNKNOWN);

    const std::string key = "rank_key";
    registerRankPrimer(&commState, key, makeTransfer(1024, 20.0, 0.0, 20.0));

    auto handled = processPendingRankPrimers(&commState, {
                                                             {key, makeTransfer(2048, 35.0, 25.0, 60.0)}
    });

    EXPECT_EQ(1u, handled.count(key));
    EXPECT_TRUE(telemetrytest::getRankExportCalls().empty());
    EXPECT_FALSE(isRankPrimerDone(&commState, key));
}

TEST_F(TelemetryPrimerStateMachineTest, P2PPrimerStabilizesBeforeExportingStandardData)
{
    setScaleUpExecMode(CommunicatorState::ScaleUpExecMode::NON_CUDA_GRAPH);

    const std::string key = "p2p_key";
    registerP2PPrimer(&commState, key, makeP2P(128, 10.0, 64, 2.0));

    EXPECT_EQ(1u, processPendingP2PPrimers(&commState,
                                           {
                                               {key, makeP2P(256, 20.0, 128, 4.0)}
    })
                      .count(key));
    EXPECT_TRUE(telemetrytest::getP2PExportCalls().empty());

    EXPECT_EQ(1u, processPendingP2PPrimers(&commState,
                                           {
                                               {key, makeP2P(512, 30.0, 256, 8.0)}
    })
                      .count(key));
    EXPECT_TRUE(telemetrytest::getP2PExportCalls().empty());

    EXPECT_EQ(1u, processPendingP2PPrimers(&commState,
                                           {
                                               {key, makeP2P(1024, 40.0, 512, 16.0)}
    })
                      .count(key));
    ASSERT_EQ(1u, telemetrytest::getP2PExportCalls().size());
    EXPECT_EQ("PRIMER", telemetrytest::getP2PExportCalls()[0].exportTag);
    EXPECT_EQ("non_cuda_graph", telemetrytest::getP2PExportCalls()[0].scaleUpExecMode);

    EXPECT_EQ(1u, processPendingP2PPrimers(&commState,
                                           {
                                               {key, makeP2P(2048, 50.0, 1024, 32.0)}
    })
                      .count(key));
    ASSERT_EQ(2u, telemetrytest::getP2PExportCalls().size());
    const auto& standardExport = telemetrytest::getP2PExportCalls()[1];
    EXPECT_EQ("STANDARD", standardExport.exportTag);
    EXPECT_DOUBLE_EQ(3968.0 / 5.0, standardExport.emit.avgBytes);
    EXPECT_DOUBLE_EQ(150.0 / 5.0, standardExport.emit.avgTimeUs);
    EXPECT_TRUE(isP2PPrimerDone(&commState, key));
}

TEST_F(TelemetryPrimerStateMachineTest, RankPrimerEmitsPrimerThenStandardDataInCudaGraphMode)
{
    setScaleUpExecMode(CommunicatorState::ScaleUpExecMode::CUDA_GRAPH);

    const std::string key = "rank_rich_key";
    registerRankPrimer(&commState, key, makeRichTransfer(0.0));

    EXPECT_EQ(1u, processPendingRankPrimers(&commState,
                                            {
                                                {key, makeRichTransfer(200.0)}
    })
                      .count(key));
    ASSERT_EQ(1u, telemetrytest::getRankExportCalls().size());
    EXPECT_EQ("PRIMER", telemetrytest::getRankExportCalls()[0].exportTag);
    EXPECT_EQ("cuda_graph", telemetrytest::getRankExportCalls()[0].scaleUpExecMode);

    EXPECT_EQ(1u, processPendingRankPrimers(&commState,
                                            {
                                                {key, makeRichTransfer(400.0)}
    })
                      .count(key));
    ASSERT_EQ(2u, telemetrytest::getRankExportCalls().size());
    const auto& standardExport = telemetrytest::getRankExportCalls()[1];
    EXPECT_EQ("STANDARD", standardExport.exportTag);
    EXPECT_EQ(21000u, standardExport.emit.totalBytes);
    EXPECT_GT(standardExport.emit.activeTimeUs, 0.0);
    EXPECT_TRUE(isRankPrimerDone(&commState, key));
}

TEST_F(TelemetryPrimerStateMachineTest, TransferPrimerForceEmitsAfterMaxWait)
{
    setScaleUpExecMode(CommunicatorState::ScaleUpExecMode::UNKNOWN);

    const std::string key = "transfer_key";
    registerTransferPrimer(&commState, key, makeRichTransfer(0.0));

    for (uint32_t window = 0; window < PRIMER_MAX_WAIT_WINDOWS; ++window)
    {
        EXPECT_EQ(1u, processPendingTransferPrimers(&commState,
                                                    {
                                                        {key, makeRichTransfer(200.0 * (window + 1))}
        })
                          .count(key));
    }
    EXPECT_TRUE(telemetrytest::getTransferExportCalls().empty());

    EXPECT_EQ(1u, processPendingTransferPrimers(&commState,
                                                {
                                                    {key, makeRichTransfer(2500.0)}
    })
                      .count(key));
    ASSERT_EQ(1u, telemetrytest::getTransferExportCalls().size());
    EXPECT_EQ("PRIMER", telemetrytest::getTransferExportCalls()[0].exportTag);
    EXPECT_EQ("unknown", telemetrytest::getTransferExportCalls()[0].scaleUpExecMode);

    EXPECT_EQ(1u, processPendingTransferPrimers(&commState,
                                                {
                                                    {key, makeRichTransfer(2800.0)}
    })
                      .count(key));
    ASSERT_EQ(2u, telemetrytest::getTransferExportCalls().size());
    EXPECT_EQ("STANDARD", telemetrytest::getTransferExportCalls()[1].exportTag);
    EXPECT_GT(telemetrytest::getTransferExportCalls()[1].emit.avgSize, 0.0);
    EXPECT_TRUE(isTransferPrimerDone(&commState, key));
}

TEST_F(TelemetryPrimerStateMachineTest, CommunicatorCleanupRemovesCompletedPrimerState)
{
    setScaleUpExecMode(CommunicatorState::ScaleUpExecMode::CUDA_GRAPH);

    const std::string key = "collective_key";
    registerCollectivePrimer(&commState, key, makeCollective(1024, 20.0, 512, 4.0));

    EXPECT_EQ(1u, processPendingCollectivePrimers(&commState,
                                                  {
                                                      {key, makeCollective(2048, 40.0, 1024, 8.0)}
    })
                      .count(key));
    EXPECT_EQ(1u, processPendingCollectivePrimers(&commState,
                                                  {
                                                      {key, makeCollective(512, 10.0, 256, 2.0)}
    })
                      .count(key));
    ASSERT_TRUE(isCollectivePrimerDone(&commState, key));

    cleanupTelemetryPrimerStateForCommunicator(&commState);

    EXPECT_FALSE(isCollectivePrimerDone(&commState, key));

    registerCollectivePrimer(&commState, key, makeCollective(256, 5.0, 128, 1.0));
    EXPECT_EQ(1u, processPendingCollectivePrimers(&commState,
                                                  {
                                                      {key, makeCollective(512, 10.0, 256, 2.0)}
    })
                      .count(key));
    ASSERT_EQ(3u, telemetrytest::getCollectiveExportCalls().size());
    EXPECT_EQ("PRIMER", telemetrytest::getCollectiveExportCalls()[2].exportTag);
}
}  // namespace