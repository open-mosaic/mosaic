// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <string>

#include "../../../telemetry_internal.h"
#include "../../support/event_handle_builders.h"
#include "../../support/telemetry_test_fixture.h"

namespace
{
void loadP2PTransferWindow(CommunicatorState& commState, uint8_t windowIdx, double baseTs)
{
    auto* buffer        = commState.buffers[windowIdx];
    buffer[0]           = testsupport::makeP2PEvent("Send", 1, 1, 512, baseTs, baseTs + 5.0);
    buffer[0].commState = &commState;

    buffer[1]           = testsupport::makeProxyOpEvent(1, 0, 64, baseTs + 1.0, baseTs + 4.0, &buffer[0]);
    buffer[1].commState = &commState;

    buffer[2]           = testsupport::makeProxyStepEvent(0, 32, baseTs + 1.0, baseTs + 2.0, baseTs + 4.0, &buffer[1]);
    buffer[2].commState = &commState;

    testsupport::loadWindowEvents(commState, windowIdx, {buffer[0], buffer[1], buffer[2]});
}

class TelemetryWindowProcessorTest : public testsupport::TelemetryFixture
{
};

TEST_F(TelemetryWindowProcessorTest, RepeatedCollectiveWindowsEmitPrimerAndResetWindowState)
{
    setScaleUpExecMode(CommunicatorState::ScaleUpExecMode::CUDA_GRAPH);
    otelEventHandle_t event =
        testsupport::makeCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, 4096, 10.0, 40.0, &commState);

    testsupport::loadWindowEvent(commState, 0, event);
    processWindow(&commState, 0);

    WindowMetadata* firstWindow = commState.get_window_metadata(0);
    EXPECT_EQ(WINDOW_READY, firstWindow->state.load(std::memory_order_acquire));
    EXPECT_EQ(0u, firstWindow->element_count.load(std::memory_order_acquire));
    EXPECT_EQ(0u, firstWindow->in_progress_count.load(std::memory_order_acquire));
    EXPECT_TRUE(telemetrytest::getCollectiveExportCalls().empty());

    testsupport::loadWindowEvent(commState, 1, event);
    processWindow(&commState, 1);

    WindowMetadata* secondWindow = commState.get_window_metadata(1);
    EXPECT_EQ(WINDOW_READY, secondWindow->state.load(std::memory_order_acquire));
    EXPECT_EQ(0u, secondWindow->element_count.load(std::memory_order_acquire));
    EXPECT_EQ(0u, secondWindow->in_progress_count.load(std::memory_order_acquire));
    ASSERT_EQ(1u, telemetrytest::getCollectiveExportCalls().size());
    const auto& primerExport = telemetrytest::getCollectiveExportCalls()[0];
    EXPECT_EQ("PRIMER", primerExport.exportTag);
    EXPECT_EQ("Comm4660_AllReduce_Ring_Simple_2Chnl", primerExport.key);
    EXPECT_EQ("cuda_graph", primerExport.scaleUpExecMode);
}

TEST_F(TelemetryWindowProcessorTest, IncompleteEventsAreSkippedAndWindowIsCleared)
{
    otelEventHandle_t event =
        testsupport::makeCollectiveEventWithCommState("AllReduce", "Ring", "Simple", 2, 1024, 10.0, 0.0, &commState);

    testsupport::loadWindowEvent(commState, 0, event);
    processWindow(&commState, 0);

    WindowMetadata* window = commState.get_window_metadata(0);
    EXPECT_EQ(WINDOW_READY, window->state.load(std::memory_order_acquire));
    EXPECT_EQ(0u, window->element_count.load(std::memory_order_acquire));
    EXPECT_EQ(0u, window->in_progress_count.load(std::memory_order_acquire));
    EXPECT_TRUE(telemetrytest::getCollectiveExportCalls().empty());
}

TEST_F(TelemetryWindowProcessorTest, RepeatedP2PWindowsCoverPrimerAndDirectExportPaths)
{
    commState.nranks    = 2;
    commState.comm_type = CommunicatorState::CommType::P2P;
    setScaleUpExecMode(CommunicatorState::ScaleUpExecMode::CUDA_GRAPH);

    loadP2PTransferWindow(commState, 0, 10.0);
    processWindow(&commState, 0);
    EXPECT_TRUE(telemetrytest::getP2PExportCalls().empty());
    EXPECT_TRUE(telemetrytest::getRankExportCalls().empty());
    EXPECT_TRUE(telemetrytest::getTransferExportCalls().empty());

    loadP2PTransferWindow(commState, 1, 110.0);
    processWindow(&commState, 1);
    ASSERT_EQ(1u, telemetrytest::getP2PExportCalls().size());
    ASSERT_EQ(1u, telemetrytest::getRankExportCalls().size());
    ASSERT_EQ(1u, telemetrytest::getTransferExportCalls().size());
    EXPECT_EQ("PRIMER", telemetrytest::getP2PExportCalls()[0].exportTag);
    EXPECT_EQ("PRIMER", telemetrytest::getRankExportCalls()[0].exportTag);
    EXPECT_EQ("PRIMER", telemetrytest::getTransferExportCalls()[0].exportTag);

    loadP2PTransferWindow(commState, 2, 210.0);
    processWindow(&commState, 2);
    ASSERT_EQ(2u, telemetrytest::getP2PExportCalls().size());
    ASSERT_EQ(2u, telemetrytest::getRankExportCalls().size());
    ASSERT_EQ(2u, telemetrytest::getTransferExportCalls().size());
    EXPECT_EQ("STANDARD", telemetrytest::getP2PExportCalls()[1].exportTag);
    EXPECT_EQ("STANDARD", telemetrytest::getRankExportCalls()[1].exportTag);
    EXPECT_EQ("STANDARD", telemetrytest::getTransferExportCalls()[1].exportTag);

    loadP2PTransferWindow(commState, 3, 310.0);
    processWindow(&commState, 3);
    ASSERT_EQ(3u, telemetrytest::getP2PExportCalls().size());
    ASSERT_EQ(3u, telemetrytest::getRankExportCalls().size());
    ASSERT_EQ(3u, telemetrytest::getTransferExportCalls().size());
    EXPECT_EQ("STANDARD", telemetrytest::getP2PExportCalls()[2].exportTag);
    EXPECT_EQ("STANDARD", telemetrytest::getRankExportCalls()[2].exportTag);
    EXPECT_EQ("STANDARD", telemetrytest::getTransferExportCalls()[2].exportTag);

    WindowMetadata* finalWindow = commState.get_window_metadata(3);
    EXPECT_EQ(WINDOW_READY, finalWindow->state.load(std::memory_order_acquire));
    EXPECT_EQ(0u, finalWindow->element_count.load(std::memory_order_acquire));
    EXPECT_EQ(0u, finalWindow->in_progress_count.load(std::memory_order_acquire));
}
}  // namespace