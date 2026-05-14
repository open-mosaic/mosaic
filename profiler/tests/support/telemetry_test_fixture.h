// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef TESTS_SUPPORT_TELEMETRY_TEST_FIXTURE_H_
#define TESTS_SUPPORT_TELEMETRY_TEST_FIXTURE_H_

#include <gtest/gtest.h>

#include <atomic>
#include <initializer_list>

#include "../../communicator_state.h"
#include "../../telemetry_primer.h"
#include "../mocks/telemetry_export_mocks.h"

namespace testsupport
{
inline void initializeTelemetryCommState(
    CommunicatorState& commState,
    CommunicatorState::ScaleUpExecMode mode = CommunicatorState::ScaleUpExecMode::CUDA_GRAPH)
{
    commState.comm_hash      = 0x1234;
    commState.rank           = 3;
    commState.nranks         = 4;
    commState.nNodes         = 1;
    commState.hostname       = "test-host";
    commState.local_rank     = 1;
    commState.gpu_pci_bus_id = "0000:01:00.0";
    commState.gpu_uuid       = "GPU-test";
    commState.comm_type      = CommunicatorState::CommType::COLLECTIVE;
    commState.scaleUpExecMode.store(static_cast<uint8_t>(mode), std::memory_order_release);
}

inline void loadWindowEvent(CommunicatorState& commState, uint8_t windowIdx, const otelEventHandle_t& event)
{
    WindowMetadata* window          = commState.get_window_metadata(windowIdx);
    commState.buffers[windowIdx][0] = event;
    window->state.store(WINDOW_PROCESSING, std::memory_order_release);
    window->element_count.store(1, std::memory_order_release);
    window->in_progress_count.store(0, std::memory_order_release);
    window->start_time = event.startTs;
}

inline void loadWindowEvents(CommunicatorState& commState, uint8_t windowIdx,
                             std::initializer_list<otelEventHandle_t> events)
{
    WindowMetadata* window = commState.get_window_metadata(windowIdx);
    uint32_t index         = 0;
    double startTime       = 0.0;
    for (const auto& event : events)
    {
        if (index == 0) startTime = event.startTs;
        commState.buffers[windowIdx][index++] = event;
    }
    window->state.store(WINDOW_PROCESSING, std::memory_order_release);
    window->element_count.store(index, std::memory_order_release);
    window->in_progress_count.store(0, std::memory_order_release);
    window->start_time = startTime;
}

class TelemetryFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        telemetrytest::resetTelemetryExportMocks();
        resetTelemetryPrimerStateForTests();
        initializeTelemetryCommState(commState);
    }

    void TearDown() override
    {
        telemetrytest::resetTelemetryExportMocks();
        resetTelemetryPrimerStateForTests();
    }

    void setScaleUpExecMode(CommunicatorState::ScaleUpExecMode mode)
    {
        commState.scaleUpExecMode.store(static_cast<uint8_t>(mode), std::memory_order_release);
    }

    CommunicatorState commState;
};
}  // namespace testsupport

#endif  // TESTS_SUPPORT_TELEMETRY_TEST_FIXTURE_H_