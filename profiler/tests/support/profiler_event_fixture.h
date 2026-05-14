// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef TESTS_SUPPORT_PROFILER_EVENT_FIXTURE_H_
#define TESTS_SUPPORT_PROFILER_EVENT_FIXTURE_H_

#include <gtest/gtest.h>

#include "../../communicator_state.h"
#include "../../events.h"
#include "../../profiler_otel.h"
#include "../test_helpers.h"
#include "event_descr_builders.h"

inline void mock_profiler_event_logger(ncclDebugLogLevel level, unsigned long flags, const char* file, int line,
                                       const char* fmt, ...)
{
    (void)level;
    (void)flags;
    (void)file;
    (void)line;
    (void)fmt;
}

class ProfilerEventTest : public ::testing::Test
{
protected:
    void* context;
    int eActivationMask;

    void SetUp() override
    {
        context         = nullptr;
        eActivationMask = 0;

        resetProfilerState();

        ncclResult_t result =
            profiler_otel_init_v5(&context, 12345, &eActivationMask, "test_comm", 2, 4, 0, mock_profiler_event_logger);
        ASSERT_EQ(result, ncclSuccess);
        ASSERT_NE(context, nullptr);
    }

    void TearDown() override
    {
        if (context)
        {
            profiler_otel_finalize_v5(context);
            context = nullptr;
        }
    }
};

#endif  // TESTS_SUPPORT_PROFILER_EVENT_FIXTURE_H_