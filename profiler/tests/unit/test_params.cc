// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "../../param.h"

// Test fixture for parameter loading
class ParamTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Clear any test environment variables
        unsetenv("NCCL_TEST_INT_PARAM");
        unsetenv("NCCL_TEST_STRING_PARAM");
    }

    void TearDown() override
    {
        // Clean up test environment variables
        unsetenv("NCCL_TEST_INT_PARAM");
        unsetenv("NCCL_TEST_STRING_PARAM");
    }
};

// Define test parameters using OTEL_PARAM and OTEL_STRING_PARAM macros
OTEL_PARAM(TestIntParam, "TEST_INT_PARAM", 42);
OTEL_STRING_PARAM(TestStringParam, "TEST_STRING_PARAM", "default_value");

// =============================================================================
// Tests for Integer Parameter Loading (OTEL_PARAM)
// =============================================================================

TEST_F(ParamTest, IntParamDefaultValue)
{
    // No environment variable set - should return default
    EXPECT_EQ(OTEL_GET_PARAM(TestIntParam), 42);
}

TEST_F(ParamTest, IntParamFromEnvironment)
{
    setenv("NCCL_TEST_INT_PARAM", "123", 1);
    EXPECT_EQ(OTEL_GET_PARAM(TestIntParam), 123);
}

TEST_F(ParamTest, IntParamZeroValue)
{
    setenv("NCCL_TEST_INT_PARAM", "0", 1);
    EXPECT_EQ(OTEL_GET_PARAM(TestIntParam), 0);
}

TEST_F(ParamTest, IntParamNegativeValue)
{
    setenv("NCCL_TEST_INT_PARAM", "-100", 1);
    EXPECT_EQ(OTEL_GET_PARAM(TestIntParam), -100);
}

TEST_F(ParamTest, IntParamHexValue)
{
    // strtoll supports hex with 0x prefix
    setenv("NCCL_TEST_INT_PARAM", "0x1E", 1);
    EXPECT_EQ(OTEL_GET_PARAM(TestIntParam), 0x1E);
}

TEST_F(ParamTest, IntParamOctalValue)
{
    // strtoll supports octal with leading 0
    setenv("NCCL_TEST_INT_PARAM", "010", 1);
    EXPECT_EQ(OTEL_GET_PARAM(TestIntParam), 8);  // Octal 010 = decimal 8
}

TEST_F(ParamTest, IntParamEmptyString)
{
    // Empty string should fall back to default
    setenv("NCCL_TEST_INT_PARAM", "", 1);
    EXPECT_EQ(OTEL_GET_PARAM(TestIntParam), 42);
}

TEST_F(ParamTest, IntParamInvalidString)
{
    // Invalid string should fall back to default
    setenv("NCCL_TEST_INT_PARAM", "not_a_number", 1);
    EXPECT_EQ(OTEL_GET_PARAM(TestIntParam), 42);
}

TEST_F(ParamTest, IntParamPartiallyValidString)
{
    // String starting with number but with trailing chars should fall back to default
    setenv("NCCL_TEST_INT_PARAM", "123abc", 1);
    EXPECT_EQ(OTEL_GET_PARAM(TestIntParam), 42);
}

TEST_F(ParamTest, IntParamWithWhitespace)
{
    // Leading/trailing whitespace should be handled
    setenv("NCCL_TEST_INT_PARAM", " 99 ", 1);
    // strtoll will parse "99" from " 99 " (leading whitespace is skipped)
    // but trailing " " makes *endptr != '\0', so it should fall back to default
    EXPECT_EQ(OTEL_GET_PARAM(TestIntParam), 42);
}

TEST_F(ParamTest, IntParamLargeValue)
{
    // Test with large value
    setenv("NCCL_TEST_INT_PARAM", "9223372036854775807", 1);  // INT64_MAX
    EXPECT_EQ(OTEL_GET_PARAM(TestIntParam), INT64_MAX);
}

// =============================================================================
// Tests for String Parameter Loading (OTEL_STRING_PARAM)
// =============================================================================

TEST_F(ParamTest, StringParamDefaultValue)
{
    // No environment variable set - should return default
    const char* result = ncclParamTestStringParam();
    EXPECT_STREQ(result, "default_value");
}

TEST_F(ParamTest, StringParamFromEnvironment)
{
    setenv("NCCL_TEST_STRING_PARAM", "custom_value", 1);
    const char* result = ncclParamTestStringParam();
    EXPECT_STREQ(result, "custom_value");
}

TEST_F(ParamTest, StringParamEmptyString)
{
    // Empty string should fall back to default
    setenv("NCCL_TEST_STRING_PARAM", "", 1);
    const char* result = ncclParamTestStringParam();
    EXPECT_STREQ(result, "default_value");
}

TEST_F(ParamTest, StringParamWithSpaces)
{
    setenv("NCCL_TEST_STRING_PARAM", "value with spaces", 1);
    const char* result = ncclParamTestStringParam();
    EXPECT_STREQ(result, "value with spaces");
}

TEST_F(ParamTest, StringParamWithSpecialChars)
{
    setenv("NCCL_TEST_STRING_PARAM", "http://localhost:4318", 1);
    const char* result = ncclParamTestStringParam();
    EXPECT_STREQ(result, "http://localhost:4318");
}

// =============================================================================
// Tests for Real Plugin Parameters (verify they exist and have reasonable defaults)
// =============================================================================

// These declarations are needed to access the actual plugin parameters
// They are defined in profiler_otel.cc and telemetry.cc
extern int64_t ncclParamEnableOTEL();
extern int64_t ncclParamProfileEventMask();
extern int64_t ncclParamWindowTimeoutIntervalSec();
extern const char* ncclParamLinearRegressionMode();

TEST_F(ParamTest, RealParamEnableOTELDefault)
{
    unsetenv("NCCL_PROFILER_OTEL_ENABLE");
    // Default should be 1 (enabled)
    EXPECT_EQ(ncclParamEnableOTEL(), 1);
}

TEST_F(ParamTest, RealParamProfileEventMaskDefault)
{
    unsetenv("NCCL_PROFILE_EVENT_MASK");
    // Default should be -1 (use internal default)
    EXPECT_EQ(ncclParamProfileEventMask(), -1);
}

TEST_F(ParamTest, RealParamWindowTimeoutDefault)
{
    unsetenv("NCCL_PROFILER_OTEL_TELEMETRY_INTERVAL_SEC");
    // Default should be 5 seconds
    EXPECT_EQ(ncclParamWindowTimeoutIntervalSec(), 5);
}

TEST_F(ParamTest, RealParamLinearRegressionModeDefault)
{
    unsetenv("NCCL_PROFILER_LINEAR_REGRESSION_MODE");
    // Default should be "AVG"
    EXPECT_STREQ(ncclParamLinearRegressionMode(), "AVG");
}
