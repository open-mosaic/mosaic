// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "../../profiler_otel.h"

// Test fixture for profiler utility functions
class ProfilerUtilsTest : public ::testing::Test
{
};

// =============================================================================
// Tests for ncclTypeSize()
// =============================================================================

TEST_F(ProfilerUtilsTest, NcclTypeSizeInt8)
{
    EXPECT_EQ(test_ncclTypeSize("ncclInt8"), 1u);
}

TEST_F(ProfilerUtilsTest, NcclTypeSizeUint8)
{
    EXPECT_EQ(test_ncclTypeSize("ncclUint8"), 1u);
}

TEST_F(ProfilerUtilsTest, NcclTypeSizeFloat16)
{
    EXPECT_EQ(test_ncclTypeSize("ncclFloat16"), 2u);
}

TEST_F(ProfilerUtilsTest, NcclTypeSizeBfloat16)
{
    EXPECT_EQ(test_ncclTypeSize("ncclBfloat16"), 2u);
}

TEST_F(ProfilerUtilsTest, NcclTypeSizeInt32)
{
    EXPECT_EQ(test_ncclTypeSize("ncclInt32"), 4u);
}

TEST_F(ProfilerUtilsTest, NcclTypeSizeUint32)
{
    EXPECT_EQ(test_ncclTypeSize("ncclUint32"), 4u);
}

TEST_F(ProfilerUtilsTest, NcclTypeSizeFloat32)
{
    EXPECT_EQ(test_ncclTypeSize("ncclFloat32"), 4u);
}

TEST_F(ProfilerUtilsTest, NcclTypeSizeInt64)
{
    EXPECT_EQ(test_ncclTypeSize("ncclInt64"), 8u);
}

TEST_F(ProfilerUtilsTest, NcclTypeSizeUint64)
{
    EXPECT_EQ(test_ncclTypeSize("ncclUint64"), 8u);
}

TEST_F(ProfilerUtilsTest, NcclTypeSizeFloat64)
{
    EXPECT_EQ(test_ncclTypeSize("ncclFloat64"), 8u);
}

TEST_F(ProfilerUtilsTest, NcclTypeSizeUnknown)
{
    EXPECT_EQ(test_ncclTypeSize("unknown"), 0u);
    EXPECT_EQ(test_ncclTypeSize("invalid"), 0u);
    EXPECT_EQ(test_ncclTypeSize("ncclFoo"), 0u);
}

TEST_F(ProfilerUtilsTest, NcclTypeSizeNull)
{
    EXPECT_EQ(test_ncclTypeSize(nullptr), 0u);
}

TEST_F(ProfilerUtilsTest, NcclTypeSizeEmptyString)
{
    EXPECT_EQ(test_ncclTypeSize(""), 0u);
}

TEST_F(ProfilerUtilsTest, NcclTypeSizeCaseSensitive)
{
    // Should be case-sensitive - these should return 0
    EXPECT_EQ(test_ncclTypeSize("NCCLINT32"), 0u);
    EXPECT_EQ(test_ncclTypeSize("NcclInt32"), 0u);
    EXPECT_EQ(test_ncclTypeSize("ncclint32"), 0u);
}

// =============================================================================
// Tests for gpuUuidToString()
// =============================================================================

TEST_F(ProfilerUtilsTest, GpuUuidToStringBasic)
{
    // UUID with all zeros
    unsigned char uuid_bytes[16] = {0};
    std::string result           = test_gpuUuidToString(uuid_bytes);
    EXPECT_EQ(result, "00000000-0000-0000-0000-000000000000");
}

TEST_F(ProfilerUtilsTest, GpuUuidToStringAllOnes)
{
    // UUID with all 0xFF bytes
    unsigned char uuid_bytes[16];
    memset(uuid_bytes, 0xFF, 16);
    std::string result = test_gpuUuidToString(uuid_bytes);
    EXPECT_EQ(result, "ffffffff-ffff-ffff-ffff-ffffffffffff");
}

TEST_F(ProfilerUtilsTest, GpuUuidToStringSequential)
{
    // UUID with sequential bytes 0x00 to 0x0F
    unsigned char uuid_bytes[16];
    for (int i = 0; i < 16; i++)
    {
        uuid_bytes[i] = (unsigned char)i;
    }
    std::string result = test_gpuUuidToString(uuid_bytes);
    EXPECT_EQ(result, "00010203-0405-0607-0809-0a0b0c0d0e0f");
}

TEST_F(ProfilerUtilsTest, GpuUuidToStringRealWorld)
{
    // A realistic UUID pattern
    unsigned char uuid_bytes[16] = {0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0,
                                    0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
    std::string result           = test_gpuUuidToString(uuid_bytes);
    EXPECT_EQ(result, "12345678-9abc-def0-1122-334455667788");
}

TEST_F(ProfilerUtilsTest, GpuUuidToStringMixedCase)
{
    // Test that hex values are lowercase
    unsigned char uuid_bytes[16] = {0xAB, 0xCD, 0xEF, 0x00, 0x00, 0x00, 0x00, 0x00,
                                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    std::string result           = test_gpuUuidToString(uuid_bytes);
    // Should produce lowercase hex
    EXPECT_EQ(result, "abcdef00-0000-0000-0000-000000000000");
}

TEST_F(ProfilerUtilsTest, GpuUuidToStringFormat)
{
    // Verify the format is correct (8-4-4-4-12)
    unsigned char uuid_bytes[16] = {0};
    std::string result           = test_gpuUuidToString(uuid_bytes);

    // Check the length (36 characters: 32 hex + 4 dashes)
    EXPECT_EQ(result.length(), 36u);

    // Check dash positions
    EXPECT_EQ(result[8], '-');
    EXPECT_EQ(result[13], '-');
    EXPECT_EQ(result[18], '-');
    EXPECT_EQ(result[23], '-');
}
