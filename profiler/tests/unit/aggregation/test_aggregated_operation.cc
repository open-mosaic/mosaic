// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "../../../aggregation.h"

class AggregatedOperationBaseTest : public ::testing::Test
{
protected:
    AggregatedOperationBase operation;
};

TEST_F(AggregatedOperationBaseTest, InitialState)
{
    EXPECT_EQ(operation.totalBytes, 0u);
    EXPECT_EQ(operation.totalTimeUs, 0.0);
    EXPECT_EQ(operation.count, 0);
    EXPECT_EQ(operation.cachedTotalTransferCount, 0);
    EXPECT_EQ(operation.cachedTotalTransferBytes, 0u);
    EXPECT_DOUBLE_EQ(operation.cachedTotalTransferTimeUs, 0.0);
}

TEST_F(AggregatedOperationBaseTest, AddOperation)
{
    operation.addOperation(2048, 15.5);

    EXPECT_EQ(operation.totalBytes, 2048u);
    EXPECT_DOUBLE_EQ(operation.totalTimeUs, 15.5);
    EXPECT_EQ(operation.count, 1);
    EXPECT_DOUBLE_EQ(operation.getAverageSize(), 2048.0);
    EXPECT_DOUBLE_EQ(operation.getAverageTime(), 15.5);
}

TEST_F(AggregatedOperationBaseTest, MultipleOperations)
{
    operation.addOperation(1000, 10.0);
    operation.addOperation(2000, 20.0);
    operation.addOperation(3000, 30.0);

    EXPECT_EQ(operation.count, 3);
    EXPECT_EQ(operation.totalBytes, 6000u);
    EXPECT_DOUBLE_EQ(operation.totalTimeUs, 60.0);
    EXPECT_DOUBLE_EQ(operation.getAverageSize(), 2000.0);
    EXPECT_DOUBLE_EQ(operation.getAverageTime(), 20.0);
}

TEST_F(AggregatedOperationBaseTest, AddTransferToCache)
{
    operation.addTransferToCache(512, 5.0);
    operation.addTransferToCache(1024, 10.0);

    EXPECT_EQ(operation.cachedTotalTransferCount, 2);
    EXPECT_EQ(operation.cachedTotalTransferBytes, 1536u);
    EXPECT_DOUBLE_EQ(operation.cachedTotalTransferTimeUs, 15.0);
}

TEST_F(AggregatedOperationBaseTest, GetTotalTransferCount)
{
    EXPECT_EQ(operation.getTotalTransferCount(), 0);

    operation.addTransferToCache(100, 1.0);
    EXPECT_EQ(operation.getTotalTransferCount(), 1);

    operation.addTransferToCache(200, 2.0);
    EXPECT_EQ(operation.getTotalTransferCount(), 2);
}

TEST_F(AggregatedOperationBaseTest, GetAverageTransferCountWithNoOperations)
{
    operation.addTransferToCache(100, 1.0);
    EXPECT_DOUBLE_EQ(operation.getAverageTransferCount(), 0.0);
}

TEST_F(AggregatedOperationBaseTest, GetAverageTransferCountWithOperations)
{
    operation.addOperation(1000, 10.0);
    operation.addOperation(2000, 20.0);
    operation.addTransferToCache(100, 1.0);
    operation.addTransferToCache(200, 2.0);
    operation.addTransferToCache(300, 3.0);
    operation.addTransferToCache(400, 4.0);

    EXPECT_EQ(operation.count, 2);
    EXPECT_EQ(operation.cachedTotalTransferCount, 4);
    EXPECT_DOUBLE_EQ(operation.getAverageTransferCount(), 2.0);
}

TEST_F(AggregatedOperationBaseTest, GetAverageTransferSizeWithNoTransfers)
{
    EXPECT_DOUBLE_EQ(operation.getAverageTransferSize(), 0.0);
}

TEST_F(AggregatedOperationBaseTest, GetAverageTransferSizeWithTransfers)
{
    operation.addTransferToCache(100, 1.0);
    operation.addTransferToCache(200, 2.0);
    operation.addTransferToCache(300, 3.0);

    double avgSize = (100.0 + 200.0 + 300.0) / 3.0;
    EXPECT_DOUBLE_EQ(operation.getAverageTransferSize(), avgSize);
}

TEST_F(AggregatedOperationBaseTest, GetAverageTransferTimeWithNoTransfers)
{
    EXPECT_DOUBLE_EQ(operation.getAverageTransferTime(), 0.0);
}

TEST_F(AggregatedOperationBaseTest, GetAverageTransferTimeWithTransfers)
{
    operation.addTransferToCache(100, 1.0);
    operation.addTransferToCache(200, 3.0);
    operation.addTransferToCache(300, 5.0);

    double avgTime = (1.0 + 3.0 + 5.0) / 3.0;
    EXPECT_DOUBLE_EQ(operation.getAverageTransferTime(), avgTime);
}

class AggregatedP2PTest : public ::testing::Test
{
protected:
    AggregatedP2P p2p;
};

TEST_F(AggregatedP2PTest, InheritsFromBase)
{
    p2p.addOperation(1000, 10.0);
    EXPECT_EQ(p2p.count, 1);
    EXPECT_EQ(p2p.totalBytes, 1000u);
}

TEST_F(AggregatedP2PTest, ConvenienceMethodDelegates)
{
    p2p.addP2P(2000, 20.0);
    EXPECT_EQ(p2p.count, 1);
    EXPECT_EQ(p2p.totalBytes, 2000u);
    EXPECT_DOUBLE_EQ(p2p.totalTimeUs, 20.0);
}

class AggregatedCollectiveTest : public ::testing::Test
{
protected:
    AggregatedCollective collective;
};

TEST_F(AggregatedCollectiveTest, InheritsFromBase)
{
    collective.addOperation(3000, 30.0);
    EXPECT_EQ(collective.count, 1);
    EXPECT_EQ(collective.totalBytes, 3000u);
}

TEST_F(AggregatedCollectiveTest, ConvenienceMethodDelegates)
{
    collective.addCollective(4000, 40.0);
    EXPECT_EQ(collective.count, 1);
    EXPECT_EQ(collective.totalBytes, 4000u);
    EXPECT_DOUBLE_EQ(collective.totalTimeUs, 40.0);
}