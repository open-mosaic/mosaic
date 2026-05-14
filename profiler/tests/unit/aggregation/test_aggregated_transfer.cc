// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "../../../aggregation.h"

class AggregatedTransferTest : public ::testing::Test
{
protected:
    AggregatedTransfer transfer;
};

TEST_F(AggregatedTransferTest, InitialState)
{
    EXPECT_EQ(transfer.totalBytes, 0u);
    EXPECT_EQ(transfer.totalTimeUs, 0.0);
    EXPECT_EQ(transfer.count, 0);
}

TEST_F(AggregatedTransferTest, SingleTransfer)
{
    transfer.addTransfer(1024, 10.5);

    EXPECT_EQ(transfer.totalBytes, 1024u);
    EXPECT_DOUBLE_EQ(transfer.totalTimeUs, 10.5);
    EXPECT_EQ(transfer.count, 1);
    EXPECT_DOUBLE_EQ(transfer.getAverageSize(), 1024.0);
    EXPECT_DOUBLE_EQ(transfer.getAverageTime(), 10.5);
    EXPECT_DOUBLE_EQ(transfer.getAverageRateMBps(), 1024.0 / 10.5);
}

TEST_F(AggregatedTransferTest, MultipleTransfers)
{
    transfer.addTransfer(1000, 10.0);
    transfer.addTransfer(2000, 20.0);
    transfer.addTransfer(3000, 30.0);

    EXPECT_EQ(transfer.totalBytes, 6000u);
    EXPECT_DOUBLE_EQ(transfer.totalTimeUs, 60.0);
    EXPECT_EQ(transfer.count, 3);
    EXPECT_DOUBLE_EQ(transfer.getAverageSize(), 2000.0);
    EXPECT_DOUBLE_EQ(transfer.getAverageTime(), 20.0);
}

TEST_F(AggregatedTransferTest, ZeroBytes)
{
    transfer.addTransfer(0, 10.0);

    EXPECT_EQ(transfer.totalBytes, 0u);
    EXPECT_DOUBLE_EQ(transfer.totalTimeUs, 10.0);
    EXPECT_EQ(transfer.count, 1);
    EXPECT_DOUBLE_EQ(transfer.getAverageSize(), 0.0);
}

TEST_F(AggregatedTransferTest, ZeroTime)
{
    transfer.addTransfer(1000, 0.0);

    EXPECT_EQ(transfer.totalBytes, 1000u);
    EXPECT_DOUBLE_EQ(transfer.totalTimeUs, 0.0);
    EXPECT_DOUBLE_EQ(transfer.getAverageRateMBps(), 0.0);
}

TEST_F(AggregatedTransferTest, LinearRegressionWithTwoPoints)
{
    transfer.addTransfer(0, 10.0);
    transfer.addTransfer(1000, 11.0);

    double latency;
    EXPECT_FALSE(transfer.getLatencyFromLinearRegression(latency));
    EXPECT_DOUBLE_EQ(latency, 0.0);
}

TEST_F(AggregatedTransferTest, LinearRegressionWithOnePoint)
{
    transfer.addTransfer(1000, 10.0);

    double latency;
    EXPECT_FALSE(transfer.getLatencyFromLinearRegression(latency));
    EXPECT_DOUBLE_EQ(latency, 0.0);
}

TEST_F(AggregatedTransferTest, LinearRegressionWithNoPoints)
{
    double latency;
    EXPECT_FALSE(transfer.getLatencyFromLinearRegression(latency));
}

TEST_F(AggregatedTransferTest, LinearRegressionWithIdenticalSizes)
{
    transfer.addTransfer(1000, 10.0);
    transfer.addTransfer(1000, 11.0);
    transfer.addTransfer(1000, 12.0);

    double latency;
    EXPECT_FALSE(transfer.getLatencyFromLinearRegression(latency));
}

TEST_F(AggregatedTransferTest, LinearRegressionWithZeroSlope)
{
    transfer.addTransfer(0, 10.0);
    transfer.addTransfer(1000, 10.0);
    transfer.addTransfer(2000, 10.0);
    transfer.addTransfer(3000, 10.0);

    double latency;
    EXPECT_FALSE(transfer.getLatencyFromLinearRegression(latency));
    EXPECT_DOUBLE_EQ(latency, 0.0);
}

TEST_F(AggregatedTransferTest, LatencyFromLinearRegressionBasic)
{
    transfer.addTransfer(0, 10.0);
    transfer.addTransfer(1000, 20.0);
    transfer.addTransfer(2000, 30.0);
    transfer.addTransfer(5000, 60.0);

    double latency;
    EXPECT_TRUE(transfer.getLatencyFromLinearRegression(latency));
    EXPECT_NEAR(latency, 10.0, 0.5);
}

TEST_F(AggregatedTransferTest, LatencyFromLinearRegressionInsufficientData)
{
    transfer.addTransfer(0, 10.0);
    transfer.addTransfer(1000, 20.0);

    double latency;
    EXPECT_FALSE(transfer.getLatencyFromLinearRegression(latency));
    EXPECT_DOUBLE_EQ(latency, 0.0);
}

class AggregatedTransferIntervalTest : public ::testing::Test
{
protected:
    AggregatedTransfer transfer;
};

TEST_F(AggregatedTransferIntervalTest, EmptyIntervalsReturnsZeroActiveTime)
{
    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 0.0);

    double rate;
    EXPECT_FALSE(transfer.getRateFromActiveTime(rate));
    EXPECT_DOUBLE_EQ(rate, 0.0);
}

TEST_F(AggregatedTransferIntervalTest, SingleIntervalActiveTime)
{
    transfer.addTransferWithTimestamps(1000, 10.0, 10.0, 20.0);

    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 10.0);
    EXPECT_EQ(transfer.count, 1);
    EXPECT_EQ(transfer.totalBytes, 1000u);

    double rate;
    EXPECT_TRUE(transfer.getRateFromActiveTime(rate));
    EXPECT_DOUBLE_EQ(rate, 100.0);
}

TEST_F(AggregatedTransferIntervalTest, NonOverlappingIntervals)
{
    transfer.addTransferWithTimestamps(1000, 10.0, 0.0, 10.0);
    transfer.addTransferWithTimestamps(2000, 10.0, 20.0, 30.0);

    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 20.0);
    EXPECT_EQ(transfer.totalBytes, 3000u);

    double rate;
    EXPECT_TRUE(transfer.getRateFromActiveTime(rate));
    EXPECT_DOUBLE_EQ(rate, 150.0);
}

TEST_F(AggregatedTransferIntervalTest, OverlappingIntervalsSimple)
{
    transfer.addTransferWithTimestamps(1000, 20.0, 0.0, 20.0);
    transfer.addTransferWithTimestamps(2000, 20.0, 10.0, 30.0);

    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 30.0);
    EXPECT_EQ(transfer.totalBytes, 3000u);

    double rate;
    EXPECT_TRUE(transfer.getRateFromActiveTime(rate));
    EXPECT_DOUBLE_EQ(rate, 100.0);
}

TEST_F(AggregatedTransferIntervalTest, FullyContainedInterval)
{
    transfer.addTransferWithTimestamps(5000, 100.0, 0.0, 100.0);
    transfer.addTransferWithTimestamps(2000, 30.0, 20.0, 50.0);

    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 100.0);
    EXPECT_EQ(transfer.totalBytes, 7000u);

    double rate;
    EXPECT_TRUE(transfer.getRateFromActiveTime(rate));
    EXPECT_DOUBLE_EQ(rate, 70.0);
}

TEST_F(AggregatedTransferIntervalTest, ComplexOverlappingScenario)
{
    transfer.addTransferWithTimestamps(1000, 30.0, 0.0, 30.0);
    transfer.addTransferWithTimestamps(2000, 40.0, 10.0, 50.0);
    transfer.addTransferWithTimestamps(1500, 20.0, 60.0, 80.0);

    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 70.0);
    EXPECT_EQ(transfer.totalBytes, 4500u);

    double rate;
    EXPECT_TRUE(transfer.getRateFromActiveTime(rate));
    EXPECT_NEAR(rate, 64.2857, 0.001);
}

TEST_F(AggregatedTransferIntervalTest, AdjacentIntervals)
{
    transfer.addTransferWithTimestamps(1000, 10.0, 0.0, 10.0);
    transfer.addTransferWithTimestamps(1000, 10.0, 10.0, 20.0);

    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 20.0);

    double rate;
    EXPECT_TRUE(transfer.getRateFromActiveTime(rate));
    EXPECT_DOUBLE_EQ(rate, 100.0);
}

TEST_F(AggregatedTransferIntervalTest, ManyOverlappingIntervals)
{
    for (int i = 0; i < 10; i++)
    {
        double start = i * 2.0;
        double end   = start + 5.0;
        transfer.addTransferWithTimestamps(100, 5.0, start, end);
    }

    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 23.0);
    EXPECT_EQ(transfer.totalBytes, 1000u);

    double rate;
    EXPECT_TRUE(transfer.getRateFromActiveTime(rate));
    EXPECT_NEAR(rate, 43.478, 0.01);
}

TEST_F(AggregatedTransferIntervalTest, UnorderedIntervalAddition)
{
    transfer.addTransferWithTimestamps(1000, 10.0, 50.0, 60.0);
    transfer.addTransferWithTimestamps(1000, 10.0, 0.0, 10.0);
    transfer.addTransferWithTimestamps(1000, 10.0, 20.0, 35.0);
    transfer.addTransferWithTimestamps(1000, 10.0, 30.0, 55.0);

    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 50.0);
}

TEST_F(AggregatedTransferIntervalTest, ZeroDurationInterval)
{
    transfer.addTransferWithTimestamps(1000, 0.0, 10.0, 10.0);

    EXPECT_TRUE(transfer.intervals.empty());
    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 0.0);
    EXPECT_EQ(transfer.totalBytes, 1000u);
    EXPECT_EQ(transfer.count, 1);
}

TEST_F(AggregatedTransferIntervalTest, NegativeDurationInterval)
{
    transfer.addTransferWithTimestamps(1000, -5.0, 20.0, 10.0);

    EXPECT_TRUE(transfer.intervals.empty());
    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 0.0);
}

TEST_F(AggregatedTransferIntervalTest, MergeIntervalsFromAnotherTransfer)
{
    transfer.addTransferWithTimestamps(1000, 10.0, 0.0, 10.0);
    transfer.addTransferWithTimestamps(1000, 10.0, 20.0, 30.0);

    AggregatedTransfer other;
    other.addTransferWithTimestamps(500, 5.0, 5.0, 15.0);
    other.addTransferWithTimestamps(500, 5.0, 40.0, 45.0);

    transfer.mergeIntervals(other);

    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 30.0);
    EXPECT_EQ(transfer.totalBytes, 2000u);
}

TEST_F(AggregatedTransferIntervalTest, RateCalculationWithZeroBytes)
{
    transfer.intervals.push_back({0.0, 10.0});

    double rate;
    EXPECT_FALSE(transfer.getRateFromActiveTime(rate));
    EXPECT_DOUBLE_EQ(rate, 0.0);
}

TEST_F(AggregatedTransferIntervalTest, LatencyStillWorksWithIntervals)
{
    transfer.addTransferWithTimestamps(0, 10.0, 0.0, 10.0);
    transfer.addTransferWithTimestamps(1000, 20.0, 10.0, 30.0);
    transfer.addTransferWithTimestamps(2000, 30.0, 30.0, 60.0);
    transfer.addTransferWithTimestamps(5000, 60.0, 60.0, 120.0);

    double latency;
    bool hasLatency = transfer.getLatencyFromLinearRegression(latency);
    EXPECT_TRUE(hasLatency);
    EXPECT_NEAR(latency, 10.0, 0.5);

    EXPECT_DOUBLE_EQ(transfer.getActiveTime(), 120.0);

    double rate;
    EXPECT_TRUE(transfer.getRateFromActiveTime(rate));
    EXPECT_NEAR(rate, 66.67, 0.1);
}