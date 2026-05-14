// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../../linear_regression.h"

class LinearRegressionTest : public ::testing::Test
{
protected:
    LinearRegression lr;

    void SetUp() override
    {
        lr.clear();
    }
};

TEST_F(LinearRegressionTest, InitialState)
{
    double slope, intercept;
    EXPECT_FALSE(lr.calculate(slope, intercept));
    EXPECT_DOUBLE_EQ(slope, 0.0);
    EXPECT_DOUBLE_EQ(intercept, 0.0);
}

TEST_F(LinearRegressionTest, SimpleLinearRelationship)
{
    // Perfect linear relationship: time = 100 + size * 0.001
    // intercept = 100us, slope = 0.001 (rate = 1000 MB/s)
    lr.addPoint(0.0, 100.0);
    lr.addPoint(1000.0, 101.0);
    lr.addPoint(2000.0, 102.0);
    lr.addPoint(3000.0, 103.0);

    double slope, intercept;
    EXPECT_TRUE(lr.calculate(slope, intercept));
    EXPECT_NEAR(intercept, 100.0, 0.1);
    EXPECT_NEAR(slope, 0.001, 0.0001);
}

TEST_F(LinearRegressionTest, NoIntercept)
{
    // No latency: time = size * 0.001
    // intercept = 0us, slope = 0.001
    lr.addPoint(0.0, 0.0);
    lr.addPoint(1000.0, 1.0);
    lr.addPoint(2000.0, 2.0);
    lr.addPoint(3000.0, 3.0);

    double slope, intercept;
    EXPECT_TRUE(lr.calculate(slope, intercept));
    EXPECT_NEAR(intercept, 0.0, 0.1);
    EXPECT_NEAR(slope, 0.001, 0.0001);
}

TEST_F(LinearRegressionTest, InsufficientDataPoints)
{
    // Only 1 point - should be invalid
    lr.addPoint(1000.0, 100.0);

    double slope, intercept;
    EXPECT_FALSE(lr.calculate(slope, intercept));
}

TEST_F(LinearRegressionTest, EmptyData)
{
    // No points - should be invalid
    double slope, intercept;
    EXPECT_FALSE(lr.calculate(slope, intercept));
}

TEST_F(LinearRegressionTest, NoisyData)
{
    // Noisy data around: time = 50 + size * 0.002
    // intercept ~= 50us, slope ~= 0.002
    lr.addPoint(0.0, 52.0);
    lr.addPoint(1000.0, 51.5);
    lr.addPoint(2000.0, 54.2);
    lr.addPoint(3000.0, 56.1);
    lr.addPoint(4000.0, 58.3);

    double slope, intercept;
    EXPECT_TRUE(lr.calculate(slope, intercept));
    EXPECT_NEAR(intercept, 50.0, 5.0);  // Allow more tolerance for noisy data
    EXPECT_NEAR(slope, 0.002, 0.001);
}

TEST_F(LinearRegressionTest, IdenticalXValues)
{
    // All same X values - should fail (vertical line)
    lr.addPoint(1000.0, 10.0);
    lr.addPoint(1000.0, 11.0);
    lr.addPoint(1000.0, 12.0);

    double slope, intercept;
    EXPECT_FALSE(lr.calculate(slope, intercept));
}

TEST_F(LinearRegressionTest, NegativeValues)
{
    // Should handle negative values correctly
    lr.addPoint(-1000.0, -10.0);
    lr.addPoint(0.0, 0.0);
    lr.addPoint(1000.0, 10.0);
    lr.addPoint(2000.0, 20.0);

    double slope, intercept;
    EXPECT_TRUE(lr.calculate(slope, intercept));
    EXPECT_NEAR(slope, 0.01, 0.001);
    EXPECT_NEAR(intercept, 0.0, 1.0);
}

TEST_F(LinearRegressionTest, ClearResetsState)
{
    lr.addPoint(0.0, 0.0);
    lr.addPoint(1000.0, 10.0);

    double slope, intercept;
    EXPECT_TRUE(lr.calculate(slope, intercept));

    // Clear and verify reset
    lr.clear();
    EXPECT_FALSE(lr.calculate(slope, intercept));

    // Add new points and verify it works
    lr.addPoint(0.0, 100.0);
    lr.addPoint(1000.0, 200.0);
    EXPECT_TRUE(lr.calculate(slope, intercept));
}

TEST_F(LinearRegressionTest, LargeValues)
{
    // Test with large values (gigabytes)
    lr.addPoint(0.0, 1000.0);
    lr.addPoint(1000000000.0, 1001000.0);  // 1GB
    lr.addPoint(2000000000.0, 1002000.0);  // 2GB

    double slope, intercept;
    EXPECT_TRUE(lr.calculate(slope, intercept));
    EXPECT_GT(intercept, 0.0);
    EXPECT_GT(slope, 0.0);
}

TEST_F(LinearRegressionTest, VerySmallSlope)
{
    // Test with very small slope (very fast transfer)
    lr.addPoint(0.0, 10.0);
    lr.addPoint(1000000.0, 10.001);
    lr.addPoint(2000000.0, 10.002);

    double slope, intercept;
    EXPECT_TRUE(lr.calculate(slope, intercept));
    EXPECT_NEAR(intercept, 10.0, 0.1);
}

// MIN mode tests
class LinearRegressionMinModeTest : public ::testing::Test
{
protected:
    LinearRegression lr_min;

    void SetUp() override
    {
        lr_min = LinearRegression(LinearRegression::Mode::MIN);
        lr_min.clear();
    }
};

TEST_F(LinearRegressionMinModeTest, MinModeUsesMinimumTimePerSize)
{
    // Add multiple times for same size - should use minimum
    lr_min.addPoint(1000.0, 10.0);
    lr_min.addPoint(1000.0, 5.0);   // This should be used (minimum)
    lr_min.addPoint(1000.0, 15.0);  // This should be ignored

    // Add different size
    lr_min.addPoint(2000.0, 25.0);
    lr_min.addPoint(2000.0, 20.0);  // This should be used (minimum)

    double slope, intercept;
    EXPECT_TRUE(lr_min.calculate(slope, intercept));

    // Should fit: time = intercept + slope * size
    // With points: (1000, 5) and (2000, 20)
    // slope = (20-5)/(2000-1000) = 15/1000 = 0.015
    // intercept = 5 - 0.015*1000 = 5 - 15 = -10
    EXPECT_NEAR(slope, 0.015, 0.001);
    EXPECT_NEAR(intercept, -10.0, 0.1);
}

TEST_F(LinearRegressionMinModeTest, MinModeSingleSizeMultipleTimes)
{
    // All same size - should use minimum time
    lr_min.addPoint(1000.0, 100.0);
    lr_min.addPoint(1000.0, 50.0);  // minimum
    lr_min.addPoint(1000.0, 75.0);

    // Need at least 2 different sizes for regression
    lr_min.addPoint(2000.0, 60.0);

    double slope, intercept;
    EXPECT_TRUE(lr_min.calculate(slope, intercept));

    // Should use points: (1000, 50) and (2000, 60)
    // slope = (60-50)/(2000-1000) = 10/1000 = 0.01
    // intercept = 50 - 0.01*1000 = 50 - 10 = 40
    EXPECT_NEAR(slope, 0.01, 0.001);
    EXPECT_NEAR(intercept, 40.0, 0.1);
}

TEST_F(LinearRegressionMinModeTest, MinModeInsufficientData)
{
    // Only one unique size - should fail
    lr_min.addPoint(1000.0, 10.0);
    lr_min.addPoint(1000.0, 5.0);

    double slope, intercept;
    EXPECT_FALSE(lr_min.calculate(slope, intercept));
}

TEST_F(LinearRegressionMinModeTest, MinModeMerge)
{
    // First regression: size 1000 with times 10, 5 -> uses 5
    lr_min.addPoint(1000.0, 10.0);
    lr_min.addPoint(1000.0, 5.0);

    // Second regression: size 1000 with times 8, 12 -> uses 8
    // size 2000 with time 25
    LinearRegression lr_min2(LinearRegression::Mode::MIN);
    lr_min2.addPoint(1000.0, 8.0);
    lr_min2.addPoint(1000.0, 12.0);
    lr_min2.addPoint(2000.0, 25.0);

    // Merge them
    lr_min.merge(lr_min2);

    double slope, intercept;
    EXPECT_TRUE(lr_min.calculate(slope, intercept));

    // After merge: size 1000 should use min(5, 8) = 5
    // Points: (1000, 5) and (2000, 25)
    // slope = (25-5)/(2000-1000) = 20/1000 = 0.02
    // intercept = 5 - 0.02*1000 = 5 - 20 = -15
    EXPECT_NEAR(slope, 0.02, 0.001);
    EXPECT_NEAR(intercept, -15.0, 0.1);
}

TEST_F(LinearRegressionMinModeTest, MinModeVsAvgModeComparison)
{
    // Create AVG mode regression
    LinearRegression lr_avg(LinearRegression::Mode::AVG);

    // Same data for both
    lr_avg.addPoint(1000.0, 10.0);
    lr_avg.addPoint(1000.0, 5.0);  // AVG will use both
    lr_avg.addPoint(2000.0, 25.0);

    lr_min.addPoint(1000.0, 10.0);
    lr_min.addPoint(1000.0, 5.0);  // MIN will use only 5
    lr_min.addPoint(2000.0, 25.0);

    double slope_avg, intercept_avg;
    double slope_min, intercept_min;

    EXPECT_TRUE(lr_avg.calculate(slope_avg, intercept_avg));
    EXPECT_TRUE(lr_min.calculate(slope_min, intercept_min));

    // They should be different
    // AVG uses points: (1000, 10), (1000, 5), (2000, 25)
    // Average time at 1000: (10+5)/2 = 7.5
    // slope = (25-7.5)/(2000-1000) = 17.5/1000 = 0.0175
    // intercept = 7.5 - 0.0175*1000 = 7.5 - 17.5 = -10

    // MIN uses points: (1000, 5), (2000, 25)
    // slope = (25-5)/(2000-1000) = 20/1000 = 0.02
    // intercept = 5 - 0.02*1000 = 5 - 20 = -15

    EXPECT_NEAR(slope_avg, 0.0175, 0.001);
    EXPECT_NEAR(intercept_avg, -10.0, 0.1);
    EXPECT_NEAR(slope_min, 0.02, 0.001);
    EXPECT_NEAR(intercept_min, -15.0, 0.1);

    // Verify they are different
    EXPECT_NE(slope_avg, slope_min);
    EXPECT_NE(intercept_avg, intercept_min);
}

TEST_F(LinearRegressionMinModeTest, MinModeClearResetsState)
{
    lr_min.addPoint(1000.0, 10.0);
    lr_min.addPoint(1000.0, 5.0);
    lr_min.addPoint(2000.0, 25.0);

    double slope, intercept;
    EXPECT_TRUE(lr_min.calculate(slope, intercept));

    // Clear and verify reset
    lr_min.clear();
    EXPECT_FALSE(lr_min.calculate(slope, intercept));

    // Add new points and verify it works
    lr_min.addPoint(500.0, 5.0);
    lr_min.addPoint(1500.0, 15.0);
    EXPECT_TRUE(lr_min.calculate(slope, intercept));
}

// =============================================================================
// hasAtLeastThreeDifferentSizes
// =============================================================================

TEST_F(LinearRegressionTest, HasAtLeastThreeDifferentSizesEmpty)
{
    EXPECT_FALSE(lr.hasAtLeastThreeDifferentSizes());
}

TEST_F(LinearRegressionTest, HasAtLeastThreeDifferentSizesOnlyOne)
{
    lr.addPoint(100.0, 10.0);
    EXPECT_FALSE(lr.hasAtLeastThreeDifferentSizes());
}

TEST_F(LinearRegressionTest, HasAtLeastThreeDifferentSizesTwo)
{
    lr.addPoint(100.0, 10.0);
    lr.addPoint(200.0, 20.0);
    EXPECT_FALSE(lr.hasAtLeastThreeDifferentSizes());
}

TEST_F(LinearRegressionTest, HasAtLeastThreeDifferentSizesThree)
{
    lr.addPoint(100.0, 10.0);
    lr.addPoint(200.0, 20.0);
    lr.addPoint(300.0, 30.0);
    EXPECT_TRUE(lr.hasAtLeastThreeDifferentSizes());
}

TEST_F(LinearRegressionTest, HasAtLeastThreeDifferentSizesDuplicates)
{
    lr.addPoint(100.0, 10.0);
    lr.addPoint(100.0, 11.0);
    lr.addPoint(200.0, 20.0);
    lr.addPoint(200.0, 21.0);
    EXPECT_FALSE(lr.hasAtLeastThreeDifferentSizes());
}

TEST_F(LinearRegressionMinModeTest, MinModeHasAtLeastThreeDifferentSizes)
{
    lr_min.addPoint(100.0, 10.0);
    lr_min.addPoint(200.0, 20.0);
    EXPECT_FALSE(lr_min.hasAtLeastThreeDifferentSizes());

    lr_min.addPoint(300.0, 30.0);
    EXPECT_TRUE(lr_min.hasAtLeastThreeDifferentSizes());
}

// =============================================================================
// calculateRSquared
// =============================================================================

TEST_F(LinearRegressionTest, RSquaredPerfectFit)
{
    lr.addPoint(0.0, 100.0);
    lr.addPoint(1000.0, 200.0);
    lr.addPoint(2000.0, 300.0);
    lr.addPoint(3000.0, 400.0);

    double rSquared;
    EXPECT_TRUE(lr.calculateRSquared(rSquared));
    EXPECT_NEAR(rSquared, 1.0, 1e-10);
}

TEST_F(LinearRegressionTest, RSquaredNoisyData)
{
    lr.addPoint(0.0, 102.0);
    lr.addPoint(1000.0, 198.0);
    lr.addPoint(2000.0, 305.0);
    lr.addPoint(3000.0, 395.0);

    double rSquared;
    EXPECT_TRUE(lr.calculateRSquared(rSquared));
    EXPECT_GT(rSquared, 0.99);
    EXPECT_LE(rSquared, 1.0);
}

TEST_F(LinearRegressionTest, RSquaredInsufficientData)
{
    lr.addPoint(100.0, 10.0);

    double rSquared;
    EXPECT_FALSE(lr.calculateRSquared(rSquared));
    EXPECT_DOUBLE_EQ(rSquared, 0.0);
}

TEST_F(LinearRegressionTest, RSquaredNoData)
{
    double rSquared;
    EXPECT_FALSE(lr.calculateRSquared(rSquared));
}

TEST_F(LinearRegressionTest, RSquaredIdenticalXFails)
{
    lr.addPoint(100.0, 10.0);
    lr.addPoint(100.0, 20.0);
    lr.addPoint(100.0, 30.0);

    double rSquared;
    EXPECT_FALSE(lr.calculateRSquared(rSquared));
}

TEST_F(LinearRegressionTest, RSquaredConstantY)
{
    lr.addPoint(100.0, 50.0);
    lr.addPoint(200.0, 50.0);
    lr.addPoint(300.0, 50.0);

    double rSquared;
    EXPECT_TRUE(lr.calculateRSquared(rSquared));
    EXPECT_DOUBLE_EQ(rSquared, 1.0);
}

TEST_F(LinearRegressionMinModeTest, MinModeRSquaredPerfectFit)
{
    lr_min.addPoint(0.0, 100.0);
    lr_min.addPoint(1000.0, 200.0);
    lr_min.addPoint(2000.0, 300.0);

    double rSquared;
    EXPECT_TRUE(lr_min.calculateRSquared(rSquared));
    EXPECT_NEAR(rSquared, 1.0, 1e-10);
}
