// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef LINEAR_REGRESSION_H_
#define LINEAR_REGRESSION_H_

#include <map>
#include <set>
#include <string>
#include <utility>  // For std::pair

/**
 * Simple linear regression class for computing latency from transfer data.
 *
 * Given pairs of (size, time), computes:
 * - Slope: transfer time per byte (us/byte)
 * - Intercept: latency at size=0 (in us) - represents fixed overhead/latency
 *
 * Primary use case: The intercept from linear regression is used to estimate the
 * fixed latency component in rank-to-rank transfers. Rate/bandwidth is now computed
 * using interval-based active time calculation instead of 1/slope.
 *
 * Supports two modes:
 * - AVG: Use all transfer data points for regression
 * - MIN: Use minimum transfer time for each unique transfer size
 *
 * NOTE: This class is NOT thread-safe. Each instance should be used by a single thread.
 */
class LinearRegression
{
public:
    enum class Mode
    {
        AVG,  // Use all data points
        MIN   // Use minimum time per size
    };

    LinearRegression(Mode mode = Mode::AVG);

    void addPoint(double x, double y);

    void merge(const LinearRegression& other);

    void clear();

    bool calculate(double& slope, double& intercept) const;

    bool hasAtLeastThreeDifferentSizes() const;

    bool calculateRSquared(double& rSquared) const;

private:
    Mode mode_;
    std::set<double> uniqueSizes;
    std::map<double, double> minTimesPerSize;  // For MIN mode: size -> min time
    double sumX, sumY, sumXY, sumX2, sumY2;
    int n;
};

#endif  // LINEAR_REGRESSION_H_
