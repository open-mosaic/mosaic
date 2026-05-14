// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "telemetry_internal.h"

#ifdef ENABLE_OTEL

/**
 * @brief Retrieve the values of the metrics which will be exported for the standard export
 * of a collective operation.
 *
 * @param[in] coll Aggregated collective data containing statistics.
 *
 * @return CollectiveEmitView containing the values of the metrics to export.
 */
CollectiveEmitView makeStandardCollectiveEmitView(const AggregatedCollective& coll)
{
    return CollectiveEmitView{static_cast<double>(coll.count),
                              static_cast<double>(coll.totalBytes),
                              coll.totalTimeUs,
                              coll.getAverageSize(),
                              coll.getAverageTime(),
                              coll.getAverageTransferCount(),
                              coll.getAverageTransferSize(),
                              coll.getAverageTransferTime()};
}

/**
 * @brief Compute the eligibility of the metrics to export for the standard export
 * of a collective operation.
 *
 * @param[in] op Aggregated collective data containing statistics.
 *
 * @return CollectiveExportEligibility containing the eligibility of the metrics to export.
 */
CollectiveExportEligibility computeCollectiveEligibility(const AggregatedCollective& op)
{
    return {op.count > 0, op.getAverageTransferCount() > 0.0, op.cachedTotalTransferTimeUs > 0.0};
}

/**
 * @brief Retrieve the values of the metrics which will be exported for the standard export
 * of a P2P operation.
 *
 * @param[in] p2p Aggregated P2P data containing statistics.
 *
 * @return P2PEmitView containing the values of the metrics to export.
 */
P2PEmitView makeStandardP2PEmitView(const AggregatedP2P& p2p)
{
    return P2PEmitView{p2p.getAverageSize(), p2p.getAverageTime(), p2p.getAverageTransferCount(),
                       p2p.getAverageTransferSize(), p2p.getAverageTransferTime()};
}

/**
 * @brief Compute the eligibility of the metrics to export for the standard export
 * of a P2P operation.
 *
 * @param[in] op Aggregated P2P data containing statistics.
 *
 * @return P2PExportEligibility containing the eligibility of the metrics to export.
 */
P2PExportEligibility computeP2PEligibility(const AggregatedP2P& op)
{
    return {op.count > 0, op.getAverageTransferCount() > 0.0, op.cachedTotalTransferTimeUs > 0.0};
}

/**
 * @brief Retrieve the values of the metrics which will be exported for the standard export
 * of a rank transfer operation.
 *
 * @param[in] t Aggregated rank transfer data containing statistics.
 *
 * @return RankEmitView containing the values of the metrics to export.
 */
RankEmitView makeStandardRankEmitView(const AggregatedTransfer& t)
{
    double latencyUs = 0.0;
    (void)t.getLatencyFromLinearRegression(latencyUs);
    double rateMBps = 0.0;
    (void)t.getRateFromActiveTime(rateMBps);
    return RankEmitView{static_cast<uint64_t>(t.totalBytes), latencyUs, rateMBps, t.getActiveTime()};
}

/**
 * @brief Compute the eligibility of the metrics to export for the standard export
 * of a rank transfer operation.
 *
 * @param[in] op Aggregated rank transfer data containing statistics.
 *
 * @return RankExportEligibility containing the eligibility of the metrics to export.
 */
RankExportEligibility computeRankEligibility(const AggregatedTransfer& op)
{
    double scratch = 0.0;
    return RankExportEligibility{
        op.getLatencyFromLinearRegression(scratch),
        op.getRateFromActiveTime(scratch),
    };
}

/**
 * @brief Retrieve the values of the metrics which will be exported for the standard export
 * of a transfer operation.
 *
 * @param[in] t Aggregated transfer data containing statistics.
 *
 * @return TransferEmitView containing the values of the metrics to export.
 */
TransferEmitView makeStandardTransferEmitView(const AggregatedTransfer& t)
{
    double latencyUs = 0.0;
    (void)t.getLatencyFromLinearRegression(latencyUs);
    return TransferEmitView{t.getAverageSize(), t.getAverageTime(), latencyUs};
}

/**
 * @brief Compute the eligibility of the metrics to export for the standard export
 * of a transfer operation.
 *
 * @param[in] op Aggregated transfer data containing statistics.
 *
 * @return TransferExportEligibility containing the eligibility of the metrics to export.
 */
TransferExportEligibility computeTransferEligibility(const AggregatedTransfer& op)
{
    double scratch = 0.0;
    return TransferExportEligibility{
        op.count > 0,
        op.totalTimeUs > 0.0,
        op.getLatencyFromLinearRegression(scratch),
    };
}

#endif  // ENABLE_OTEL