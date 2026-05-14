// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef TESTS_MOCKS_TELEMETRY_EXPORT_MOCKS_H_
#define TESTS_MOCKS_TELEMETRY_EXPORT_MOCKS_H_

#include <string>
#include <vector>

#include "../../telemetry_internal.h"

namespace telemetrytest
{
struct CollectiveExportRecord
{
    std::string key;
    CollectiveEmitView emit;
    CollectiveExportEligibility eligibility;
    int rank;
    std::string scaleUpExecMode;
    std::string exportTag;
};

struct P2PExportRecord
{
    std::string key;
    P2PEmitView emit;
    P2PExportEligibility eligibility;
    int rank;
    std::string scaleUpExecMode;
    std::string exportTag;
};

struct RankExportRecord
{
    std::string key;
    RankEmitView emit;
    RankExportEligibility eligibility;
    int rank;
    std::string scaleUpExecMode;
    std::string exportTag;
};

struct TransferExportRecord
{
    std::string key;
    TransferEmitView emit;
    TransferExportEligibility eligibility;
    int rank;
    std::string scaleUpExecMode;
    std::string exportTag;
};

void resetTelemetryExportMocks();

const std::vector<CollectiveExportRecord>& getCollectiveExportCalls();
const std::vector<P2PExportRecord>& getP2PExportCalls();
const std::vector<RankExportRecord>& getRankExportCalls();
const std::vector<TransferExportRecord>& getTransferExportCalls();
}  // namespace telemetrytest

#endif  // TESTS_MOCKS_TELEMETRY_EXPORT_MOCKS_H_