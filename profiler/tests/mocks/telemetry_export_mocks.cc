// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "telemetry_export_mocks.h"

namespace
{
std::vector<telemetrytest::CollectiveExportRecord> g_collectiveExportCalls;
std::vector<telemetrytest::P2PExportRecord> g_p2pExportCalls;
std::vector<telemetrytest::RankExportRecord> g_rankExportCalls;
std::vector<telemetrytest::TransferExportRecord> g_transferExportCalls;
}  // namespace

namespace telemetrytest
{
void resetTelemetryExportMocks()
{
    g_collectiveExportCalls.clear();
    g_p2pExportCalls.clear();
    g_rankExportCalls.clear();
    g_transferExportCalls.clear();
}

const std::vector<CollectiveExportRecord>& getCollectiveExportCalls()
{
    return g_collectiveExportCalls;
}

const std::vector<P2PExportRecord>& getP2PExportCalls()
{
    return g_p2pExportCalls;
}

const std::vector<RankExportRecord>& getRankExportCalls()
{
    return g_rankExportCalls;
}

const std::vector<TransferExportRecord>& getTransferExportCalls()
{
    return g_transferExportCalls;
}
}  // namespace telemetrytest

void exportCollectiveMetrics(const std::string& key, const CollectiveEmitView& emit,
                             const CollectiveExportEligibility& eligibility, int rank, const std::string& hostname,
                             int local_rank, uint64_t comm_hash, const std::string& gpu_pci_bus_id,
                             const std::string& gpu_uuid, const std::string& comm_type, int nranks,
                             const std::string& scale_up_exec_mode, const char* export_tag)
{
    (void)hostname;
    (void)local_rank;
    (void)comm_hash;
    (void)gpu_pci_bus_id;
    (void)gpu_uuid;
    (void)comm_type;
    (void)nranks;
    g_collectiveExportCalls.push_back({key, emit, eligibility, rank, scale_up_exec_mode, export_tag ? export_tag : ""});
}

void exportP2PMetrics(const std::string& key, const P2PEmitView& emit, const P2PExportEligibility& eligibility,
                      int rank, const std::string& hostname, int local_rank, uint64_t comm_hash,
                      const std::string& gpu_pci_bus_id, const std::string& gpu_uuid, const std::string& comm_type,
                      int nranks, const std::string& scale_up_exec_mode, const char* export_tag)
{
    (void)hostname;
    (void)local_rank;
    (void)comm_hash;
    (void)gpu_pci_bus_id;
    (void)gpu_uuid;
    (void)comm_type;
    (void)nranks;
    g_p2pExportCalls.push_back({key, emit, eligibility, rank, scale_up_exec_mode, export_tag ? export_tag : ""});
}

void exportRankMetrics(const std::string& key, const RankEmitView& emit, const RankExportEligibility& eligibility,
                       int rank, const std::string& hostname, const std::string& gpu_pci_bus_id,
                       const std::string& gpu_uuid, const std::string& comm_type, int nranks, int local_rank,
                       const std::string& scale_up_exec_mode, const char* export_tag)
{
    (void)hostname;
    (void)gpu_pci_bus_id;
    (void)gpu_uuid;
    (void)comm_type;
    (void)nranks;
    (void)local_rank;
    g_rankExportCalls.push_back({key, emit, eligibility, rank, scale_up_exec_mode, export_tag ? export_tag : ""});
}

void exportTransferMetrics(const std::string& key, const TransferEmitView& emit,
                           const TransferExportEligibility& eligibility, int rank, const std::string& hostname,
                           const std::string& gpu_pci_bus_id, const std::string& gpu_uuid, const std::string& comm_type,
                           int nranks, int local_rank, const std::string& scale_up_exec_mode, const char* export_tag)
{
    (void)hostname;
    (void)gpu_pci_bus_id;
    (void)gpu_uuid;
    (void)comm_type;
    (void)nranks;
    (void)local_rank;
    g_transferExportCalls.push_back({key, emit, eligibility, rank, scale_up_exec_mode, export_tag ? export_tag : ""});
}