// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "telemetry_internal.h"

#ifdef ENABLE_OTEL

#include <opentelemetry/context/context.h>
#include <opentelemetry/metrics/sync_instruments.h>

#include <array>
#include <string>
#include <string_view>

#include "profiler_otel.h"

namespace common = opentelemetry::common;

namespace
{
using Attribute = std::pair<nostd::string_view, common::AttributeValue>;

/**
 * @brief Convert a standard string view into an OpenTelemetry string view.
 *
 * @param[in] value Source string view.
 *
 * @return Non-owning OpenTelemetry view of the same characters.
 */
static inline nostd::string_view makeView(std::string_view value)
{
    return nostd::string_view{value.data(), value.size()};
}

/**
 * @brief Convert a standard string into an OpenTelemetry string view.
 *
 * @param[in] value Source string.
 *
 * @return Non-owning OpenTelemetry view of the same characters.
 */
static inline nostd::string_view makeView(const std::string& value)
{
    return nostd::string_view{value.data(), value.size()};
}

/**
 * @brief Build a string-valued OpenTelemetry attribute from a string view.
 *
 * @param[in] key Attribute name.
 * @param[in] value Attribute value.
 *
 * @return Attribute pair ready for metric emission.
 */
static inline Attribute makeStringAttribute(const char* key, std::string_view value)
{
    return {nostd::string_view{key}, common::AttributeValue{makeView(value)}};
}

/**
 * @brief Build a string-valued OpenTelemetry attribute from a standard string.
 *
 * @param[in] key Attribute name.
 * @param[in] value Attribute value.
 *
 * @return Attribute pair ready for metric emission.
 */
static inline Attribute makeStringAttribute(const char* key, const std::string& value)
{
    return {nostd::string_view{key}, common::AttributeValue{makeView(value)}};
}

/**
 * @brief Build a string-valued OpenTelemetry attribute from a C string.
 *
 * @param[in] key Attribute name.
 * @param[in] value Attribute value.
 *
 * @return Attribute pair ready for metric emission.
 */
static inline Attribute makeStringAttribute(const char* key, const char* value)
{
    return {nostd::string_view{key}, common::AttributeValue{nostd::string_view{value}}};
}

/**
 * @brief Parsed components extracted from rank and channel transfer keys.
 */
struct ParsedLinkKey
{
    std::string_view communicator;
    std::string_view channel;
    std::string sourceRank;
    std::string destRank;
    std::string_view metricCommType;
};

/**
 * @brief Parse a rank or channel transfer aggregation key into export labels.
 *
 * @param[in] key Aggregation key emitted by the window aggregator.
 * @param[in] commType Default communicator type for the current communicator.
 * @param[in] includeChannel Whether to extract an optional channel suffix.
 *
 * @return Parsed communicator, rank, channel, and metric-type fields.
 */
static ParsedLinkKey parseLinkKey(const std::string& key, const std::string& commType, bool includeChannel)
{
    ParsedLinkKey parts;
    parts.metricCommType = std::string_view{commType};

    std::string_view keyView{key};
    size_t commPos     = keyView.find("Comm");
    size_t firstSep    = keyView.find('_', commPos + 4);
    size_t pipelinePos = keyView.find("_Pipeline");
    size_t peerPos     = keyView.find("_ToPeer");
    size_t chnlPos     = includeChannel ? keyView.find("_Chnl") : std::string_view::npos;

    if (commPos != std::string_view::npos && firstSep != std::string_view::npos)
    {
        parts.communicator = keyView.substr(commPos + 4, firstSep - commPos - 4);
    }

    if (pipelinePos != std::string_view::npos && peerPos == std::string_view::npos)
    {
        size_t srcStart = pipelinePos + 9;
        size_t toPos    = keyView.find("_ToPipeline", srcStart);
        if (toPos != std::string_view::npos)
        {
            std::string_view srcPipeline = keyView.substr(srcStart, toPos - srcStart);
            size_t dstStart              = toPos + 11;
            size_t dstEnd                = (chnlPos != std::string_view::npos) ? chnlPos : keyView.size();
            std::string_view dstPipeline = keyView.substr(dstStart, dstEnd - dstStart);

            parts.sourceRank.reserve(8 + srcPipeline.size());
            parts.sourceRank.append("Pipeline");
            parts.sourceRank.append(srcPipeline);

            parts.destRank.reserve(8 + dstPipeline.size());
            parts.destRank.append("Pipeline");
            parts.destRank.append(dstPipeline);

            parts.metricCommType = "P2P";
        }
    }
    else if (peerPos != std::string_view::npos)
    {
        size_t rankPos = keyView.find("_Rank");
        if (rankPos != std::string_view::npos)
        {
            size_t sourceStart = rankPos + 5;
            parts.sourceRank.assign(keyView.substr(sourceStart, peerPos - sourceStart));
        }

        size_t destStart = peerPos + 7;
        size_t destEnd   = (chnlPos != std::string_view::npos) ? chnlPos : keyView.size();
        parts.destRank.assign(keyView.substr(destStart, destEnd - destStart));
        parts.metricCommType = "COLLECTIVE";
    }

    if (chnlPos != std::string_view::npos)
    {
        parts.channel = keyView.substr(chnlPos + 5);
    }

    return parts;
}

/**
 * @brief Build the human-readable P2P operation label from an aggregation key.
 *
 * @param[in] key P2P aggregation key.
 *
 * @return Operation label in the form `PipelineX -> PipelineY`.
 */
static std::string makeP2POperationLabel(const std::string& key)
{
    std::string_view keyView{key};
    std::string_view srcPipeline;
    std::string_view dstPipeline;

    size_t pipelinePos = keyView.find("_Pipeline");
    if (pipelinePos != std::string_view::npos)
    {
        size_t srcStart = pipelinePos + 9;
        size_t toPos    = keyView.find("ToPipeline", srcStart);
        if (toPos != std::string_view::npos)
        {
            srcPipeline     = keyView.substr(srcStart, toPos - srcStart);
            size_t dstStart = toPos + 10;
            size_t dstEnd   = keyView.find('_', dstStart);
            dstPipeline     = (dstEnd != std::string_view::npos) ? keyView.substr(dstStart, dstEnd - dstStart)
                                                                 : keyView.substr(dstStart);
        }
    }

    std::string operation;
    operation.reserve(24 + srcPipeline.size() + dstPipeline.size());
    operation.append("Pipeline");
    operation.append(srcPipeline);
    operation.append(" -> Pipeline");
    operation.append(dstPipeline);
    return operation;
}
}  // namespace

/**
 * @brief Export collective operation metrics to OpenTelemetry.
 *
 * @param[in] key Aggregation key for the collective operation.
 * @param[in] emit Values to record.
 * @param[in] eligibility Export guards derived from aggregation state.
 * @param[in] rank Global rank of the process.
 * @param[in] hostname Hostname of the node.
 * @param[in] local_rank Local rank within the node.
 * @param[in] comm_hash Communicator hash for labeling.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string.
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] scale_up_exec_mode Scale-up execution mode string.
 * @param[in] export_tag Trace-only export tag.
 */
void exportCollectiveMetrics(const std::string& key, const CollectiveEmitView& emit,
                             const CollectiveExportEligibility& eligibility, int rank, const std::string& hostname,
                             int local_rank, uint64_t comm_hash, const std::string& gpu_pci_bus_id,
                             const std::string& gpu_uuid, const std::string& comm_type, int nranks,
                             const std::string& scale_up_exec_mode, [[maybe_unused]] const char* export_tag)
{
    (void)comm_type;
    std::string rank_str       = std::to_string(rank);
    std::string local_rank_str = std::to_string(local_rank);
    std::string communicator   = std::to_string(comm_hash);
    std::string nranks_str     = std::to_string(nranks);

    if (eligibility.export_core)
    {
        OTEL_TRACE(NCCL_INIT,
                   "Exporting Collective (%s): %s, count=%.0f, totalBytes=%.0f, totalTime=%.2f us -> AvgBytes=%.2f, "
                   "AvgTime=%.2f us",
                   export_tag, key.c_str(), emit.count, emit.totalBytes, emit.totalTimeUs, emit.avgBytes,
                   emit.avgTimeUs);

        const auto context               = opentelemetry::context::Context{};
        std::array<Attribute, 10> labels = {
            makeStringAttribute("communicator", communicator),
            makeStringAttribute("operation", key),
            makeStringAttribute("rank", rank_str),
            makeStringAttribute("hostname", hostname),
            makeStringAttribute("local_rank", local_rank_str),
            makeStringAttribute("gpu_pci_bus_id", gpu_pci_bus_id),
            makeStringAttribute("gpu_uuid", gpu_uuid),
            makeStringAttribute("comm_type", "COLLECTIVE"),
            makeStringAttribute("comm_nranks", nranks_str),
            makeStringAttribute("scale_up_exec_mode", scale_up_exec_mode),
        };

        g_collBytesCounter->Add(emit.totalBytes, labels, context);
        g_collTimeHist->Record(emit.avgTimeUs, labels, context);
        g_collCountHist->Record((double)emit.count, labels, context);

        if (eligibility.export_transfers)
        {
            g_collNumTransfersHist->Record(emit.avgNumTransfers, labels, context);
            g_collTransferSizeHist->Record(emit.avgTransferSize, labels, context);
            if (eligibility.export_transfer_time)
            {
                g_collTransferTimeHist->Record(emit.avgTransferTime, labels, context);
            }

            OTEL_TRACE(NCCL_INIT,
                       "Exported Collective (%s): %s, AvgBytes: %.2f, AvgTime: %.2f us, "
                       "AvgNumTransfers: %.2f, AvgTransferSize: %.2f, AvgTransferTime: %.2f us",
                       export_tag, key.c_str(), emit.avgBytes, emit.avgTimeUs, emit.avgNumTransfers,
                       emit.avgTransferSize, emit.avgTransferTime);
        }
        else
        {
            OTEL_TRACE(NCCL_INIT, "Exported Collective (%s): %s, AvgBytes: %.2f, AvgTime: %.2f us (no transfers)",
                       export_tag, key.c_str(), emit.avgBytes, emit.avgTimeUs);
        }
    }
}

/**
 * @brief Export P2P operation metrics to OpenTelemetry.
 *
 * @param[in] key Aggregation key for the P2P operation.
 * @param[in] emit Values to record.
 * @param[in] eligibility Export guards derived from aggregation state.
 * @param[in] rank Global rank of the process.
 * @param[in] hostname Hostname of the node.
 * @param[in] local_rank Local rank within the node.
 * @param[in] comm_hash Communicator hash for labeling.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string.
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] scale_up_exec_mode Scale-up execution mode string.
 * @param[in] export_tag Trace-only export tag.
 */
void exportP2PMetrics(const std::string& key, const P2PEmitView& emit, const P2PExportEligibility& eligibility,
                      int rank, const std::string& hostname, int local_rank, uint64_t comm_hash,
                      const std::string& gpu_pci_bus_id, const std::string& gpu_uuid, const std::string& comm_type,
                      int nranks, const std::string& scale_up_exec_mode, [[maybe_unused]] const char* export_tag)
{
    (void)comm_type;
    std::string rank_str       = std::to_string(rank);
    std::string local_rank_str = std::to_string(local_rank);
    std::string communicator   = std::to_string(comm_hash);
    std::string nranks_str     = std::to_string(nranks);
    std::string operation      = makeP2POperationLabel(key);

    if (eligibility.export_core)
    {
        const auto context               = opentelemetry::context::Context{};
        std::array<Attribute, 10> labels = {
            makeStringAttribute("communicator", communicator),
            makeStringAttribute("operation", operation),
            makeStringAttribute("rank", rank_str),
            makeStringAttribute("hostname", hostname),
            makeStringAttribute("local_rank", local_rank_str),
            makeStringAttribute("gpu_pci_bus_id", gpu_pci_bus_id),
            makeStringAttribute("gpu_uuid", gpu_uuid),
            makeStringAttribute("comm_type", "P2P"),
            makeStringAttribute("comm_nranks", nranks_str),
            makeStringAttribute("scale_up_exec_mode", scale_up_exec_mode),
        };

        g_p2pBytesHist->Record(emit.avgBytes, labels, context);
        g_p2pTimeHist->Record(emit.avgTimeUs, labels, context);

        if (eligibility.export_transfers)
        {
            g_p2pNumTransfersHist->Record(emit.avgNumTransfers, labels, context);
            g_p2pTransferSizeHist->Record(emit.avgTransferSize, labels, context);
            if (eligibility.export_transfer_time)
            {
                g_p2pTransferTimeHist->Record(emit.avgTransferTime, labels, context);
            }

            OTEL_TRACE(NCCL_INIT,
                       "Exported P2P (%s): %s, AvgBytes: %.2f, AvgTime: %.2f us, "
                       "AvgNumTransfers: %.2f, AvgTransferSize: %.2f, AvgTransferTime: %.2f us",
                       export_tag, key.c_str(), emit.avgBytes, emit.avgTimeUs, emit.avgNumTransfers,
                       emit.avgTransferSize, emit.avgTransferTime);
        }
        else
        {
            OTEL_TRACE(NCCL_INIT, "Exported P2P (%s): %s, AvgBytes: %.2f, AvgTime: %.2f us (no transfers)", export_tag,
                       key.c_str(), emit.avgBytes, emit.avgTimeUs);
        }
    }
}

/**
 * @brief Export rank transfer metrics to OpenTelemetry.
 *
 * @param[in] key Aggregation key for the rank transfer.
 * @param[in] emit Values to record.
 * @param[in] eligibility Export guards derived from aggregation state.
 * @param[in] rank Global rank of the process.
 * @param[in] hostname Hostname of the node.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string.
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] local_rank Local rank within the node.
 * @param[in] scale_up_exec_mode Scale-up execution mode string.
 * @param[in] export_tag Trace-only export tag.
 */
void exportRankMetrics(const std::string& key, const RankEmitView& emit, const RankExportEligibility& eligibility,
                       int rank, const std::string& hostname, const std::string& gpu_pci_bus_id,
                       const std::string& gpu_uuid, const std::string& comm_type, int nranks, int local_rank,
                       const std::string& scale_up_exec_mode, [[maybe_unused]] const char* export_tag)
{
    (void)rank;
    ParsedLinkKey parts              = parseLinkKey(key, comm_type, false);
    std::string nranks_str           = std::to_string(nranks);
    std::string localRankStr         = std::to_string(local_rank);
    const auto context               = opentelemetry::context::Context{};
    std::array<Attribute, 10> labels = {
        makeStringAttribute("communicator", parts.communicator),
        makeStringAttribute("source_rank", parts.sourceRank),
        makeStringAttribute("dest_rank", parts.destRank),
        makeStringAttribute("hostname", hostname),
        makeStringAttribute("gpu_pci_bus_id", gpu_pci_bus_id),
        makeStringAttribute("gpu_uuid", gpu_uuid),
        makeStringAttribute("comm_type", parts.metricCommType),
        makeStringAttribute("comm_nranks", nranks_str),
        makeStringAttribute("local_rank", localRankStr),
        makeStringAttribute("scale_up_exec_mode", scale_up_exec_mode),
    };

    g_rankBytesCounter->Add(emit.totalBytes, labels, context);

    if (eligibility.export_latency)
    {
        g_rankLatencyHist->Record(emit.latencyUs, labels, context);
        OTEL_TRACE(NCCL_INIT, "Exported Rank Latency (%s): %s, Latency: %.2f us", export_tag, key.c_str(),
                   emit.latencyUs);
    }

    if (eligibility.export_rate)
    {
        g_rankRateHist->Record(emit.rateMBps, labels, context);
        OTEL_TRACE(NCCL_INIT, "Exported Rank Rate (%s): %s, Bytes: %llu, ActiveTime: %.2f us, Rate: %.2f MB/s",
                   export_tag, key.c_str(), static_cast<unsigned long long>(emit.totalBytes), emit.activeTimeUs,
                   emit.rateMBps);
    }
    else
    {
        OTEL_TRACE(NCCL_INIT, "Exported Rank Metrics (%s): %s, Bytes: %llu (no rate data)", export_tag, key.c_str(),
                   static_cast<unsigned long long>(emit.totalBytes));
    }
}

/**
 * @brief Export per-channel transfer metrics to OpenTelemetry.
 *
 * @param[in] key Aggregation key for the channel transfer.
 * @param[in] emit Values to record.
 * @param[in] eligibility Export guards derived from aggregation state.
 * @param[in] rank Global rank of the process.
 * @param[in] hostname Hostname of the node.
 * @param[in] gpu_pci_bus_id GPU PCI BUS ID.
 * @param[in] gpu_uuid GPU UUID.
 * @param[in] comm_type Communicator type string.
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] local_rank Local rank within the node.
 * @param[in] scale_up_exec_mode Scale-up execution mode string.
 * @param[in] export_tag Trace-only export tag.
 */
void exportTransferMetrics(const std::string& key, const TransferEmitView& emit,
                           const TransferExportEligibility& eligibility, int rank, const std::string& hostname,
                           const std::string& gpu_pci_bus_id, const std::string& gpu_uuid, const std::string& comm_type,
                           int nranks, int local_rank, const std::string& scale_up_exec_mode,
                           [[maybe_unused]] const char* export_tag)
{
    (void)rank;
    ParsedLinkKey parts      = parseLinkKey(key, comm_type, true);
    std::string nranks_str   = std::to_string(nranks);
    std::string localRankStr = std::to_string(local_rank);

    if (eligibility.export_channel_metrics)
    {
        const auto context               = opentelemetry::context::Context{};
        std::array<Attribute, 11> labels = {
            makeStringAttribute("communicator", parts.communicator),
            makeStringAttribute("source_rank", parts.sourceRank),
            makeStringAttribute("dest_rank", parts.destRank),
            makeStringAttribute("channel", parts.channel),
            makeStringAttribute("hostname", hostname),
            makeStringAttribute("gpu_pci_bus_id", gpu_pci_bus_id),
            makeStringAttribute("gpu_uuid", gpu_uuid),
            makeStringAttribute("comm_type", parts.metricCommType),
            makeStringAttribute("comm_nranks", nranks_str),
            makeStringAttribute("local_rank", localRankStr),
            makeStringAttribute("scale_up_exec_mode", scale_up_exec_mode),
        };

        g_transferSizeHist->Record(emit.avgSize, labels, context);
        if (eligibility.export_avg_time)
        {
            g_transferTimeHist->Record(emit.avgTime, labels, context);
        }

        if (eligibility.export_latency)
        {
            g_transferLatencyHist->Record(emit.latencyUs, labels, context);
            OTEL_TRACE(NCCL_INIT, "Exported Transfer (%s): %s, AvgSize: %.2f, AvgTime: %.2f us, Latency: %.2f us",
                       export_tag, key.c_str(), emit.avgSize, emit.avgTime, emit.latencyUs);
        }
        else
        {
            OTEL_TRACE(NCCL_INIT, "Exported Transfer (%s): %s, AvgSize: %.2f, AvgTime: %.2f us", export_tag,
                       key.c_str(), emit.avgSize, emit.avgTime);
        }
    }
}

#endif  // ENABLE_OTEL