// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "telemetry_internal.h"

#ifdef ENABLE_OTEL

#include <atomic>
#include <cstdint>

#include "communicator_state.h"
#include "profiler_otel.h"
#include "telemetry_primer.h"

/**
 * @brief Process a window of events and export metrics.
 *
 * Aggregates all events in a window, links ProxyOps to their parent Collectives/P2Ps,
 * calculates metrics, and exports them to OpenTelemetry.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] window_idx Index of the window to process.
 */
void processWindow(CommunicatorState* commState, int window_idx)
{
    OTEL_TRACE(NCCL_INIT, "==> processWindow(window_idx=%d, rank=%d)", window_idx, commState->rank);

    WindowMetadata* window = commState->get_window_metadata(window_idx);
    WindowAggregator aggregator(commState->rank);

    otelEventHandle_t* buffer = commState->buffers[window_idx];
    uint32_t count            = window->element_count.load(std::memory_order_acquire);

    uint32_t skippedEvents = 0;
    for (uint32_t i = 0; i < count && i < BUFFER_SIZE; ++i)
    {
        const otelEventHandle_t& event = buffer[i];
        if (event.endTs > 0 && event.endTs >= event.startTs)
        {
            aggregator.addEvent(event);
        }
        else
        {
            skippedEvents++;
            OTEL_TRACE(NCCL_INIT, "Event %u type=%lu has endTs %f < startTs %f (skipped)", i, (unsigned long)event.type,
                       event.endTs, event.startTs);
        }
    }
    if (skippedEvents > 0)
    {
        OTEL_WARN(NCCL_INIT, "Window %u: skipped %u incomplete events", window_idx, skippedEvents);
    }

    aggregator.finalize();

    const std::map<std::string, AggregatedCollective>& collectives    = aggregator.getCollectives();
    const std::map<std::string, AggregatedP2P>& p2ps                  = aggregator.getP2Ps();
    const std::map<std::string, AggregatedTransfer>& rankTransfers    = aggregator.getRankTransfers();
    const std::map<std::string, AggregatedTransfer>& channelTransfers = aggregator.getChannelTransfers();

    std::set<std::string> handledCollectives      = processPendingCollectivePrimers(commState, collectives);
    std::set<std::string> handledP2Ps             = processPendingP2PPrimers(commState, p2ps);
    std::set<std::string> handledRankTransfers    = processPendingRankPrimers(commState, rankTransfers);
    std::set<std::string> handledChannelTransfers = processPendingTransferPrimers(commState, channelTransfers);

    for (const auto& pair : collectives)
    {
        if (handledCollectives.count(pair.first)) continue;

        if (isCollectivePrimerDone(commState, pair.first))
        {
            const AggregatedCollective& coll        = pair.second;
            CollectiveExportEligibility eligibility = computeCollectiveEligibility(coll);
            CollectiveEmitView emit                 = makeStandardCollectiveEmitView(coll);
            exportCollectiveMetrics(pair.first, emit, eligibility, commState->rank, commState->hostname,
                                    commState->local_rank, commState->comm_hash, commState->gpu_pci_bus_id,
                                    commState->gpu_uuid, commState->getCommTypeString(), commState->nranks,
                                    commState->getScaleUpExecModeString(), "STANDARD");
        }
        else
        {
            registerCollectivePrimer(commState, pair.first, pair.second);
        }
    }

    for (const auto& pair : p2ps)
    {
        if (handledP2Ps.count(pair.first)) continue;

        if (isP2PPrimerDone(commState, pair.first))
        {
            const AggregatedP2P& p2p         = pair.second;
            P2PExportEligibility eligibility = computeP2PEligibility(p2p);
            P2PEmitView emit                 = makeStandardP2PEmitView(p2p);
            exportP2PMetrics(pair.first, emit, eligibility, commState->rank, commState->hostname, commState->local_rank,
                             commState->comm_hash, commState->gpu_pci_bus_id, commState->gpu_uuid,
                             commState->getCommTypeString(), commState->nranks, commState->getScaleUpExecModeString(),
                             "STANDARD");
        }
        else
        {
            registerP2PPrimer(commState, pair.first, pair.second);
        }
    }

    for (const auto& pair : rankTransfers)
    {
        if (handledRankTransfers.count(pair.first)) continue;

        if (isRankPrimerDone(commState, pair.first))
        {
            const AggregatedTransfer& xfer    = pair.second;
            RankExportEligibility eligibility = computeRankEligibility(xfer);
            RankEmitView emit                 = makeStandardRankEmitView(xfer);
            exportRankMetrics(pair.first, emit, eligibility, commState->rank, commState->hostname,
                              commState->gpu_pci_bus_id, commState->gpu_uuid, commState->getCommTypeString(),
                              commState->nranks, commState->local_rank, commState->getScaleUpExecModeString(),
                              "STANDARD");
        }
        else
        {
            registerRankPrimer(commState, pair.first, pair.second);
        }
    }

    for (const auto& pair : channelTransfers)
    {
        if (handledChannelTransfers.count(pair.first)) continue;

        if (isTransferPrimerDone(commState, pair.first))
        {
            const AggregatedTransfer& xfer        = pair.second;
            TransferExportEligibility eligibility = computeTransferEligibility(xfer);
            TransferEmitView emit                 = makeStandardTransferEmitView(xfer);
            exportTransferMetrics(pair.first, emit, eligibility, commState->rank, commState->hostname,
                                  commState->gpu_pci_bus_id, commState->gpu_uuid, commState->getCommTypeString(),
                                  commState->nranks, commState->local_rank, commState->getScaleUpExecModeString(),
                                  "STANDARD");
        }
        else
        {
            registerTransferPrimer(commState, pair.first, pair.second);
        }
    }

    window->state.store(WINDOW_READY, std::memory_order_release);
    window->element_count.store(0, std::memory_order_release);
    window->in_progress_count.store(0, std::memory_order_release);

    OTEL_INFO(NCCL_INIT, "Window %d processed: %zu collectives, %zu P2Ps, %zu rank-transfers, %zu channel-transfers",
              window_idx, collectives.size(), p2ps.size(), rankTransfers.size(), channelTransfers.size());
    OTEL_TRACE(NCCL_INIT, "<== processWindow()");
}

#endif  // ENABLE_OTEL