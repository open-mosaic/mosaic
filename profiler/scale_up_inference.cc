// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "scale_up_inference.h"

#include <cmath>
#include <cstring>

/**
 * @file scale_up_inference.cc
 * @brief Heuristics for reconstructing scale-up transfer size/count from NCCL collective metadata.
 *
 * The profiler only records the collective API payload (`count * datatype_size` for this rank),
 * the NCCL algorithm string, and the channel count. There are no ProxyOp / ProxyStep events for
 * intra-node PCIe/NVLink traffic, so this file mirrors the chunking families that NCCL wires in
 * `nccl/src/enqueue.cc::calcCollChunking()` and turns them into transfer telemetry.
 *
 * The reconstruction uses three stages:
 * 1. Expand the per-rank API payload into NCCL's "global bytes" input to chunking.
 *    - AllReduce / Broadcast / Reduce: `nBytesGlobal = collectiveBytes`
 *    - AllGather / ReduceScatter: `nBytesGlobal = collectiveBytes * nranks`
 * 2. Estimate the total bytes this rank moves across the scale-up fabric.
 *    - AllReduce: `2 * (nranks - 1) / nranks * collectiveBytes`
 *    - AllGather / ReduceScatter: `(nranks - 1) * collectiveBytes`
 *    - Broadcast / Reduce: `(nranks - 1) / nranks * collectiveBytes`
 * 3. Choose the transfer family from NCCL's algorithm-to-pattern mapping.
 *    - Ring/pipeline family: `RING` and the non-tree Broadcast/Reduce pipeline forms.
 *      We model one logical step per ring stage, then multiply by channels and the local slice /
 *      1 MiB sub-transfer split used by the plugin heuristic.
 *    - Tree-like family: `TREE`, `NVLS_TREE`, `COLLNET_CHAIN`.
 *      NCCL uses one chunked tree path rather than ring steps, so we model chunk count as
 *      `ceil(totalRankBytes / perTransferBytes)` with a 32 KiB tree transfer cap.
 *    - Direct chunked family: `NVLS`, `COLLNET_DIRECT`, `PAT`.
 *      NCCL maps these to one-step chunked patterns (`Nvls`, `CollnetDirect`, `PatUp/PatDown`).
 *      They should not be multiplied by ring step count; instead we derive a per-channel chunk size
 *      from `nBytesGlobal / nChannels` and count chunks from `totalRankBytes`.
 *
 * These formulas intentionally smooth over rank-position details inside trees / NVLS heads. The goal
 * is stable Grafana telemetry that tracks the dominant chunking regime, not an exact replay of every
 * internal NCCL lane.
 */

namespace
{
enum class ScaleUpAlgoFamily
{
    RingPipeline,
    TreeLike,
    DirectChunked,
};

/**
 * @brief Classify an NCCL algorithm string into the profiler's scale-up transfer families.
 *
 * @param[in] algo NCCL algorithm string from the profiler event.
 *
 * @return The transfer family used by the inference model.
 */
ScaleUpAlgoFamily classifyScaleUpAlgo(const char* algo)
{
    if (!algo) return ScaleUpAlgoFamily::RingPipeline;

    if (strstr(algo, "COLLNET_CHAIN")) return ScaleUpAlgoFamily::TreeLike;
    if (strstr(algo, "TREE") || strstr(algo, "Tree")) return ScaleUpAlgoFamily::TreeLike;
    if (strstr(algo, "NVLS") || strstr(algo, "COLLNET_DIRECT") || strstr(algo, "PAT"))
    {
        return ScaleUpAlgoFamily::DirectChunked;
    }

    return ScaleUpAlgoFamily::RingPipeline;
}

/**
 * @brief Derive the representative transfer size for a chunked path.
 *
 * Applies the plugin's local slice split and 1 MiB cap heuristics and returns
 * the resulting per-transfer byte count while reporting the split factors.
 *
 * @param[in] bytesPerPath Bytes assigned to one logical path.
 * @param[out] slicesPerChunk Number of slices created by the local split heuristic.
 * @param[out] numSubTransfers Number of 1 MiB-capped sub-transfers per slice.
 *
 * @return Inferred bytes per transfer.
 */
size_t inferChunkedTransferSize(size_t bytesPerPath, int& slicesPerChunk, int& numSubTransfers)
{
    if (bytesPerPath == 0) bytesPerPath = 1;

    slicesPerChunk = 1;
    if (bytesPerPath >= SCALE_UP_SLICE_SPLIT_THRESHOLD)
    {
        slicesPerChunk = 2;
        bytesPerPath /= 2;
    }

    numSubTransfers    = 1;
    size_t perTransfer = bytesPerPath;
    if (bytesPerPath > SCALE_UP_MAX_TRANSFER_BYTES)
    {
        numSubTransfers = (int)std::ceil((double)bytesPerPath / SCALE_UP_MAX_TRANSFER_BYTES);
        perTransfer     = SCALE_UP_MAX_TRANSFER_BYTES;
    }

    return perTransfer;
}
}  // namespace

/**
 * @brief Infer transfer characteristics for a collective operation on scale-up.
 *
 * Uses the collective type, NCCL algorithm family, data size, rank count and channel count to estimate:
 * - Per-transfer size
 * - Total number of transfers for this rank
 * - Total bytes transferred by this rank on the internal network
 *
 * @param[in] func Collective function name.
 * @param[in] algo Algorithm name.
 * @param[in] collectiveBytes Data size from the profiler event.
 * @param[in] nRanks Number of ranks in the communicator.
 * @param[in] nChannels Number of channels used by this collective.
 * @param[in] networkPct Percentage of collective time assumed spent on networking.
 *
 * @return Inferred transfer parameters for the collective.
 */
InferredTransfers inferCollectiveTransfers(const char* func, const char* algo, size_t collectiveBytes, int nRanks,
                                           uint8_t nChannels, double networkPct)
{
    InferredTransfers result   = {};
    result.networkTimeFraction = (networkPct > 0 && networkPct <= 100) ? networkPct / 100.0 : 1.0;
    result.numChannels         = nChannels > 0 ? nChannels : 1;

    if (collectiveBytes == 0 || nRanks <= 1)
    {
        result.perTransferBytes = 0;
        result.numTransfers     = 0;
        result.totalRankBytes   = 0;
        result.stepsPerRank     = 0;
        return result;
    }

    size_t nBytesGlobal      = collectiveBytes;
    double trafficMultiplier = 1.0;
    int stepsPerRank         = 1;

    if (func)
    {
        if (strstr(func, "AllReduce"))
        {
            nBytesGlobal      = collectiveBytes;
            trafficMultiplier = 2.0;
            stepsPerRank      = 2 * (nRanks - 1);
        }
        else if (strstr(func, "AllGather"))
        {
            nBytesGlobal      = collectiveBytes * (size_t)nRanks;
            trafficMultiplier = (double)nRanks;
            stepsPerRank      = nRanks - 1;
        }
        else if (strstr(func, "ReduceScatter"))
        {
            nBytesGlobal      = collectiveBytes * (size_t)nRanks;
            trafficMultiplier = (double)nRanks;
            stepsPerRank      = nRanks - 1;
        }
        else if (strstr(func, "Broadcast") || strstr(func, "Reduce"))
        {
            nBytesGlobal      = collectiveBytes;
            trafficMultiplier = 1.0;
            stepsPerRank      = 1;
        }
    }

    result.stepsPerRank = stepsPerRank;
    result.totalRankBytes =
        (size_t)((double)collectiveBytes * trafficMultiplier * (double)(nRanks - 1) / (double)nRanks);

    ScaleUpAlgoFamily algoFamily = classifyScaleUpAlgo(algo);

    if (algoFamily == ScaleUpAlgoFamily::TreeLike)
    {
        size_t treePerChannel = nBytesGlobal / (size_t)result.numChannels;
        if (treePerChannel == 0) treePerChannel = 1;

        size_t treeTransferSize =
            treePerChannel <= SCALE_UP_TREE_TRANSFER_BYTES ? treePerChannel : SCALE_UP_TREE_TRANSFER_BYTES;

        result.perTransferBytes = treeTransferSize;
        result.numTransfers =
            result.totalRankBytes > 0 ? (int)std::ceil((double)result.totalRankBytes / treeTransferSize) : 0;
        return result;
    }

    const bool isDirectChunked = (algoFamily == ScaleUpAlgoFamily::DirectChunked);
    size_t bytesPerPath        = isDirectChunked ? (nBytesGlobal / (size_t)result.numChannels)
                                                 : (nBytesGlobal / (size_t)nRanks / (size_t)result.numChannels);

    int slicesPerChunk  = 1;
    int numSubTransfers = 1;
    size_t perTransfer  = inferChunkedTransferSize(bytesPerPath, slicesPerChunk, numSubTransfers);

    result.perTransferBytes = perTransfer;
    if (isDirectChunked)
    {
        result.numTransfers =
            result.totalRankBytes > 0 ? (int)std::ceil((double)result.totalRankBytes / (double)perTransfer) : 0;
        return result;
    }

    result.numTransfers = stepsPerRank * result.numChannels * slicesPerChunk * numSubTransfers;
    return result;
}

/**
 * @brief Infer transfer characteristics for a P2P operation on scale-up.
 *
 * @param[in] p2pBytes Total bytes in the P2P operation.
 * @param[in] nChannels Number of channels used.
 * @param[in] networkPct Percentage of P2P time assumed spent on networking.
 *
 * @return Inferred transfer parameters for the P2P operation.
 */
InferredTransfers inferP2PTransfers(size_t p2pBytes, uint8_t nChannels, double networkPct)
{
    InferredTransfers result   = {};
    result.networkTimeFraction = (networkPct > 0 && networkPct <= 100) ? networkPct / 100.0 : 1.0;
    result.numChannels         = nChannels > 0 ? nChannels : 1;
    result.stepsPerRank        = 1;

    if (p2pBytes == 0)
    {
        result.perTransferBytes = 0;
        result.numTransfers     = 0;
        result.totalRankBytes   = 0;
        return result;
    }

    size_t perChannelBytes = p2pBytes / (size_t)result.numChannels;
    if (perChannelBytes == 0) perChannelBytes = 1;

    int slicesPerChunk = 1;
    if (perChannelBytes >= SCALE_UP_SLICE_SPLIT_THRESHOLD)
    {
        slicesPerChunk = 2;
        perChannelBytes /= 2;
    }

    int numSubTransfers = 1;
    size_t perTransfer  = perChannelBytes;
    if (perChannelBytes > SCALE_UP_MAX_TRANSFER_BYTES)
    {
        numSubTransfers = (int)std::ceil((double)perChannelBytes / SCALE_UP_MAX_TRANSFER_BYTES);
        perTransfer     = SCALE_UP_MAX_TRANSFER_BYTES;
    }

    result.perTransferBytes = perTransfer;
    result.numTransfers     = result.numChannels * slicesPerChunk * numSubTransfers;
    result.totalRankBytes   = p2pBytes;
    return result;
}