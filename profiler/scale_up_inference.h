// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef SCALE_UP_INFERENCE_H_
#define SCALE_UP_INFERENCE_H_

#include <cstddef>
#include <cstdint>

#define SCALE_UP_MAX_TRANSFER_BYTES    ((size_t)1024 * 1024)
#define SCALE_UP_SLICE_SPLIT_THRESHOLD ((size_t)64 * 1024)
#define SCALE_UP_TREE_TRANSFER_BYTES   ((size_t)32 * 1024)

/**
 * Result of transfer inference for a single collective or P2P operation.
 */
struct InferredTransfers
{
    size_t perTransferBytes;     // Individual transfer size (capped at 1 MB)
    int numTransfers;            // Total number of transfers for this rank
    size_t totalRankBytes;       // Total bytes this rank transfers through the internal network
    double networkTimeFraction;  // Fraction of collective time assumed to be networking [0.0, 1.0]
    int stepsPerRank;            // Number of logical steps this rank participates in
    int numChannels;             // Number of channels used
};

InferredTransfers inferCollectiveTransfers(const char* func, const char* algo, size_t collectiveBytes, int nRanks,
                                           uint8_t nChannels, double networkPct);

InferredTransfers inferP2PTransfers(size_t p2pBytes, uint8_t nChannels, double networkPct);

#endif  // SCALE_UP_INFERENCE_H_
