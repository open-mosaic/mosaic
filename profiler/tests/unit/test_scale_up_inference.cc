// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>

#include "../../scale_up_inference.h"

// =============================================================================
// inferCollectiveTransfers - Edge Cases
// =============================================================================

class ScaleUpInferenceCollTest : public ::testing::Test
{
protected:
    static constexpr double kFullNetwork = 100.0;
};

TEST_F(ScaleUpInferenceCollTest, ZeroBytes)
{
    auto r = inferCollectiveTransfers("AllReduce", "Ring", 0, 4, 2, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 0u);
    EXPECT_EQ(r.numTransfers, 0);
    EXPECT_EQ(r.totalRankBytes, 0u);
    EXPECT_EQ(r.stepsPerRank, 0);
}

TEST_F(ScaleUpInferenceCollTest, SingleRank)
{
    auto r = inferCollectiveTransfers("AllReduce", "Ring", 1024, 1, 2, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 0u);
    EXPECT_EQ(r.numTransfers, 0);
    EXPECT_EQ(r.totalRankBytes, 0u);
}

TEST_F(ScaleUpInferenceCollTest, NullFunc)
{
    auto r = inferCollectiveTransfers(nullptr, "Ring", 4096, 4, 2, kFullNetwork);
    EXPECT_GT(r.perTransferBytes, 0u);
    EXPECT_GT(r.numTransfers, 0);
}

TEST_F(ScaleUpInferenceCollTest, NullAlgo)
{
    auto r = inferCollectiveTransfers("AllReduce", nullptr, 4096, 4, 2, kFullNetwork);
    EXPECT_GT(r.perTransferBytes, 0u);
}

TEST_F(ScaleUpInferenceCollTest, ZeroChannels)
{
    auto r = inferCollectiveTransfers("AllReduce", "Ring", 4096, 4, 0, kFullNetwork);
    EXPECT_EQ(r.numChannels, 1);
    EXPECT_GT(r.perTransferBytes, 0u);
}

TEST_F(ScaleUpInferenceCollTest, NetworkPctBounds)
{
    auto r0 = inferCollectiveTransfers("AllReduce", "Ring", 4096, 4, 2, 0.0);
    EXPECT_DOUBLE_EQ(r0.networkTimeFraction, 1.0);

    auto r50 = inferCollectiveTransfers("AllReduce", "Ring", 4096, 4, 2, 50.0);
    EXPECT_DOUBLE_EQ(r50.networkTimeFraction, 0.5);

    auto rNeg = inferCollectiveTransfers("AllReduce", "Ring", 4096, 4, 2, -10.0);
    EXPECT_DOUBLE_EQ(rNeg.networkTimeFraction, 1.0);

    auto rOver = inferCollectiveTransfers("AllReduce", "Ring", 4096, 4, 2, 200.0);
    EXPECT_DOUBLE_EQ(rOver.networkTimeFraction, 1.0);
}

// =============================================================================
// inferCollectiveTransfers - Ring Algorithm
// =============================================================================

TEST_F(ScaleUpInferenceCollTest, AllReduceRingSmall)
{
    // AllReduce 4KB, 4 ranks, 2 channels, Ring
    // nBytesGlobal = 4096
    // baseTransferSize = 4096 / 4 / 2 = 512  (< 64KB, no slice split)
    // stepsPerRank = 2*(4-1) = 6
    // numTransfers = 6 * 2 * 1 * 1 = 12
    // totalRankBytes = 4096 * 2.0 * 3/4 = 6144
    auto r = inferCollectiveTransfers("AllReduce", "Ring", 4096, 4, 2, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 512u);
    EXPECT_EQ(r.numTransfers, 12);
    EXPECT_EQ(r.totalRankBytes, 6144u);
    EXPECT_EQ(r.stepsPerRank, 6);
    EXPECT_EQ(r.numChannels, 2);
}

TEST_F(ScaleUpInferenceCollTest, AllReduceRingSliceSplit)
{
    // AllReduce with baseTransferSize >= 64KB triggers slice split
    // 2MB data, 2 ranks, 1 channel, Ring
    // nBytesGlobal = 2MB = 2097152
    // baseTransferSize = 2097152 / 2 / 1 = 1048576 (1MB >= 64KB → split)
    // After split: 524288 (512KB, <= 1MB cap)
    // slicesPerChunk = 2
    // stepsPerRank = 2*(2-1) = 2
    // numTransfers = 2 * 1 * 2 * 1 = 4
    auto r = inferCollectiveTransfers("AllReduce", "Ring", 2097152, 2, 1, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 524288u);
    EXPECT_EQ(r.numTransfers, 4);
    EXPECT_EQ(r.stepsPerRank, 2);
}

TEST_F(ScaleUpInferenceCollTest, AllReduceRingExactSliceThreshold)
{
    // AllReduce where baseTransferSize == 64KB exactly
    // collectiveBytes such that nBytesGlobal / nRanks / nChannels = 64KB
    // 2 ranks, 1 channel → need nBytesGlobal = 64KB * 2 = 128KB = 131072
    // baseTransferSize = 131072 / 2 / 1 = 65536 = 64KB → split
    // After split: 32768
    auto r = inferCollectiveTransfers("AllReduce", "Ring", 131072, 2, 1, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 32768u);
    EXPECT_EQ(r.numTransfers, 2 * 1 * 2 * 1);  // steps=2, ch=1, slices=2, sub=1
}

TEST_F(ScaleUpInferenceCollTest, AllReduceRingJustBelowSliceThreshold)
{
    // baseTransferSize just below 64KB → no slice split
    // 2 ranks, 1 channel → nBytesGlobal = 2 * (64KB - 2) = 131068
    // But integer math: 131068 / 2 / 1 = 65534 < 65536 → no split
    auto r = inferCollectiveTransfers("AllReduce", "Ring", 131068, 2, 1, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 65534u);
    EXPECT_EQ(r.numTransfers, 2 * 1 * 1 * 1);  // no slice split
}

TEST_F(ScaleUpInferenceCollTest, AllReduceRing1MBCap)
{
    // Very large data triggers 1MB cap after slice split
    // 64MB AllReduce, 2 ranks, 1 channel
    // nBytesGlobal = 67108864
    // baseTransferSize = 67108864 / 2 / 1 = 33554432 (32MB, >= 64KB → split)
    // After split: 16777216 (16MB > 1MB → subdivide)
    // numSubTransfers = ceil(16777216 / 1048576) = 16
    // perTransfer = 1MB
    auto r = inferCollectiveTransfers("AllReduce", "Ring", 67108864, 2, 1, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, (size_t)SCALE_UP_MAX_TRANSFER_BYTES);
    EXPECT_EQ(r.numTransfers, 2 * 1 * 2 * 16);  // steps=2, ch=1, slices=2, sub=16
}

TEST_F(ScaleUpInferenceCollTest, AllGatherRing)
{
    // AllGather: collectiveBytes = per-rank sendcount * typeSize = D/N
    // nBytesGlobal = collectiveBytes * nRanks
    // 1024 bytes per rank, 4 ranks, 2 channels
    // nBytesGlobal = 1024 * 4 = 4096
    // baseTransferSize = 4096 / 4 / 2 = 512
    // trafficMultiplier = 4.0
    // totalRankBytes = 1024 * 4.0 * 3/4 = 3072
    // stepsPerRank = 3
    // numTransfers = 3 * 2 * 1 * 1 = 6
    auto r = inferCollectiveTransfers("AllGather", "Ring", 1024, 4, 2, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 512u);
    EXPECT_EQ(r.numTransfers, 6);
    EXPECT_EQ(r.totalRankBytes, 3072u);
    EXPECT_EQ(r.stepsPerRank, 3);
}

TEST_F(ScaleUpInferenceCollTest, ReduceScatterRing)
{
    // ReduceScatter: same nBytesGlobal logic as AllGather
    auto r = inferCollectiveTransfers("ReduceScatter", "Ring", 1024, 4, 2, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 512u);
    EXPECT_EQ(r.numTransfers, 6);
    EXPECT_EQ(r.totalRankBytes, 3072u);
    EXPECT_EQ(r.stepsPerRank, 3);
}

TEST_F(ScaleUpInferenceCollTest, BroadcastRing)
{
    // Broadcast: trafficMultiplier=1, stepsPerRank=1
    // 4096 bytes, 4 ranks, 2 channels
    // baseTransferSize = 4096 / 4 / 2 = 512
    // totalRankBytes = 4096 * 1.0 * 3/4 = 3072
    auto r = inferCollectiveTransfers("Broadcast", "Ring", 4096, 4, 2, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 512u);
    EXPECT_EQ(r.stepsPerRank, 1);
    EXPECT_EQ(r.numTransfers, 1 * 2 * 1 * 1);  // steps=1, ch=2
    EXPECT_EQ(r.totalRankBytes, 3072u);
}

TEST_F(ScaleUpInferenceCollTest, ReduceRing)
{
    // Reduce: same as Broadcast
    auto r = inferCollectiveTransfers("Reduce", "Ring", 4096, 4, 2, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 512u);
    EXPECT_EQ(r.stepsPerRank, 1);
    EXPECT_EQ(r.totalRankBytes, 3072u);
}

// =============================================================================
// inferCollectiveTransfers - Tree Algorithm
// =============================================================================

TEST_F(ScaleUpInferenceCollTest, AllReduceTreeSmall)
{
    // Tree: small collective where treePerChannel < 32KB → uses actual size
    // 4KB AllReduce, 4 ranks, 2 channels
    // nBytesGlobal = 4096
    // treePerChannel = 4096 / 2 = 2048 < 32KB → perTransfer = 2048
    // totalRankBytes = 4096 * 2.0 * 3/4 = 6144
    // numTransfers = ceil(6144 / 2048) = 3
    auto r = inferCollectiveTransfers("AllReduce", "Tree", 4096, 4, 2, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 2048u);
    EXPECT_EQ(r.numTransfers, 3);
    EXPECT_EQ(r.totalRankBytes, 6144u);
}

TEST_F(ScaleUpInferenceCollTest, AllReduceTreeLarge)
{
    // Tree: large collective → capped at 32KB
    // 1MB AllReduce, 4 ranks, 2 channels
    // nBytesGlobal = 1048576
    // treePerChannel = 1048576 / 2 = 524288 > 32KB → perTransfer = 32KB = 32768
    // totalRankBytes = 1048576 * 2.0 * 3/4 = 1572864
    // numTransfers = ceil(1572864 / 32768) = 48
    auto r = inferCollectiveTransfers("AllReduce", "TREE", 1048576, 4, 2, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, (size_t)SCALE_UP_TREE_TRANSFER_BYTES);
    EXPECT_EQ(r.numTransfers, 48);
    EXPECT_EQ(r.totalRankBytes, 1572864u);
}

TEST_F(ScaleUpInferenceCollTest, AllReduceTreeExact32KB)
{
    // Tree: treePerChannel == 32KB exactly → uses 32KB
    // Need nBytesGlobal / nChannels = 32KB
    // 1 channel, 2 ranks → nBytesGlobal = 32768
    // treePerChannel = 32768 / 1 = 32768 = 32KB → perTransfer = 32KB
    auto r = inferCollectiveTransfers("AllReduce", "Tree", 32768, 2, 1, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, (size_t)SCALE_UP_TREE_TRANSFER_BYTES);
}

TEST_F(ScaleUpInferenceCollTest, AllReduceTreeJustBelow32KB)
{
    // Tree: treePerChannel just below 32KB → uses actual value
    // nBytesGlobal = 32766, 1 channel, 2 ranks
    // treePerChannel = 32766 / 1 = 32766 < 32KB → perTransfer = 32766
    auto r = inferCollectiveTransfers("AllReduce", "Tree", 32766, 2, 1, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 32766u);
}

TEST_F(ScaleUpInferenceCollTest, TreeCaseInsensitive)
{
    // Both "Tree" and "TREE" should be recognized
    auto r1 = inferCollectiveTransfers("AllReduce", "Tree", 1048576, 4, 2, kFullNetwork);
    auto r2 = inferCollectiveTransfers("AllReduce", "TREE", 1048576, 4, 2, kFullNetwork);
    EXPECT_EQ(r1.perTransferBytes, r2.perTransferBytes);
    EXPECT_EQ(r1.numTransfers, r2.numTransfers);
}

TEST_F(ScaleUpInferenceCollTest, NvlsTreeUsesTreeLikeModel)
{
    auto r = inferCollectiveTransfers("AllReduce", "NVLS_TREE", 1048576, 4, 2, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, (size_t)SCALE_UP_TREE_TRANSFER_BYTES);
    EXPECT_EQ(r.numTransfers, 48);
    EXPECT_EQ(r.totalRankBytes, 1572864u);
}

TEST_F(ScaleUpInferenceCollTest, CollNetChainUsesTreeLikeModel)
{
    auto r = inferCollectiveTransfers("AllReduce", "COLLNET_CHAIN", 1048576, 4, 2, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, (size_t)SCALE_UP_TREE_TRANSFER_BYTES);
    EXPECT_EQ(r.numTransfers, 48);
    EXPECT_EQ(r.totalRankBytes, 1572864u);
}

// =============================================================================
// inferCollectiveTransfers - Direct chunked families
// =============================================================================

TEST_F(ScaleUpInferenceCollTest, AllReduceNvlsUsesDirectChunkedModel)
{
    // NVLS AllReduce uses the chunked direct path in NCCL rather than RingTwice.
    // For 4KB, 4 ranks, 2 channels:
    // nBytesGlobal = 4096
    // bytesPerPath = 4096 / 2 = 2048
    // totalRankBytes = 6144
    // numTransfers = ceil(6144 / 2048) = 3
    auto r = inferCollectiveTransfers("AllReduce", "NVLS", 4096, 4, 2, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 2048u);
    EXPECT_EQ(r.numTransfers, 3);
    EXPECT_EQ(r.totalRankBytes, 6144u);
}

TEST_F(ScaleUpInferenceCollTest, AllReduceCollNetDirectUsesDirectChunkedModel)
{
    auto r = inferCollectiveTransfers("AllReduce", "COLLNET_DIRECT", 4096, 4, 2, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 2048u);
    EXPECT_EQ(r.numTransfers, 3);
    EXPECT_EQ(r.totalRankBytes, 6144u);
}

TEST_F(ScaleUpInferenceCollTest, AllGatherPatUsesDirectChunkedModel)
{
    // PAT AllGather uses one chunked transfer stream per channel rather than ring steps.
    // nBytesGlobal = 1024 * 4 = 4096
    // bytesPerPath = 4096 / 2 = 2048
    // totalRankBytes = 3072
    // numTransfers = ceil(3072 / 2048) = 2
    auto r = inferCollectiveTransfers("AllGather", "PAT", 1024, 4, 2, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 2048u);
    EXPECT_EQ(r.numTransfers, 2);
    EXPECT_EQ(r.totalRankBytes, 3072u);
}

// =============================================================================
// inferCollectiveTransfers - 8-rank scenarios (realistic)
// =============================================================================

TEST_F(ScaleUpInferenceCollTest, AllReduce8Ranks4Bytes)
{
    // Tiny AllReduce: 4 bytes, 8 ranks, 2 channels, Ring
    // nBytesGlobal = 4
    // baseTransferSize = 4 / 8 / 2 = 0 → clamped to 1
    // stepsPerRank = 2*(8-1) = 14
    // numTransfers = 14 * 2 * 1 * 1 = 28
    auto r = inferCollectiveTransfers("AllReduce", "Ring", 4, 8, 2, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 1u);
    EXPECT_EQ(r.stepsPerRank, 14);
    EXPECT_EQ(r.numTransfers, 28);
    EXPECT_EQ(r.totalRankBytes, 7u);
}

TEST_F(ScaleUpInferenceCollTest, AllReduce8Ranks1MB)
{
    // 1MB AllReduce, 8 ranks, 2 channels, Ring
    // nBytesGlobal = 1048576
    // baseTransferSize = 1048576 / 8 / 2 = 65536 (= 64KB → slice split)
    // After split: 32768
    // stepsPerRank = 14
    // numTransfers = 14 * 2 * 2 * 1 = 56
    auto r = inferCollectiveTransfers("AllReduce", "Ring", 1048576, 8, 2, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 32768u);
    EXPECT_EQ(r.numTransfers, 56);
}

// =============================================================================
// inferP2PTransfers
// =============================================================================

class ScaleUpInferenceP2PTest : public ::testing::Test
{
protected:
    static constexpr double kFullNetwork = 100.0;
};

TEST_F(ScaleUpInferenceP2PTest, ZeroBytes)
{
    auto r = inferP2PTransfers(0, 1, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 0u);
    EXPECT_EQ(r.numTransfers, 0);
    EXPECT_EQ(r.totalRankBytes, 0u);
}

TEST_F(ScaleUpInferenceP2PTest, SmallTransfer)
{
    // 4KB, 1 channel
    // perChannelBytes = 4096 / 1 = 4096 (< 64KB, no slice split)
    // numTransfers = 1 * 1 * 1 = 1
    auto r = inferP2PTransfers(4096, 1, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 4096u);
    EXPECT_EQ(r.numTransfers, 1);
    EXPECT_EQ(r.totalRankBytes, 4096u);
    EXPECT_EQ(r.stepsPerRank, 1);
}

TEST_F(ScaleUpInferenceP2PTest, MultiChannel)
{
    // 8KB, 2 channels
    // perChannelBytes = 8192 / 2 = 4096 (< 64KB, no split)
    // numTransfers = 2 * 1 * 1 = 2
    auto r = inferP2PTransfers(8192, 2, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 4096u);
    EXPECT_EQ(r.numTransfers, 2);
    EXPECT_EQ(r.totalRankBytes, 8192u);
}

TEST_F(ScaleUpInferenceP2PTest, SliceSplit)
{
    // 128KB, 1 channel
    // perChannelBytes = 131072 / 1 = 131072 (>= 64KB → split)
    // After split: 65536 (64KB, <= 1MB)
    // numTransfers = 1 * 2 * 1 = 2
    auto r = inferP2PTransfers(131072, 1, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 65536u);
    EXPECT_EQ(r.numTransfers, 2);
    EXPECT_EQ(r.totalRankBytes, 131072u);
}

TEST_F(ScaleUpInferenceP2PTest, ExactSliceThreshold)
{
    // Exactly 64KB, 1 channel → triggers split
    auto r = inferP2PTransfers(65536, 1, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 32768u);
    EXPECT_EQ(r.numTransfers, 2);
}

TEST_F(ScaleUpInferenceP2PTest, JustBelowSliceThreshold)
{
    // Just below 64KB → no split
    auto r = inferP2PTransfers(65534, 1, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, 65534u);
    EXPECT_EQ(r.numTransfers, 1);
}

TEST_F(ScaleUpInferenceP2PTest, OneMBCap)
{
    // 4MB, 1 channel
    // perChannelBytes = 4194304 (>= 64KB → split)
    // After split: 2097152 (2MB > 1MB → subdivide)
    // numSubTransfers = ceil(2097152 / 1048576) = 2
    // numTransfers = 1 * 2 * 2 = 4
    auto r = inferP2PTransfers(4194304, 1, kFullNetwork);
    EXPECT_EQ(r.perTransferBytes, (size_t)SCALE_UP_MAX_TRANSFER_BYTES);
    EXPECT_EQ(r.numTransfers, 4);
    EXPECT_EQ(r.totalRankBytes, 4194304u);
}

TEST_F(ScaleUpInferenceP2PTest, ZeroChannelsDefaultsToOne)
{
    auto r = inferP2PTransfers(4096, 0, kFullNetwork);
    EXPECT_EQ(r.numChannels, 1);
    EXPECT_EQ(r.perTransferBytes, 4096u);
    EXPECT_EQ(r.numTransfers, 1);
}

TEST_F(ScaleUpInferenceP2PTest, NetworkPctHalf)
{
    auto r = inferP2PTransfers(4096, 1, 50.0);
    EXPECT_DOUBLE_EQ(r.networkTimeFraction, 0.5);
    EXPECT_EQ(r.perTransferBytes, 4096u);
}
