// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef PROFILER_GPU_METADATA_H_
#define PROFILER_GPU_METADATA_H_

struct CommunicatorState;

void populateGpuMetadata(CommunicatorState* commState);
void resolveLocalRankAndCommType(CommunicatorState* commState, int rank, int nranks);

#endif  // PROFILER_GPU_METADATA_H_