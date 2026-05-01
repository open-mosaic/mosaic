// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef EVENTS_H_
#define EVENTS_H_

#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

#include "profiler_otel.h"

// Forward declaration
struct CommunicatorState;

/**
 * Context of the profiler plugin instance.
 */
struct eventContext
{
    const char* commName;
    uint64_t commHash;
    int nNodes;
    int nranks;
    int rank;
    struct CommunicatorState* commState;  // Pointer to circular buffer state
};

/**
 * Lightweight event handle for circular buffer storage.
 * This structure is optimized for lock-free operations and minimal memory footprint.
 * It stores the essential information needed for telemetry export.
 */
typedef struct
{
    uint64_t type;       // ncclProfileGroup, ncclProfileColl, ncclProfileP2p, ncclProfileProxyOp, ncclProfileProxyStep,
                         // ncclProfileProxyCtrl, ncclProfileKernelCh, ncclProfileKernelLaunch
    uint8_t buffer_idx;  // Window/buffer index this event belongs to (for mark_operation_complete)
    void* parentObj;     // pointer to parent profiler object
    struct CommunicatorState* commState;  // Back-pointer to commState (for mark_operation_complete)
    int rank;                             // originating rank
    double startTs;
    double endTs;
    union
    {
        struct
        {
            uint64_t seqNumber;
            const char* func;
            size_t bytes;
            uint8_t nChannels;
            const char* algo;
            const char* proto;
            bool firstChildCompleted;
        } coll;
        struct
        {
            const char* func;
            size_t bytes;
            int peer;
            uint8_t nChannels;
            bool firstChildCompleted;
        } p2p;
        struct
        {
            uint8_t channelId;
            int peer;
            int chunkSize;
        } proxyOp;
        struct
        {
            int step;
            size_t transSize;   // Real transfer size from recordEventState with ProxyStepSendWait
            double sendWaitTs;  // Timestamp when ProxyStepSendWait state is recorded (start of actual transfer)
            bool hasSendWait;   // Whether SendWait state was recorded
        } proxyStep;
        struct
        {
            uint8_t channelId;
            uint64_t pTimerStart;  // GPU globaltimer at kernel channel start
            uint64_t pTimerStop;   // GPU globaltimer at kernel channel stop (from KernelChStop state)
            bool hasStop;          // Whether KernelChStop was recorded
        } kernelCh;
        struct
        {
            // KernelLaunch: no per-event payload; timing is captured in startTs/endTs.
        } kernelLaunch;
        struct
        {
            // P2pApi: carries the original collective function name (e.g., "AlltoAll").
            // Used as a grouping parent for P2P tasks decomposed from a collective call.
            const char* func;
        } p2pApi;
    };
} otelEventHandle_t;

#endif  // EVENTS_H_
