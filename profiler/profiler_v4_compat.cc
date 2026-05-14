// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0
//
// Thin v4 -> v5 adapter layer.
// Converts v4 API types to v5 and forwards to the v5 implementation.
// Exported as ncclProfiler_v4 so older NCCL builds (v4-only) can load us.

#include "profiler_v4_compat.h"

#include <cstring>

// v4 activation mask: strip event types that don't exist in v4 (bits > 7).
// KernelCh (bit 6) and NetPlugin (bit 7) are the highest v4 knows about.
static constexpr int V4_EVENT_MASK = 0xFF;

/**
 * @brief Initialize the profiler through the v4 compatibility entry point.
 *
 * @param[out] context Pointer that receives the profiler context.
 * @param[out] eActivationMask Activation mask returned to NCCL v4.
 * @param[in] commName Communicator display name.
 * @param[in] commHash Communicator unique identifier.
 * @param[in] nNodes Number of nodes in the communicator.
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] rank Rank of the current process.
 * @param[in] logfn NCCL logger callback.
 *
 * @return ncclSuccess on success.
 */
OTEL_HIDDEN ncclResult_t profiler_otel_init_v4(void** context, int* eActivationMask, const char* commName,
                                               uint64_t commHash, int nNodes, int nranks, int rank,
                                               ncclDebugLogger_t logfn)
{
    ncclResult_t ret = profiler_otel_init_v5(context, commHash, eActivationMask, commName, nNodes, nranks, rank, logfn);

    if (ret == ncclSuccess && eActivationMask)
    {
        *eActivationMask &= V4_EVENT_MASK;
    }

    return ret;
}

/**
 * @brief Translate a v4 event descriptor and forward it to the v5 implementation.
 *
 * @param[in] context Profiler context returned by initialization.
 * @param[out] eHandle Pointer that receives the allocated event handle.
 * @param[in] eDescr v4 event descriptor to translate.
 *
 * @return ncclSuccess on success.
 */
OTEL_HIDDEN ncclResult_t profiler_otel_start_event_v4(void* context, void** eHandle,
                                                      ncclProfilerEventDescr_v4_t* eDescr)
{
    ncclProfilerEventDescr_v5_t descr_v5 = {};
    descr_v5.type                        = (uint64_t)eDescr->type;
    descr_v5.parentObj                   = eDescr->parentObj;
    descr_v5.rank                        = eDescr->rank;

    switch (eDescr->type)
    {
        case ncclProfileGroup:
            break;
        case ncclProfileColl:
            descr_v5.coll.seqNumber = eDescr->coll.seqNumber;
            descr_v5.coll.func      = eDescr->coll.func;
            descr_v5.coll.sendBuff  = eDescr->coll.sendBuff;
            descr_v5.coll.recvBuff  = eDescr->coll.recvBuff;
            descr_v5.coll.count     = eDescr->coll.count;
            descr_v5.coll.root      = eDescr->coll.root;
            descr_v5.coll.datatype  = eDescr->coll.datatype;
            descr_v5.coll.nChannels = eDescr->coll.nChannels;
            descr_v5.coll.nWarps    = eDescr->coll.nWarps;
            descr_v5.coll.algo      = eDescr->coll.algo;
            descr_v5.coll.proto     = eDescr->coll.proto;
            break;
        case ncclProfileP2p:
            descr_v5.p2p.func      = eDescr->p2p.func;
            descr_v5.p2p.buff      = eDescr->p2p.buff;
            descr_v5.p2p.datatype  = eDescr->p2p.datatype;
            descr_v5.p2p.count     = eDescr->p2p.count;
            descr_v5.p2p.peer      = eDescr->p2p.peer;
            descr_v5.p2p.nChannels = eDescr->p2p.nChannels;
            break;
        case ncclProfileProxyOp:
            descr_v5.proxyOp.pid       = eDescr->proxyOp.pid;
            descr_v5.proxyOp.channelId = eDescr->proxyOp.channelId;
            descr_v5.proxyOp.peer      = eDescr->proxyOp.peer;
            descr_v5.proxyOp.nSteps    = eDescr->proxyOp.nSteps;
            descr_v5.proxyOp.chunkSize = eDescr->proxyOp.chunkSize;
            descr_v5.proxyOp.isSend    = eDescr->proxyOp.isSend;
            break;
        case ncclProfileProxyStep:
            descr_v5.proxyStep.step = eDescr->proxyStep.step;
            break;
        case ncclProfileProxyCtrl:
            break;
        case ncclProfileKernelCh:
            descr_v5.kernelCh.channelId = eDescr->kernelCh.channelId;
            descr_v5.kernelCh.pTimer    = eDescr->kernelCh.pTimer;
            break;
        case ncclProfileNetPlugin:
            descr_v5.netPlugin.id   = eDescr->netPlugin.id;
            descr_v5.netPlugin.data = eDescr->netPlugin.data;
            break;
        default:
            *eHandle = nullptr;
            return ncclSuccess;
    }

    return profiler_otel_start_event_v5(context, eHandle, &descr_v5);
}

/**
 * @brief Forward a v4 event state transition to the v5 implementation.
 *
 * @param[in] eHandle Event handle returned by the start-event entry point.
 * @param[in] eState v4 state enum value.
 * @param[in] eStateArgs Optional state-specific arguments.
 *
 * @return ncclSuccess on success.
 */
OTEL_HIDDEN ncclResult_t profiler_otel_record_event_state_v4(void* eHandle, ncclProfilerEventState_v4_t eState,
                                                             ncclProfilerEventStateArgs_v4_t* eStateArgs)
{
    // v4 and v5 state enums are identical typedefs; state arg unions have the
    // same layout (proxyStep.transSize, proxyCtrl.appendedProxyOps,
    // netPlugin.data, kernelCh.pTimer).  Safe to reinterpret.
    return profiler_otel_record_event_state_v5(eHandle, static_cast<ncclProfilerEventState_v5_t>(eState),
                                               reinterpret_cast<ncclProfilerEventStateArgs_v5_t*>(eStateArgs));
}
