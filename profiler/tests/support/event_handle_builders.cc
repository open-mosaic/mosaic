// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "event_handle_builders.h"

namespace testsupport
{
otelEventHandle_t makeCollectiveEvent(const char* func, const char* algo, const char* proto, uint8_t channels,
                                      size_t bytes, double startTs, double endTs)
{
    otelEventHandle_t event = {};
    event.type              = ncclProfileColl;
    event.coll.func         = func;
    event.coll.algo         = algo;
    event.coll.proto        = proto;
    event.coll.nChannels    = channels;
    event.coll.bytes        = bytes;
    event.startTs           = startTs;
    event.endTs             = endTs;
    event.parentObj         = (void*)0x1234;
    event.rank              = 0;
    return event;
}

otelEventHandle_t makeP2PEvent(const char* func, int peer, uint8_t channels, size_t bytes, double startTs, double endTs)
{
    otelEventHandle_t event = {};
    event.type              = ncclProfileP2p;
    event.p2p.func          = func;
    event.p2p.peer          = peer;
    event.p2p.nChannels     = channels;
    event.p2p.bytes         = bytes;
    event.startTs           = startTs;
    event.endTs             = endTs;
    event.parentObj         = (void*)0x5678;
    event.rank              = 0;
    return event;
}

otelEventHandle_t makeProxyOpEvent(int peer, uint8_t channelId, int chunkSize, double startTs, double endTs,
                                   void* parentObj)
{
    otelEventHandle_t event = {};
    event.type              = ncclProfileProxyOp;
    event.proxyOp.peer      = peer;
    event.proxyOp.channelId = channelId;
    event.proxyOp.chunkSize = chunkSize;
    event.startTs           = startTs;
    event.endTs             = endTs;
    event.parentObj         = parentObj;
    event.rank              = 0;
    return event;
}

otelEventHandle_t makeProxyStepEvent(int step, size_t transSize, double startTs, double sendWaitTs, double endTs,
                                     void* parentObj)
{
    otelEventHandle_t event     = {};
    event.type                  = ncclProfileProxyStep;
    event.proxyStep.step        = step;
    event.proxyStep.transSize   = transSize;
    event.proxyStep.sendWaitTs  = sendWaitTs;
    event.proxyStep.hasSendWait = (transSize > 0);
    event.startTs               = startTs;
    event.endTs                 = endTs;
    event.parentObj             = parentObj;
    event.rank                  = 0;
    return event;
}

otelEventHandle_t makeKernelChEvent(uint8_t channelId, uint64_t pTimerStart, uint64_t pTimerStop, double startTs,
                                    double endTs, void* parentObj)
{
    otelEventHandle_t event    = {};
    event.type                 = ncclProfileKernelCh;
    event.kernelCh.channelId   = channelId;
    event.kernelCh.pTimerStart = pTimerStart;
    event.kernelCh.pTimerStop  = pTimerStop;
    event.kernelCh.hasStop     = true;
    event.startTs              = startTs;
    event.endTs                = endTs;
    event.parentObj            = parentObj;
    event.rank                 = 0;
    return event;
}

otelEventHandle_t makeGroupEvent(double startTs, double endTs, CommunicatorState* commState)
{
    otelEventHandle_t event = {};
    event.type              = ncclProfileGroup;
    event.startTs           = startTs;
    event.endTs             = endTs;
    event.commState         = commState;
    event.rank              = commState ? commState->rank : 0;
    return event;
}

otelEventHandle_t makeP2pApiEvent(const char* func, double startTs, double endTs, CommunicatorState* commState)
{
    otelEventHandle_t event = {};
    event.type              = ncclProfileP2pApi;
    event.p2pApi.func       = func;
    event.startTs           = startTs;
    event.endTs             = endTs;
    event.commState         = commState;
    event.rank              = commState ? commState->rank : 0;
    return event;
}

otelEventHandle_t makeCollectiveEventWithCommState(const char* func, const char* algo, const char* proto,
                                                   uint8_t channels, size_t bytes, double startTs, double endTs,
                                                   CommunicatorState* commState)
{
    auto event      = makeCollectiveEvent(func, algo, proto, channels, bytes, startTs, endTs);
    event.commState = commState;
    return event;
}
}  // namespace testsupport