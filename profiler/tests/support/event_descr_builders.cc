// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "event_descr_builders.h"

#include <unistd.h>

namespace testsupport
{
ncclProfilerEventDescr_v5_t makeCollDescr(const char* func, size_t count, const char* datatype, uint8_t nChannels,
                                          const char* algo, const char* proto, void* parent)
{
    ncclProfilerEventDescr_v5_t descr = {};
    descr.type                        = ncclProfileColl;
    descr.coll.func                   = func;
    descr.coll.datatype               = datatype;
    descr.coll.count                  = count;
    descr.coll.nChannels              = nChannels;
    descr.coll.algo                   = algo;
    descr.coll.proto                  = proto;
    descr.parentObj                   = parent;
    return descr;
}

ncclProfilerEventDescr_v5_t makeP2pDescr(const char* func, size_t count, const char* datatype, int peer,
                                         uint8_t nChannels, void* parent)
{
    ncclProfilerEventDescr_v5_t descr = {};
    descr.type                        = ncclProfileP2p;
    descr.p2p.func                    = func;
    descr.p2p.datatype                = datatype;
    descr.p2p.count                   = count;
    descr.p2p.peer                    = peer;
    descr.p2p.nChannels               = nChannels;
    descr.parentObj                   = parent;
    return descr;
}

ncclProfilerEventDescr_v5_t makeProxyOpDescr(uint8_t channelId, int peer, int chunkSize, int isSend, void* parent)
{
    ncclProfilerEventDescr_v5_t descr = {};
    descr.type                        = ncclProfileProxyOp;
    descr.proxyOp.channelId           = channelId;
    descr.proxyOp.peer                = peer;
    descr.proxyOp.chunkSize           = chunkSize;
    descr.proxyOp.isSend              = isSend;
    descr.proxyOp.pid                 = getpid();
    descr.parentObj                   = parent;
    return descr;
}

ncclProfilerEventDescr_v5_t makeProxyStepDescr(int step, void* parent)
{
    ncclProfilerEventDescr_v5_t descr = {};
    descr.type                        = ncclProfileProxyStep;
    descr.proxyStep.step              = step;
    descr.parentObj                   = parent;
    return descr;
}

ncclProfilerEventDescr_v5_t makeGroupDescr(void* parent)
{
    ncclProfilerEventDescr_v5_t descr = {};
    descr.type                        = ncclProfileGroup;
    descr.parentObj                   = parent;
    return descr;
}

ncclProfilerEventDescr_v5_t makeP2pApiDescr(const char* func, void* parent)
{
    ncclProfilerEventDescr_v5_t descr = {};
    descr.type                        = ncclProfileP2pApi;
    descr.p2pApi.func                 = func;
    descr.parentObj                   = parent;
    return descr;
}

ncclProfilerEventDescr_v5_t makeKernelChDescr(uint8_t channelId, void* parent)
{
    ncclProfilerEventDescr_v5_t descr = {};
    descr.type                        = ncclProfileKernelCh;
    descr.kernelCh.channelId          = channelId;
    descr.kernelCh.pTimer             = 0;
    descr.parentObj                   = parent;
    return descr;
}
}  // namespace testsupport