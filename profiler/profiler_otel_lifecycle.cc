// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "profiler_otel_lifecycle.h"

#include <pthread.h>
#include <unistd.h>

#include <cstdlib>
#include <string>

#include "communicator_state.h"
#include "events.h"
#include "param.h"
#include "profiler_gpu_metadata.h"
#include "profiler_otel.h"
#include "profiler_runtime_state.h"
#include "telemetry.h"

OTEL_HIDDEN double gettime(void);

// PARAM: EnableOTEL
// ENV: NCCL_PROFILER_OTEL_ENABLE
// DEFAULT: 1
// DESCRIPTION: Master enable/disable switch for the profiler plugin (0 disables plugin).
OTEL_PARAM(EnableOTEL, "PROFILER_OTEL_ENABLE", 1);
// PARAM: ProfileEventMask
// ENV: NCCL_PROFILE_EVENT_MASK
// DEFAULT: -1 (use internal default)
// DESCRIPTION: Override NCCL profiler activation mask; if unset, plugin uses 0x85E
//              (Coll+P2P+ProxyOp+ProxyStep+KernelCh+KernelLaunch).
OTEL_PARAM(ProfileEventMask, "PROFILE_EVENT_MASK", -1);
// PARAM: WindowTimeoutIntervalSec
// ENV: NCCL_PROFILER_OTEL_TELEMETRY_INTERVAL_SEC
// DEFAULT: 5
// DESCRIPTION: Window timeout used for time-based window closing (seconds). Kept with lifecycle
//              bookkeeping so init/finalize work stays out of the event-path translation unit.
OTEL_PARAM(WindowTimeoutIntervalSec, "PROFILER_OTEL_TELEMETRY_INTERVAL_SEC", 5);

/**
 * @brief Initialize the NCCL activation mask shared across plugin contexts.
 *
 * @param[out] eActivationMask Activation mask returned to NCCL.
 */
static void initializeActivationMask(int* eActivationMask)
{
    static int localActivationMask = 0;

    pthread_mutex_lock(&otelLock);
    if (__atomic_fetch_add(&initialized, 1, __ATOMIC_RELAXED) == 0)
    {
        int64_t envMask     = OTEL_GET_PARAM(ProfileEventMask);
        localActivationMask = (envMask >= 0) ? (int)envMask : 0x85E;

        OTEL_INFO(NCCL_INIT, "Event activation mask set to 0x%x", localActivationMask);

        pid       = getpid();
        startTime = gettime();
    }

    __atomic_store_n(eActivationMask, localActivationMask, __ATOMIC_RELAXED);
    pthread_mutex_unlock(&otelLock);
}

/**
 * @brief Populate communicator hostname metadata.
 *
 * @param[in,out] commState Communicator state to populate.
 */
static void populateHostMetadata(CommunicatorState* commState)
{
    char hostname_buf[256];
    if (gethostname(hostname_buf, sizeof(hostname_buf)) == 0)
    {
        commState->hostname = std::string(hostname_buf);
    }
    else
    {
        commState->hostname = "unknown";
    }
}

/**
 * @brief Assign the configured window timeout to a communicator state.
 *
 * @param[in,out] commState Communicator state to configure.
 */
static void assignWindowTimeout(CommunicatorState* commState)
{
    int interval_sec = (int)OTEL_GET_PARAM(WindowTimeoutIntervalSec);
    if (interval_sec <= 0)
    {
        interval_sec = 5;
    }

    commState->window_timeout_usec = interval_sec * 1e6;
    OTEL_INFO(NCCL_INIT, "Window timeout set to %d seconds (%.0f us)", interval_sec, commState->window_timeout_usec);
}

/**
 * @brief Allocate and populate the profiler event context for a communicator.
 *
 * @param[in] commId Communicator unique identifier.
 * @param[in] commName Communicator display name.
 * @param[in] nNodes Number of nodes in the communicator.
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] rank Rank of the current process.
 *
 * @return Newly allocated event context.
 */
static eventContext* createEventContext(uint64_t commId, const char* commName, int nNodes, int nranks, int rank)
{
    eventContext* ctx = static_cast<eventContext*>(calloc(1, sizeof(eventContext)));
    ctx->commName     = commName;
    ctx->commHash     = commId;
    ctx->nNodes       = nNodes;
    ctx->nranks       = nranks;
    ctx->rank         = rank;
    ctx->commState    = new CommunicatorState();

    ctx->commState->comm_name = commName;
    ctx->commState->comm_hash = commId;
    ctx->commState->nNodes    = nNodes;
    ctx->commState->nranks    = nranks;
    ctx->commState->rank      = rank;
    ctx->commState->commName  = commName ? std::string(commName) : std::string("");

    populateHostMetadata(ctx->commState);
    populateGpuMetadata(ctx->commState);
    resolveLocalRankAndCommType(ctx->commState, rank, nranks);
    assignWindowTimeout(ctx->commState);

    OTEL_INFO(
        NCCL_INIT,
        "Created communicator state: name=%s, hash=%lu, rank=%d, nranks=%d, nNodes=%d, hostname=%s, local_rank=%d, "
        "gpu_pci_bus_id=%s, gpu_uuid=%s, comm_type=%s",
        commName, commId, rank, nranks, nNodes, ctx->commState->hostname.c_str(), ctx->commState->local_rank,
        ctx->commState->gpu_pci_bus_id.c_str(), ctx->commState->gpu_uuid.c_str(), ctx->commState->getCommTypeString());

    return ctx;
}

/**
 * @brief Initialize profiler state for a communicator.
 *
 * @param[out] context Pointer that receives the allocated profiler context.
 * @param[in] commId Communicator unique identifier.
 * @param[out] eActivationMask Activation mask returned to NCCL.
 * @param[in] commName Communicator display name.
 * @param[in] nNodes Number of nodes in the communicator.
 * @param[in] nranks Number of ranks in the communicator.
 * @param[in] rank Rank of the current process.
 *
 * @return ncclSuccess on success.
 */
ncclResult_t initializeProfilerContext(void** context, uint64_t commId, int* eActivationMask, const char* commName,
                                       int nNodes, int nranks, int rank)
{
    int enable = OTEL_GET_PARAM(EnableOTEL);
    OTEL_TRACE(NCCL_INIT, "Checking enable parameter: NCCL_PROFILER_OTEL_ENABLE=%d", enable);
    if (enable == 0)
    {
        OTEL_WARN(NCCL_INIT, "Plugin disabled by environment variable NCCL_PROFILER_OTEL_ENABLE=0");
        *context = nullptr;
        return ncclSuccess;
    }

    initializeActivationMask(eActivationMask);
    *context = createEventContext(commId, commName, nNodes, nranks, rank);

    if (__atomic_fetch_add(&telemetry_initialized, 1, __ATOMIC_RELAXED) == 0)
    {
        profiler_otel_telemetry_init();
    }

    __atomic_fetch_add(&active_communicators, 1, __ATOMIC_RELAXED);
    return ncclSuccess;
}

/**
 * @brief Finalize profiler state for a communicator.
 *
 * @param[in] context Profiler context to destroy.
 *
 * @return ncclSuccess on success.
 */
ncclResult_t finalizeProfilerContext(void* context)
{
    eventContext* ctx = static_cast<eventContext*>(context);

    if (ctx && ctx->commState)
    {
        profiler_otel_telemetry_unregister_communicator(ctx->commState);
        OTEL_INFO(NCCL_INIT, "Destroying communicator state: name=%s, hash=%lu", ctx->commState->comm_name,
                  ctx->commState->comm_hash);
        delete ctx->commState;
        ctx->commState = nullptr;
    }

    free(ctx);

    int remaining = __atomic_sub_fetch(&active_communicators, 1, __ATOMIC_ACQ_REL);
    if (remaining == 0)
    {
        profiler_otel_telemetry_cleanup();
        __atomic_store_n(&telemetry_initialized, 0, __ATOMIC_RELEASE);
    }

    return ncclSuccess;
}