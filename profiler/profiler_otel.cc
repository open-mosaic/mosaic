// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "profiler_otel.h"

#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include "communicator_state.h"

// =============================================================================
// Global GPU ID to Rank Map
// =============================================================================
// Maps GPU identifier (PCI bus ID) to the correct rank within the pipeline.
// Populated from COLLECTIVE communicators (nNodes=1) where rank == GPU index.
// Used by P2P communicators to look up the correct local_rank.
static std::map<std::string, int> g_gpu_id_to_rank;
static std::mutex g_gpu_rank_map_mutex;
#include "events.h"
#include "param.h"
#include "telemetry.h"

// =============================================================================
// GPU Platform Abstraction Layer
// Provides unified API for CUDA and ROCm/HIP
// =============================================================================

#if defined(GPU_PLATFORM_ROCM)
// ROCm/HIP platform
#include <hip/hip_runtime.h>

// Type aliases for HIP
using gpuError_t    = hipError_t;
using gpuDeviceProp = hipDeviceProp_t;

// HIP's UUID is in hipDeviceProp_t as hipUUID (char[16] or similar structure)
// Define a compatible UUID type
struct gpuUUID_t
{
    char bytes[16];
};

// Error codes
#define gpuSuccess hipSuccess

// Function mappings
#define gpuGetDevice           hipGetDevice
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceGetPCIBusId   hipDeviceGetPCIBusId
#define gpuGetErrorString      hipGetErrorString

// Platform name for logging
#define GPU_PLATFORM_NAME "ROCm/HIP"

#else
// CUDA platform (default)
#include <cuda_runtime.h>

// Type aliases for CUDA
using gpuError_t    = cudaError_t;
using gpuDeviceProp = cudaDeviceProp;
using gpuUUID_t     = cudaUUID_t;

// Error codes
#define gpuSuccess             cudaSuccess

// Function mappings
#define gpuGetDevice           cudaGetDevice
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceGetPCIBusId   cudaDeviceGetPCIBusId
#define gpuGetErrorString      cudaGetErrorString

// Platform name for logging
#define GPU_PLATFORM_NAME      "CUDA"

#endif  // GPU_PLATFORM_ROCM

// =============================================================================
// End of GPU Platform Abstraction Layer
// =============================================================================

/**
 * @brief Get the size in bytes of an NCCL datatype.
 *
 * @param[in] datatype NCCL datatype string (e.g., "ncclFloat32", "ncclInt64").
 *
 * @return Size in bytes, or 0 if datatype is unknown or nullptr.
 */
static size_t ncclTypeSize(const char* datatype)
{
    if (!datatype) return 0;
    if (strcmp(datatype, "ncclInt8") == 0 || strcmp(datatype, "ncclUint8") == 0) return 1;
    if (strcmp(datatype, "ncclFloat16") == 0 || strcmp(datatype, "ncclBfloat16") == 0) return 2;
    if (strcmp(datatype, "ncclInt32") == 0 || strcmp(datatype, "ncclUint32") == 0 ||
        strcmp(datatype, "ncclFloat32") == 0)
        return 4;
    if (strcmp(datatype, "ncclInt64") == 0 || strcmp(datatype, "ncclUint64") == 0 ||
        strcmp(datatype, "ncclFloat64") == 0)
        return 8;
    return 0;  // Unknown type
}

/**
 * @brief Convert GPU UUID to standard UUID string format.
 *
 * Converts a gpuUUID_t struct to a human-readable UUID string in the format:
 * xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (8-4-4-4-12 hexadecimal digits).
 *
 * @param[in] uuid GPU UUID structure (cudaUUID_t on CUDA, gpuUUID_t on ROCm).
 *
 * @return UUID string in standard format, or "unknown" if conversion fails.
 */
static std::string gpuUuidToString(const gpuUUID_t& uuid)
{
    char uuid_str[64];
    // uuid.bytes is const char*, cast to unsigned char* for proper formatting
    const unsigned char* uuid_bytes = reinterpret_cast<const unsigned char*>(uuid.bytes);
    // Cast each byte to unsigned int to match %02x format specifier
    int result =
        snprintf(uuid_str, sizeof(uuid_str), "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                 (unsigned int)uuid_bytes[0], (unsigned int)uuid_bytes[1], (unsigned int)uuid_bytes[2],
                 (unsigned int)uuid_bytes[3], (unsigned int)uuid_bytes[4], (unsigned int)uuid_bytes[5],
                 (unsigned int)uuid_bytes[6], (unsigned int)uuid_bytes[7], (unsigned int)uuid_bytes[8],
                 (unsigned int)uuid_bytes[9], (unsigned int)uuid_bytes[10], (unsigned int)uuid_bytes[11],
                 (unsigned int)uuid_bytes[12], (unsigned int)uuid_bytes[13], (unsigned int)uuid_bytes[14],
                 (unsigned int)uuid_bytes[15]);
    if (result < 0 || result >= (int)sizeof(uuid_str))
    {
        return "unknown";
    }
    return std::string(uuid_str);
}

/**
 * @brief Check if a pointer is a valid event handle within our circular buffers.
 *
 * Verifies if a pointer falls within the address range of any circular buffer
 * and is properly aligned to an event handle boundary.
 *
 * @param[in] commState Communicator state containing the circular buffers.
 * @param[in] ptr Pointer to check.
 *
 * @return Pointer to the event handle if valid, nullptr otherwise.
 *
 * @note Used by the first-child mechanism (ProxyOp/KernelCh stop) and by getParentChain (TRACE).
 */
static otelEventHandle_t* findEventHandleInBuffers(CommunicatorState* commState, void* ptr)
{
    if (!commState || !ptr || !commState->buffers) return NULL;

    for (int i = 0; i < NUM_BUFFERS; ++i)
    {
        otelEventHandle_t* buffer = commState->buffers[i];
        if (!buffer) continue;

        uintptr_t buffer_start = (uintptr_t)buffer;
        uintptr_t buffer_end   = buffer_start + (BUFFER_SIZE * sizeof(otelEventHandle_t));
        uintptr_t ptr_addr     = (uintptr_t)ptr;

        if (ptr_addr >= buffer_start && ptr_addr < buffer_end)
        {
            if ((ptr_addr - buffer_start) % sizeof(otelEventHandle_t) == 0)
            {
                return (otelEventHandle_t*)ptr;
            }
        }
    }

    return NULL;
}

#ifdef PROFILER_OTEL_ENABLE_TRACE
/**
 * @brief Get human-readable name for an NCCL event type.
 *
 * @param[in] type NCCL event type code (ncclProfileGroup, ncclProfileColl, etc.).
 *
 * @return String name of the event type, or "Unknown" if type is unrecognized.
 */
static const char* getEventTypeName(uint64_t type)
{
    switch (type)
    {
        case 1:  // ncclProfileGroup
            return "Group";
        case 2:  // ncclProfileColl
            return "Coll";
        case 4:  // ncclProfileP2p
            return "P2p";
        case 8:  // ncclProfileProxyOp
            return "ProxyOp";
        case 16:  // ncclProfileProxyStep
            return "ProxyStep";
        case 32:  // ncclProfileProxyCtrl
            return "ProxyCtrl";
        case 64:  // ncclProfileKernelCh
            return "KernelCh";
        case 1024:  // ncclProfileP2pApi
            return "P2pApi";
        case 2048:  // ncclProfileKernelLaunch
            return "KernelLaunch";
        default:
            return "Unknown";
    }
}

/**
 * @brief Build a human-readable string representation of the parent object chain.
 *
 * Traverses the parentObj chain and builds a string like "Group@0x123->Coll@0x456".
 * Used for debug logging to show event hierarchy.
 *
 * @param[in] parentObj Starting parent object pointer (may be nullptr).
 * @param[in] commState Communicator state for buffer validation.
 *
 * @return Static thread-local buffer containing the chain string. Overwritten on each call.
 *
 * @note Thread-safe due to thread-local storage.
 * @note Maximum chain depth is limited to prevent infinite loops.
 */
static const char* getParentChain(void* parentObj, CommunicatorState* commState)
{
    static __thread char chainBuffer[512];
    char tempBuffer[128];

    if (parentObj == NULL)
    {
        snprintf(chainBuffer, sizeof(chainBuffer), "(nil)");
        return chainBuffer;
    }

    chainBuffer[0]      = '\0';
    int depth           = 0;
    const int MAX_DEPTH = 5;  // Prevent infinite loops

    void* currentObj = parentObj;

    while (currentObj != NULL && depth < MAX_DEPTH)
    {
        // Try to find this object in our event buffers
        otelEventHandle_t* eventHandle = commState ? findEventHandleInBuffers(commState, currentObj) : NULL;

        if (eventHandle && eventHandle->type > 0)
        {
            // Found it! Build the chain entry with type name
            if (depth > 0)
            {
                strncat(chainBuffer, "->", sizeof(chainBuffer) - strlen(chainBuffer) - 1);
            }
            snprintf(tempBuffer, sizeof(tempBuffer), "%s@%p", getEventTypeName(eventHandle->type), currentObj);
            strncat(chainBuffer, tempBuffer, sizeof(chainBuffer) - strlen(chainBuffer) - 1);

            // Try to go up the chain
            currentObj = eventHandle->parentObj;
        }
        else
        {
            // Not one of our event handles, just show the pointer
            if (depth > 0)
            {
                strncat(chainBuffer, "->", sizeof(chainBuffer) - strlen(chainBuffer) - 1);
            }
            snprintf(tempBuffer, sizeof(tempBuffer), "0x%p", currentObj);
            strncat(chainBuffer, tempBuffer, sizeof(chainBuffer) - strlen(chainBuffer) - 1);
            break;  // Can't traverse further
        }

        depth++;
    }

    if (depth >= MAX_DEPTH && currentObj != NULL)
    {
        strncat(chainBuffer, "->...", sizeof(chainBuffer) - strlen(chainBuffer) - 1);
    }

    // If we didn't find anything, just show the pointer
    if (chainBuffer[0] == '\0')
    {
        snprintf(chainBuffer, sizeof(chainBuffer), "0x%p", parentObj);
    }

    return chainBuffer;
}
#else
// When TRACE is disabled, provide stubs that return simple values.
// These are marked unused since they're only referenced in OTEL_TRACE calls which become no-ops.
static __attribute__((unused)) const char* getEventTypeName(uint64_t type)
{
    (void)type;  // Suppress unused parameter warning
    return "Unknown";
}

static __attribute__((unused)) const char* getParentChain(void* parentObj, CommunicatorState* commState)
{
    (void)commState;  // Suppress unused parameter warning
    static __thread char chainBuffer[32];
    snprintf(chainBuffer, sizeof(chainBuffer), "%p", parentObj);
    return chainBuffer;
}
#endif

static int initialized;  // initialization counter for profiler
static double startTime;

ncclDebugLogger_t otel_log_func = nullptr;
static pthread_mutex_t otelLock = PTHREAD_MUTEX_INITIALIZER;
static pid_t pid;

// Atomic counters for telemetry init/cleanup management
static int telemetry_initialized =
    0;                                // Track if telemetry has been initialized (0 = not initialized, >0 = initialized)
static int active_communicators = 0;  // Track number of active communicators

// Test interface functions for unit testing
#ifdef UNIT_TESTING
int getInitialized()
{
    return initialized;
}
void setInitialized(int value)
{
    initialized = value;
}
double getStartTime()
{
    return startTime;
}
void setStartTime(double value)
{
    startTime = value;
}
pid_t getPid()
{
    return pid;
}
void setPid(pid_t value)
{
    pid = value;
}

// Test wrapper for ncclTypeSize (static function)
size_t test_ncclTypeSize(const char* datatype)
{
    return ncclTypeSize(datatype);
}

// Test wrapper for gpuUuidToString that takes raw bytes (avoids GPU platform dependency)
std::string test_gpuUuidToString(const unsigned char* uuid_bytes)
{
    gpuUUID_t uuid;
    memcpy(uuid.bytes, uuid_bytes, 16);
    return gpuUuidToString(uuid);
}
#endif  // UNIT_TESTING

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
// DESCRIPTION: Window timeout used for time-based window closing (seconds). Kept in this TU so unit tests do not need
// to link telemetry.cc.
OTEL_PARAM(WindowTimeoutIntervalSec, "PROFILER_OTEL_TELEMETRY_INTERVAL_SEC", 5);
// PARAM: LinearRegressionMode
// ENV: NCCL_PROFILER_LINEAR_REGRESSION_MODE
// DEFAULT: MIN
// DESCRIPTION: Linear regression mode for latency/rate estimation. Supported: MIN, AVG.
OTEL_STRING_PARAM(LinearRegressionMode, "PROFILER_LINEAR_REGRESSION_MODE", "MIN");
// PARAM: ScaleUpNetworkPct
// ENV: NCCL_PROFILER_OTEL_SCALEUP_NETWORK_PCT
// DEFAULT: 100
// DESCRIPTION: Estimated percentage of collective time spent on networking for scale-up inference (1-100).
OTEL_PARAM(ScaleUpNetworkPct, "PROFILER_OTEL_SCALEUP_NETWORK_PCT", 100);

/**
 * @brief Fallback logging function when NCCL logger is not available.
 *
 * Provides basic printf-style logging when the NCCL-provided logger is not available
 * during plugin initialization.
 *
 * @param[in] level Log level (WARN, INFO, TRACE, etc.).
 * @param[in] flags Log flags (unused, reserved for future use).
 * @param[in] file Source file name where log was called.
 * @param[in] line Line number where log was called.
 * @param[in] fmt Printf-style format string.
 * @param[in] ... Variable arguments for format string.
 */
static void fallback_log(ncclDebugLogLevel level, unsigned long flags, const char* file, int line, const char* fmt, ...)
{
    (void)flags;  // Cast to void to indicate intentional non-use

    const char* level_str = "";
    switch (level)
    {
        case NCCL_LOG_WARN:
            level_str = "WARN";
            break;
        case NCCL_LOG_INFO:
            level_str = "INFO";
            break;
        case NCCL_LOG_TRACE:
            level_str = "TRACE";
            break;
        default:
            level_str = "DEBUG";
            break;
    }

    printf("[PROFILER/OTEL_%s] %s:%d ", level_str, file, line);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
}

#ifndef UNIT_TESTING
/**
 * @brief Get current monotonic time in microseconds.
 *
 * @return Current time in microseconds since an unspecified epoch.
 *
 * @note Uses CLOCK_MONOTONIC for consistent timing across system clock adjustments.
 */
OTEL_HIDDEN double gettime(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1e6 + t.tv_nsec * 1e-3);
}
#endif

// NCCL Profiler Plugin v5

/**
 * @brief Initialize the profiler plugin.
 *
 * This function is called by NCCL to initialize the profiler plugin for a communicator.
 * It creates the plugin context, initializes circular buffers, and sets up telemetry
 * collection if enabled.
 *
 * @param[out] context Pointer to store the plugin context. Set to nullptr if plugin is disabled.
 * @param[in] commId Communicator unique identifier.
 * @param[out] eActivationMask Pointer to store the event activation mask.
 * @param[in] commName Name of the NCCL communicator (for identification in metrics).
 * @param[in] nNodes Number of nodes in the communicator.
 * @param[in] nranks Total number of ranks in the communicator.
 * @param[in] rank Rank of the current process.
 * @param[in] logfn NCCL logging function for plugin logging.
 *
 * @return ncclSuccess on success, ncclError on failure.
 *
 * @note The plugin can be disabled via NCCL_PROFILER_OTEL_ENABLE=0 environment variable.
 * @note Telemetry is initialized on the first communicator initialization.
 */
OTEL_HIDDEN ncclResult_t profiler_otel_init_v5(void** context, uint64_t commId, int* eActivationMask,
                                               const char* commName, int nNodes, int nranks, int rank,
                                               ncclDebugLogger_t logfn)
{
    // Store the log function provided by NCCL
    otel_log_func = logfn ? logfn : fallback_log;

    OTEL_TRACE(NCCL_INIT, "Plugin initialized(commName=%s, nNodes=%d, nranks=%d, rank=%d)", commName, nNodes, nranks,
               rank);

    int enable = OTEL_GET_PARAM(EnableOTEL);
    OTEL_TRACE(NCCL_INIT, "Checking enable parameter: NCCL_PROFILER_OTEL_ENABLE=%d", enable);
    if (enable == 0)
    {
        OTEL_WARN(NCCL_INIT, "Plugin disabled by environment variable NCCL_PROFILER_OTEL_ENABLE=0");
        *context = nullptr;  // Explicitly set context to nullptr when disabled
        return ncclSuccess;
    }

    static int localActivationMask = 0;  // Will hold the event mask determined by the first call

    pthread_mutex_lock(&otelLock);
    if (__atomic_fetch_add(&initialized, 1, __ATOMIC_RELAXED) == 0)
    {
        // First thread/process calculates the activation mask
        int64_t envMask = OTEL_GET_PARAM(ProfileEventMask);
        // Default mask: enable telemetry-relevant events
        // 0x85E = Coll + P2p + ProxyOp + ProxyStep + KernelCh + KernelLaunch
        localActivationMask = (envMask >= 0) ? (int)envMask : 0x85E;

        OTEL_INFO(NCCL_INIT, "Event activation mask set to 0x%x", localActivationMask);

        // Pid of the process initializing the plugin first.
        pid = getpid();

        startTime = gettime();
    }
    // Always assign the actual event mask to the caller's pointer
    __atomic_store_n(eActivationMask, localActivationMask, __ATOMIC_RELAXED);

    pthread_mutex_unlock(&otelLock);

    struct eventContext* ctx = (struct eventContext*)calloc(1, sizeof(*ctx));
    ctx->commName            = commName;
    ctx->commHash            = commId;
    ctx->nNodes              = nNodes;
    ctx->nranks              = nranks;
    ctx->rank                = rank;

    // Create circular buffer state for this communicator
    ctx->commState            = new CommunicatorState();
    ctx->commState->comm_name = commName;
    ctx->commState->comm_hash = commId;
    ctx->commState->nNodes    = nNodes;
    ctx->commState->nranks    = nranks;
    ctx->commState->rank      = rank;
    ctx->commState->commName  = commName ? std::string(commName) : std::string("");

    // Get hostname
    char hostname_buf[256];
    if (gethostname(hostname_buf, sizeof(hostname_buf)) == 0)
    {
        ctx->commState->hostname = std::string(hostname_buf);
    }
    else
    {
        ctx->commState->hostname = "unknown";
    }

    // Get GPU PCI BUS ID and UUID from GPU runtime (CUDA or ROCm/HIP)
    // Note: GPU runtime should be initialized by NCCL/RCCL at this point, but handle errors gracefully
    int gpu_device_id  = -1;
    gpuError_t gpu_err = gpuGetDevice(&gpu_device_id);
    if (gpu_err == gpuSuccess && gpu_device_id >= 0)
    {
        // Get device properties
        gpuDeviceProp device_prop;
        gpu_err = gpuGetDeviceProperties(&device_prop, gpu_device_id);
        if (gpu_err == gpuSuccess)
        {
            // Get PCI BUS ID as string
            char pci_bus_id_str[256];
            gpu_err = gpuDeviceGetPCIBusId(pci_bus_id_str, sizeof(pci_bus_id_str), gpu_device_id);
            if (gpu_err == gpuSuccess)
            {
                ctx->commState->gpu_pci_bus_id = std::string(pci_bus_id_str);
            }
            else
            {
                ctx->commState->gpu_pci_bus_id = "unknown";
                OTEL_WARN(NCCL_INIT, "Failed to get PCI Bus ID for device %d: %s", gpu_device_id,
                          gpuGetErrorString(gpu_err));
            }

            // Convert UUID to string format
#if defined(GPU_PLATFORM_ROCM)
            // ROCm/HIP: Copy UUID bytes from hipDeviceProp_t
            // HIP stores UUID differently - use gcnArchName or hdpMemFlushCntl as fallback
            // For now, construct a pseudo-UUID from device properties
            gpuUUID_t hip_uuid;
            memset(&hip_uuid, 0, sizeof(hip_uuid));
            // Use PCI bus ID as a unique identifier since HIP UUID handling varies by version
            if (ctx->commState->gpu_pci_bus_id != "unknown")
            {
                // Hash the PCI bus ID into UUID bytes
                const char* pci_str = ctx->commState->gpu_pci_bus_id.c_str();
                for (size_t i = 0; i < 16 && pci_str[i] != '\0'; ++i)
                {
                    hip_uuid.bytes[i] = pci_str[i];
                }
            }
            ctx->commState->gpu_uuid = gpuUuidToString(hip_uuid);
#else
            // CUDA: Use the UUID from device properties directly
            ctx->commState->gpu_uuid = gpuUuidToString(device_prop.uuid);
#endif

            OTEL_TRACE(NCCL_INIT, GPU_PLATFORM_NAME " device: id=%d, PCI_BUS_ID=%s, UUID=%s", gpu_device_id,
                       ctx->commState->gpu_pci_bus_id.c_str(), ctx->commState->gpu_uuid.c_str());
        }
        else
        {
            ctx->commState->gpu_pci_bus_id = "unknown";
            ctx->commState->gpu_uuid       = "unknown";
            OTEL_WARN(NCCL_INIT, "Failed to get " GPU_PLATFORM_NAME " device properties for device %d: %s",
                      gpu_device_id, gpuGetErrorString(gpu_err));
        }
    }
    else
    {
        // GPU runtime may not be initialized or available - this is acceptable for some configurations
        ctx->commState->gpu_pci_bus_id = "unknown";
        ctx->commState->gpu_uuid       = "unknown";
        OTEL_TRACE(NCCL_INIT,
                   GPU_PLATFORM_NAME " device not available: %s (this may be normal if GPU runtime is not initialized)",
                   gpuGetErrorString(gpu_err));
    }

    // Determine local_rank (GPU index within the node)
    // Strategy:
    // - COLLECTIVE comms (nranks > 2): rank == GPU index, cache in GPU ID → rank map
    // - P2P comms (nranks == 2): look up from GPU ID → rank map
    // - Fallback: use provided rank

    if (nranks > 2)
    {
        // COLLECTIVE communicator: rank is the GPU index
        // Cache GPU ID → rank mapping for P2P communicators
        ctx->commState->local_rank = rank;
        ctx->commState->comm_type  = CommunicatorState::CommType::COLLECTIVE;
        if (!ctx->commState->gpu_pci_bus_id.empty() && ctx->commState->gpu_pci_bus_id != "unknown")
        {
            std::lock_guard<std::mutex> lock(g_gpu_rank_map_mutex);
            g_gpu_id_to_rank[ctx->commState->gpu_pci_bus_id] = rank;
            OTEL_TRACE(NCCL_INIT, "COLLECTIVE: Cached GPU %s → rank %d", ctx->commState->gpu_pci_bus_id.c_str(), rank);
        }
        OTEL_TRACE(NCCL_INIT, "COLLECTIVE (nranks=%d): local_rank = rank = %d", nranks, ctx->commState->local_rank);
    }
    else
    {
        // P2P communicator (nranks == 2): always classified as P2P regardless of context.
        // This covers both pipeline-parallel communicators in LLM training and standalone
        // send/recv scenario.  The GPU→rank map is consulted only for local_rank resolution
        // (which GPU slot this rank occupies within the node); comm_type is set unconditionally.
        ctx->commState->comm_type = CommunicatorState::CommType::P2P;

        bool found = false;
        if (!ctx->commState->gpu_pci_bus_id.empty() && ctx->commState->gpu_pci_bus_id != "unknown")
        {
            std::lock_guard<std::mutex> lock(g_gpu_rank_map_mutex);
            auto it = g_gpu_id_to_rank.find(ctx->commState->gpu_pci_bus_id);
            if (it != g_gpu_id_to_rank.end())
            {
                ctx->commState->local_rank = it->second;
                found                      = true;
                OTEL_TRACE(NCCL_INIT, "P2P: Found GPU %s → rank %d from map", ctx->commState->gpu_pci_bus_id.c_str(),
                           ctx->commState->local_rank);
            }
        }

        if (!found)
        {
            // Map not yet populated (e.g. standalone sendrecv test with no prior collective comm).
            // Fall back to the provided rank as best estimate for local_rank.
            ctx->commState->local_rank = rank;
            OTEL_TRACE(NCCL_INIT, "P2P: GPU ID not in map, using rank=%d as local_rank (GPU=%s)", rank,
                       ctx->commState->gpu_pci_bus_id.c_str());
        }
    }

    // Set window timeout from telemetry interval (convert seconds to microseconds)
    int interval_sec = (int)OTEL_GET_PARAM(WindowTimeoutIntervalSec);
    if (interval_sec <= 0) interval_sec = 5;
    ctx->commState->window_timeout_usec = interval_sec * 1e6;
    OTEL_INFO(NCCL_INIT, "Window timeout set to %d seconds (%.0f us)", interval_sec,
              ctx->commState->window_timeout_usec);

    OTEL_INFO(
        NCCL_INIT,
        "Created communicator state: name=%s, hash=%lu, rank=%d, nranks=%d, nNodes=%d, hostname=%s, local_rank=%d, "
        "gpu_pci_bus_id=%s, gpu_uuid=%s, comm_type=%s",
        commName, commId, rank, nranks, nNodes, ctx->commState->hostname.c_str(), ctx->commState->local_rank,
        ctx->commState->gpu_pci_bus_id.c_str(), ctx->commState->gpu_uuid.c_str(), ctx->commState->getCommTypeString());

    *context = ctx;

    // Start telemetry only on first communicator initialization
    if (__atomic_fetch_add(&telemetry_initialized, 1, __ATOMIC_RELAXED) == 0)
    {
        profiler_otel_telemetry_init();
    }

    // Increment active communicator count
    __atomic_fetch_add(&active_communicators, 1, __ATOMIC_RELAXED);

    return ncclSuccess;
}

/**
 * @brief Start a new profiling event.
 *
 * Called by NCCL when a profiled event starts (Coll, P2P, ProxyOp, ProxyStep, Group,
 * KernelLaunch, KernelCh).
 * Allocates an event handle from the circular buffer and initializes event data.
 *
 * @param[in] context Plugin context from profiler_otel_init_v5().
 * @param[out] eHandle Pointer to store the event handle. Set to nullptr if event is filtered.
 * @param[in] eDescr Event descriptor containing event type and type-specific data.
 *
 * @return ncclSuccess on success (even if event is filtered).
 *
 * @note Events are filtered based on type (ProxyCtrl, receive ProxyOps are skipped).
 * @note P2P Recv events are skipped (only Send is tracked).
 * @note Event handle is allocated from a lock-free circular buffer.
 */
OTEL_HIDDEN ncclResult_t profiler_otel_start_event_v5(void* context, void** eHandle,
                                                      ncclProfilerEventDescr_v5_t* eDescr)
{
    struct eventContext* ctx = (struct eventContext*)context;
    *eHandle                 = NULL;

    uint64_t type = eDescr->type;

    // Filter out event types that are not used in telemetry aggregation.
    // Note: Group events must still return a valid handle (even if not used for metrics),
    // because NCCL may use Group as part of the parent chain for subsequent events.
    if (type == ncclProfileProxyCtrl)
    {
        OTEL_TRACE(NCCL_INIT, "Skipping ProxyCtrl event (not used in telemetry aggregation)");
        return ncclSuccess;
    }

    // Only process send direction for proxyOp, skip receive
    if (type == ncclProfileProxyOp && !eDescr->proxyOp.isSend)
    {
        OTEL_TRACE(NCCL_INIT, "Skipping receive proxyOp (only processing send direction)");
        return ncclSuccess;
    }

    // Skip receive ProxySteps - they have NULL parentObj because we don't track receive ProxyOps
    // Only send ProxySteps have parentObj pointing to their send ProxyOp
    if (type == ncclProfileProxyStep && eDescr->parentObj == NULL)
    {
        OTEL_TRACE(NCCL_INIT, "Skipping receive ProxyStep (parentObj is NULL, only tracking send direction)");
        return ncclSuccess;
    }

    // Skip P2P Recv operations - we filter out receive ProxyOps, so P2P Recv would have no
    // ProxyOp children and result in 0 duration (lastProxyOpEnd == startTs)
    if (type == ncclProfileP2p && eDescr->p2p.func && strstr(eDescr->p2p.func, "Recv") != nullptr)
    {
        OTEL_TRACE(NCCL_INIT, "Skipping P2P Recv event (only tracking send direction)");
        return ncclSuccess;
    }

    // Skip KernelCh events with NULL parentObj.
    //
    // NCCL sets eDescr->parentObj = sub->taskEventHandle (the profiler handle for the P2P
    // or Coll task that owns this GPU channel).  For P2P Recv sub-operations our plugin
    // returns NULL eHandle (we only track the send direction), so taskEventHandle == NULL,
    // and the resulting KernelCh arrives with parentObj == NULL.
    //
    // A null-parent KernelCh:
    //  - contributes nothing to aggregation (no parent Coll/P2P to link to), and
    //  - is NOT counted in in_progress_count, so the window can transition to PROCESSING
    //    before NCCL calls stop_event for it, producing a spurious "endTs=0" WARN.
    //
    // Filtering it here is symmetric with the Recv P2P / Recv ProxyOp filters above.
    if (type == ncclProfileKernelCh && eDescr->parentObj == NULL)
    {
        OTEL_TRACE(NCCL_INIT, "Skipping KernelCh with NULL parentObj (Recv sub-op, no aggregation value)");
        return ncclSuccess;
    }

    // Skip API-level and plugin events that NCCL sends as parents of our requested events.
    // NCCL uses a combined mask: if any child event type is active (e.g., ncclProfileColl),
    // the parent API event (ncclProfileCollApi) is also sent to the plugin regardless of
    // whether the plugin requested it. These waste circular buffer slots and produce
    // zombie events (endTs=0) if not filtered here before allocation.
    //
    // Exception: ncclProfileP2pApi is NOT filtered — it carries the original collective
    // function name (e.g., "AlltoAll") and serves as the grouping parent for P2P tasks
    // that are decomposed from a collective call.  Tracking it allows the aggregator to
    // synthesize a single Collective metric entry for AlltoAll operations.
    if (type == ncclProfileGroupApi || type == ncclProfileCollApi || type == ncclProfileNetPlugin)
    {
        return ncclSuccess;
    }

    // CRITICAL: Check window closing BEFORE allocation to avoid allocating on a closed window
    // Get current buffer index and time
    uint8_t buffer_idx  = ctx->commState->get_active_buffer_idx();
    double current_time = gettime() - startTime;

    // Check if we should close the current window (10k elements or timeout from telemetry interval)
    // This must happen BEFORE allocation to ensure we allocate from the correct (new) window
    if (ctx->commState->should_close_window(buffer_idx, current_time))
    {
        OTEL_TRACE(NCCL_INIT, "Triggering window closing for buffer %u before allocation", buffer_idx);
        ctx->commState->trigger_window_closing(buffer_idx);
        // Re-read buffer_idx in case it switched (if in_progress == 0)
        buffer_idx = ctx->commState->get_active_buffer_idx();
    }

    // Get a new event handle from the circular buffer
    // Pass parentObj so allocation can route to correct window (parent's window if parent exists)
    // Pass current_time for time-based window closing checks
    otelEventHandle_t* otel_event = get_next_event_handle(ctx->commState, eDescr->parentObj, current_time);
    if (!otel_event)
    {
        OTEL_WARN(NCCL_INIT, "Failed to get event handle from circular buffer. Dropping event.");
        return ncclSuccess;
    }

    // buffer_idx is already set by allocate_event_slot() based on parent window routing
    buffer_idx = otel_event->buffer_idx;

    // Fill in common fields
    otel_event->type = type;
    // buffer_idx already set by allocate_event_slot()
    otel_event->parentObj = eDescr->parentObj;
    otel_event->commState = ctx->commState;  // Back-pointer for mark_operation_complete() on stop
    otel_event->rank      = ctx->rank;
    otel_event->startTs   = current_time;
    otel_event->endTs     = 0;  // Will be set in stop_event

    // Set window start time on first event (for time-based window closing)
    ctx->commState->set_window_start_time_if_needed(buffer_idx, otel_event->startTs);

    // Fill in type-specific fields
    if (type == ncclProfileColl)
    {
        otel_event->coll.seqNumber           = eDescr->coll.seqNumber;
        otel_event->coll.func                = eDescr->coll.func;
        otel_event->coll.bytes               = ncclTypeSize(eDescr->coll.datatype) * eDescr->coll.count;
        otel_event->coll.nChannels           = eDescr->coll.nChannels;
        otel_event->coll.algo                = eDescr->coll.algo;
        otel_event->coll.proto               = eDescr->coll.proto;
        otel_event->coll.firstChildCompleted = false;

        // Increment in_progress_count by 1 for this Coll.
        // The matching -1 comes from the first child (ProxyOp or KernelCh) to
        // complete for this Coll. If neither arrives, force-processing handles it.
        ctx->commState->mark_operation_start(buffer_idx);
        ctx->commState->windows[buffer_idx].pending_first_child.fetch_add(1, std::memory_order_acq_rel);

        OTEL_TRACE(NCCL_INIT,
                   "Started Coll [eHandle=%p], parentChain=%s: %s, bytes=%zu, algo=%s, proto=%s, channels=%d (pending "
                   "ops+=1)",
                   otel_event, getParentChain(eDescr->parentObj, ctx->commState), eDescr->coll.func,
                   otel_event->coll.bytes, eDescr->coll.algo, eDescr->coll.proto, eDescr->coll.nChannels);
    }
    else if (type == ncclProfileP2p)
    {
        otel_event->p2p.func                = eDescr->p2p.func;
        otel_event->p2p.bytes               = ncclTypeSize(eDescr->p2p.datatype) * eDescr->p2p.count;
        otel_event->p2p.peer                = eDescr->p2p.peer;
        otel_event->p2p.nChannels           = eDescr->p2p.nChannels;
        otel_event->p2p.firstChildCompleted = false;

        // Only mark P2P Send operations as in-progress for window management.
        // P2P Recv events are filtered at the top of start_event (eHandle=NULL),
        // so they never reach stop_event and would never get mark_complete.
        bool isSend = eDescr->p2p.func && (strstr(eDescr->p2p.func, "Send") != nullptr);

        // AlltoAll (and similar collectives) decomposes into P2P operations that include a
        // "self-send" where peer == rank.  Self-sends are handled by NCCL via a direct local
        // memory copy — no proxy thread is involved, so no ProxyOp or KernelCh child event
        // ever arrives.  If we count them in in_progress_count / pending_first_child, the
        // window will never drain and will remain stuck in CLOSING until the 5-second timeout
        // forces it to READY, silently discarding all buffered telemetry data.
        bool isSelfSend = isSend && (eDescr->p2p.peer == eDescr->rank);

        if (isSend && !isSelfSend)
        {
            // Increment in_progress_count by 1 for this P2P Send.
            // The matching -1 comes from the first child (ProxyOp or KernelCh) to complete.
            ctx->commState->mark_operation_start(buffer_idx);
            ctx->commState->windows[buffer_idx].pending_first_child.fetch_add(1, std::memory_order_acq_rel);
            OTEL_TRACE(
                NCCL_INIT,
                "Started P2P Send [eHandle=%p], parentChain=%s: %s, bytes=%zu, peer=%d, channels=%d (pending ops+=1)",
                otel_event, getParentChain(eDescr->parentObj, ctx->commState), eDescr->p2p.func, otel_event->p2p.bytes,
                eDescr->p2p.peer, eDescr->p2p.nChannels);
        }
        else if (isSelfSend)
        {
            // Self-send: local copy, no proxy child expected — do NOT touch in_progress counters.
            OTEL_TRACE(NCCL_INIT,
                       "Started P2P Send-to-self [eHandle=%p], parentChain=%s: %s, bytes=%zu, peer=%d, channels=%d "
                       "(self-send, no proxy child expected, not tracked in window)",
                       otel_event, getParentChain(eDescr->parentObj, ctx->commState), eDescr->p2p.func,
                       otel_event->p2p.bytes, eDescr->p2p.peer, eDescr->p2p.nChannels);
        }
        else
        {
            OTEL_TRACE(
                NCCL_INIT,
                "Started P2P Recv [eHandle=%p], parentChain=%s: %s, bytes=%zu, peer=%d, channels=%d (no pending ops)",
                otel_event, getParentChain(eDescr->parentObj, ctx->commState), eDescr->p2p.func, otel_event->p2p.bytes,
                eDescr->p2p.peer, eDescr->p2p.nChannels);
        }
    }
    else if (type == ncclProfileProxyOp)
    {
        if (eDescr->proxyOp.pid != pid)
        {
            OTEL_WARN(NCCL_INIT, "ncclProfileProxyOp - not in this process, eDescr->proxyOp.pid != pid");
            // Mark as invalid so it won't be processed
            return ncclSuccess;
        }

        otel_event->proxyOp.channelId = eDescr->proxyOp.channelId;
        otel_event->proxyOp.peer      = eDescr->proxyOp.peer;
        otel_event->proxyOp.chunkSize = eDescr->proxyOp.chunkSize;

        // Track ProxyOps in progress for this window (used to detect when all ProxyOps complete)
        ctx->commState->windows[buffer_idx].proxy_ops_in_progress.fetch_add(1, std::memory_order_acq_rel);

        // Increment in_progress_count when ProxyOp starts.
        // Each ProxyOp does +1 at start and -1 at stop. The first ProxyOp
        // completion for a parent Coll/P2P also does an extra -1 to balance
        // the Coll/P2P's +1 from start_event.
        if (otel_event->commState && otel_event->parentObj)
        {
            otel_event->commState->mark_operation_start(buffer_idx);
            OTEL_TRACE(
                NCCL_INIT,
                "Started ProxyOp [eHandle=%p], parentChain=%s: peer=%d, channel=%d, chunkSize=%d (pending ops+=1)",
                otel_event, getParentChain(eDescr->parentObj, ctx->commState), eDescr->proxyOp.peer,
                eDescr->proxyOp.channelId, eDescr->proxyOp.chunkSize);
        }
        else
        {
            OTEL_TRACE(NCCL_INIT, "Started ProxyOp [eHandle=%p], parentChain=%s: peer=%d, channel=%d, chunkSize=%d",
                       otel_event, getParentChain(eDescr->parentObj, ctx->commState), eDescr->proxyOp.peer,
                       eDescr->proxyOp.channelId, eDescr->proxyOp.chunkSize);
        }
    }
    else if (type == ncclProfileProxyStep)
    {
        otel_event->proxyStep.step        = eDescr->proxyStep.step;
        otel_event->proxyStep.transSize   = 0;      // Will be set in recordEventState with ProxyStepSendWait
        otel_event->proxyStep.sendWaitTs  = 0.0;    // Will be set when ProxyStepSendWait state occurs
        otel_event->proxyStep.hasSendWait = false;  // Will be set to true when SendWait state recorded

        OTEL_TRACE(NCCL_INIT, "Started ProxyStep [eHandle=%p], parentChain=%s: step=%d", otel_event,
                   getParentChain(eDescr->parentObj, ctx->commState), eDescr->proxyStep.step);
    }
    else if (type == ncclProfileGroup)
    {
        // Group events: required to return a valid handle, because they can appear in the parent chain.
        // We don't export metrics for Group events, but we do use them for window management.
        ctx->commState->mark_operation_start(buffer_idx);
        ctx->commState->windows[buffer_idx].groups_in_progress.fetch_add(1, std::memory_order_acq_rel);
        OTEL_TRACE(NCCL_INIT, "Started Group [eHandle=%p] (pending ops+=1, groups+=1)", otel_event);
    }
    // =========================================================================
    // Kernel event handling (scale-up monitoring)
    // KernelCh events participate in in_progress_count to ensure the window
    // waits for GPU kernel completion before processing. The first KernelCh
    // (or ProxyOp) to complete for a parent Coll/P2P does an extra decrement
    // to balance the parent's +1 via the shared firstChildCompleted flag.
    // =========================================================================
    else if (type == ncclProfileKernelCh)
    {
        otel_event->kernelCh.channelId   = eDescr->kernelCh.channelId;
        otel_event->kernelCh.pTimerStart = eDescr->kernelCh.pTimer;
        otel_event->kernelCh.pTimerStop  = 0;
        otel_event->kernelCh.hasStop     = false;

        // Track KernelCh in window lifecycle (similar to ProxyOps)
        ctx->commState->windows[buffer_idx].kernel_ch_in_progress.fetch_add(1, std::memory_order_acq_rel);

        if (otel_event->commState && otel_event->parentObj)
        {
            ctx->commState->mark_operation_start(buffer_idx);
            OTEL_TRACE(NCCL_INIT,
                       "Started KernelCh [eHandle=%p], parentChain=%s: channelId=%d, pTimerStart=%lu (pending ops+=1)",
                       otel_event, getParentChain(eDescr->parentObj, ctx->commState), eDescr->kernelCh.channelId,
                       (unsigned long)eDescr->kernelCh.pTimer);
        }
        else
        {
            OTEL_TRACE(NCCL_INIT, "Started KernelCh [eHandle=%p], parentChain=%s: channelId=%d, pTimerStart=%lu",
                       otel_event, getParentChain(eDescr->parentObj, ctx->commState), eDescr->kernelCh.channelId,
                       (unsigned long)eDescr->kernelCh.pTimer);
        }
    }
    else if (type == ncclProfileKernelLaunch)
    {
        OTEL_TRACE(NCCL_INIT, "Started KernelLaunch [eHandle=%p], parentChain=%s", otel_event,
                   getParentChain(eDescr->parentObj, ctx->commState));
    }
    else if (type == ncclProfileP2pApi)
    {
        // Store the original collective function name (e.g., "AlltoAll").
        // This event acts as a grouping anchor for the individual P2P Send tasks
        // that NCCL emits when decomposing a collective such as AlltoAll into P2P ops.
        // No window management (in_progress tracking) is needed for this marker event.
        otel_event->p2pApi.func = eDescr->p2pApi.func;
        OTEL_TRACE(NCCL_INIT, "Stored P2pApi marker [eHandle=%p], func=%s (AlltoAll collective grouping anchor)",
                   otel_event, eDescr->p2pApi.func ? eDescr->p2pApi.func : "NULL");
    }
    else
    {
        OTEL_WARN(NCCL_INIT, "Unsupported event type %lu for start event (ignoring)", (unsigned long)type);
        return ncclSuccess;
    }

    *eHandle = otel_event;

    return ncclSuccess;
}

/**
 * @brief Stop a profiling event.
 *
 * Called by NCCL when a profiled event completes. Records the end timestamp and
 * updates window management state for ProxyOp events.
 *
 * @param[in] eHandle Event handle from profiler_otel_start_event_v5().
 *
 * @return ncclSuccess on success.
 *
 * @note If eHandle is nullptr, the function returns successfully (event was filtered).
 * @note For ProxyOp events, this decrements the window's in_progress_count.
 */
OTEL_HIDDEN ncclResult_t profiler_otel_stop_event_v5(void* eHandle)
{
    if (eHandle == NULL)
    {
        OTEL_WARN(NCCL_INIT, "profiler_otel_stop_event skipped, eHandle is NULL");
        return ncclSuccess;
    }

    otelEventHandle_t* otel_event = (otelEventHandle_t*)eHandle;
    otel_event->endTs             = gettime() - startTime;

    // For Group events, mark operation complete for window management.
    if (otel_event->type == ncclProfileGroup)
    {
        if (otel_event->commState)
        {
            uint8_t buf_idx = otel_event->buffer_idx;
            otel_event->commState->mark_operation_complete(buf_idx);

            uint32_t prev_groups =
                otel_event->commState->windows[buf_idx].groups_in_progress.fetch_sub(1, std::memory_order_acq_rel);

            OTEL_TRACE(NCCL_INIT, "Stopped Group [eHandle=%p], duration=%.2f us (pending ops-=1, groups: %u->%u)",
                       eHandle, otel_event->endTs - otel_event->startTs, prev_groups, prev_groups - 1);

            // Force-processing: if the last group completes and all ProxyOps,
            // KernelCh, and pending first-child events are done, any remaining
            // in_progress is from truly orphaned Coll/P2P +1s.
            if (prev_groups == 1)
            {
                WindowMetadata* window       = &otel_event->commState->windows[buf_idx];
                WindowState state            = window->state.load(std::memory_order_acquire);
                uint32_t proxy_ops_pending   = window->proxy_ops_in_progress.load(std::memory_order_acquire);
                uint32_t kernel_ch_pending   = window->kernel_ch_in_progress.load(std::memory_order_acquire);
                uint32_t first_child_pending = window->pending_first_child.load(std::memory_order_acquire);
                uint32_t in_progress_count   = window->in_progress_count.load(std::memory_order_acquire);

                if (state == WINDOW_CLOSING && proxy_ops_pending == 0 && kernel_ch_pending == 0 &&
                    first_child_pending == 0 && in_progress_count > 0)
                {
                    OTEL_WARN(NCCL_INIT,
                              "Window %u: Last group completed, no pending ops, forcing PROCESSING "
                              "(%u orphaned in-progress ops)",
                              buf_idx, in_progress_count);
                    window->in_progress_count.store(0, std::memory_order_release);
                    WindowState closing_state = WINDOW_CLOSING;
                    if (window->state.compare_exchange_strong(closing_state, WINDOW_PROCESSING,
                                                              std::memory_order_acq_rel))
                    {
                        profiler_otel_telemetry_notify_window_ready(otel_event->commState, buf_idx);
                    }
                }
            }
        }
        else
        {
            OTEL_TRACE(NCCL_INIT, "Stopped Group [eHandle=%p], duration=%.2f us", eHandle,
                       otel_event->endTs - otel_event->startTs);
        }
    }
    // For ProxyStep with SendWait, show actual transfer time (SendWait -> Stop)
    else if (otel_event->type == ncclProfileProxyStep && otel_event->proxyStep.hasSendWait)
    {
        OTEL_TRACE(NCCL_INIT,
                   "Stopped %s [eHandle=%p], parentChain=%s, duration=%.2f us, TRANSFER: %.2f us (%zu bytes)",
                   getEventTypeName(otel_event->type), eHandle, getParentChain(otel_event->parentObj, NULL),
                   otel_event->endTs - otel_event->startTs, otel_event->endTs - otel_event->proxyStep.sendWaitTs,
                   otel_event->proxyStep.transSize);
    }
    // For ProxyOp, mark operation complete for window management
    else if (otel_event->type == ncclProfileProxyOp)
    {
        if (otel_event->commState && otel_event->parentObj)
        {
            // Decrement for the ProxyOp itself (1:1 with mark_start at ProxyOp start)
            otel_event->commState->mark_operation_complete(otel_event->buffer_idx);

            // Decrement proxy_ops_in_progress
            uint32_t prev_proxy_ops =
                otel_event->commState->windows[otel_event->buffer_idx].proxy_ops_in_progress.fetch_sub(
                    1, std::memory_order_acq_rel);

            // First-child mechanism: the first child (ProxyOp or KernelCh) to complete
            // for a parent Coll/P2P does an extra mark_complete to balance the parent's
            // +1 from start_event. Uses a shared firstChildCompleted flag so that only
            // one child (whichever finishes first) performs the extra decrement.
            otelEventHandle_t* parent_event = findEventHandleInBuffers(otel_event->commState, otel_event->parentObj);
            bool is_first                   = false;

            if (parent_event && (parent_event->type == ncclProfileColl || parent_event->type == ncclProfileP2p))
            {
                if (parent_event->type == ncclProfileColl)
                {
                    bool expected = false;
                    if (__atomic_compare_exchange_n(&parent_event->coll.firstChildCompleted, &expected, true, false,
                                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
                    {
                        is_first = true;
                    }
                }
                else if (parent_event->type == ncclProfileP2p)
                {
                    bool expected = false;
                    if (__atomic_compare_exchange_n(&parent_event->p2p.firstChildCompleted, &expected, true, false,
                                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
                    {
                        is_first = true;
                    }
                }
            }

            if (is_first)
            {
                // The parent Coll/P2P may have been allocated in a different window than
                // this ProxyOp if the window rotated between the parent start and child start.
                // Use the parent's buffer_idx for the parent-decrement so the counters of
                // the correct window are updated.
                uint8_t parent_buf_idx = parent_event ? parent_event->buffer_idx : otel_event->buffer_idx;
                otel_event->commState->mark_operation_complete(parent_buf_idx);
                otel_event->commState->windows[parent_buf_idx].pending_first_child.fetch_sub(1,
                                                                                             std::memory_order_acq_rel);
                OTEL_TRACE(
                    NCCL_INIT,
                    "Stopped ProxyOp [eHandle=%p], parentChain=%s, duration=%.2f us (pending ops-=2, first child, "
                    "parent_buf=%u self_buf=%u)",
                    eHandle, getParentChain(otel_event->parentObj, NULL), otel_event->endTs - otel_event->startTs,
                    parent_buf_idx, otel_event->buffer_idx);
            }
            else
            {
                OTEL_TRACE(NCCL_INIT, "Stopped ProxyOp [eHandle=%p], parentChain=%s, duration=%.2f us (pending ops-=1)",
                           eHandle, getParentChain(otel_event->parentObj, NULL),
                           otel_event->endTs - otel_event->startTs);
            }

            // Force-processing: if this was the last ProxyOp and all other tracked
            // operations (KernelCh, groups, pending first-child) are also done, any
            // remaining in_progress is from orphaned Coll/P2P +1s.
            if (prev_proxy_ops == 1)
            {
                uint8_t buf_idx              = otel_event->buffer_idx;
                WindowMetadata* window       = &otel_event->commState->windows[buf_idx];
                WindowState state            = window->state.load(std::memory_order_acquire);
                uint32_t groups_pending      = window->groups_in_progress.load(std::memory_order_acquire);
                uint32_t kernel_ch_pending   = window->kernel_ch_in_progress.load(std::memory_order_acquire);
                uint32_t first_child_pending = window->pending_first_child.load(std::memory_order_acquire);
                uint32_t in_progress_count   = window->in_progress_count.load(std::memory_order_acquire);

                if (state == WINDOW_CLOSING && groups_pending == 0 && kernel_ch_pending == 0 &&
                    first_child_pending == 0 && in_progress_count > 0)
                {
                    OTEL_WARN(NCCL_INIT,
                              "Window %u: Last ProxyOp completed, no pending ops/groups, forcing PROCESSING "
                              "(%u orphaned in-progress ops)",
                              buf_idx, in_progress_count);
                    window->in_progress_count.store(0, std::memory_order_release);
                    WindowState closing_state = WINDOW_CLOSING;
                    if (window->state.compare_exchange_strong(closing_state, WINDOW_PROCESSING,
                                                              std::memory_order_acq_rel))
                    {
                        profiler_otel_telemetry_notify_window_ready(otel_event->commState, buf_idx);
                    }
                }
            }
        }
        else
        {
            OTEL_TRACE(NCCL_INIT, "Stopped ProxyOp [eHandle=%p], parentChain=%s, duration=%.2f us", eHandle,
                       getParentChain(otel_event->parentObj, NULL), otel_event->endTs - otel_event->startTs);
        }
    }
    // For KernelCh, mark operation complete + first-child mechanism + force-processing
    else if (otel_event->type == ncclProfileKernelCh)
    {
        if (otel_event->commState && otel_event->parentObj)
        {
            // Decrement for the KernelCh itself (1:1 with mark_start at KernelCh start)
            otel_event->commState->mark_operation_complete(otel_event->buffer_idx);

            // Decrement kernel_ch_in_progress
            uint32_t prev_kernel_ch =
                otel_event->commState->windows[otel_event->buffer_idx].kernel_ch_in_progress.fetch_sub(
                    1, std::memory_order_acq_rel);

            // First-child mechanism: compete with ProxyOps for the extra decrement
            otelEventHandle_t* parent_event = findEventHandleInBuffers(otel_event->commState, otel_event->parentObj);
            bool is_first                   = false;

            if (parent_event && (parent_event->type == ncclProfileColl || parent_event->type == ncclProfileP2p))
            {
                if (parent_event->type == ncclProfileColl)
                {
                    bool expected = false;
                    if (__atomic_compare_exchange_n(&parent_event->coll.firstChildCompleted, &expected, true, false,
                                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
                    {
                        is_first = true;
                    }
                }
                else if (parent_event->type == ncclProfileP2p)
                {
                    bool expected = false;
                    if (__atomic_compare_exchange_n(&parent_event->p2p.firstChildCompleted, &expected, true, false,
                                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
                    {
                        is_first = true;
                    }
                }
            }

            if (is_first)
            {
                // The parent Coll/P2P may have been allocated in a different window than
                // this KernelCh if the window rotated between the parent start and child start.
                // Use the parent's buffer_idx for the parent-decrement so the counters of
                // the correct window are updated.
                uint8_t parent_buf_idx = parent_event ? parent_event->buffer_idx : otel_event->buffer_idx;
                otel_event->commState->mark_operation_complete(parent_buf_idx);
                otel_event->commState->windows[parent_buf_idx].pending_first_child.fetch_sub(1,
                                                                                             std::memory_order_acq_rel);
                OTEL_TRACE(
                    NCCL_INIT,
                    "Stopped KernelCh [eHandle=%p], parentChain=%s, duration=%.2f us (pending ops-=2, first child, "
                    "parent_buf=%u self_buf=%u)",
                    eHandle, getParentChain(otel_event->parentObj, NULL), otel_event->endTs - otel_event->startTs,
                    parent_buf_idx, otel_event->buffer_idx);
            }
            else
            {
                OTEL_TRACE(NCCL_INIT,
                           "Stopped KernelCh [eHandle=%p], parentChain=%s, duration=%.2f us (pending ops-=1)", eHandle,
                           getParentChain(otel_event->parentObj, NULL), otel_event->endTs - otel_event->startTs);
            }

            // Force-processing: if this was the last KernelCh and all other tracked
            // operations (ProxyOps, groups, pending first-child) are also done, any
            // remaining in_progress is from orphaned Coll/P2P +1s.
            if (prev_kernel_ch == 1)
            {
                uint8_t buf_idx              = otel_event->buffer_idx;
                WindowMetadata* window       = &otel_event->commState->windows[buf_idx];
                WindowState state            = window->state.load(std::memory_order_acquire);
                uint32_t groups_pending      = window->groups_in_progress.load(std::memory_order_acquire);
                uint32_t proxy_ops_pending   = window->proxy_ops_in_progress.load(std::memory_order_acquire);
                uint32_t first_child_pending = window->pending_first_child.load(std::memory_order_acquire);
                uint32_t in_progress_count   = window->in_progress_count.load(std::memory_order_acquire);

                if (state == WINDOW_CLOSING && groups_pending == 0 && proxy_ops_pending == 0 &&
                    first_child_pending == 0 && in_progress_count > 0)
                {
                    OTEL_WARN(NCCL_INIT,
                              "Window %u: Last KernelCh completed, no pending ops/groups, forcing PROCESSING "
                              "(%u orphaned in-progress ops)",
                              buf_idx, in_progress_count);
                    window->in_progress_count.store(0, std::memory_order_release);
                    WindowState closing_state = WINDOW_CLOSING;
                    if (window->state.compare_exchange_strong(closing_state, WINDOW_PROCESSING,
                                                              std::memory_order_acq_rel))
                    {
                        profiler_otel_telemetry_notify_window_ready(otel_event->commState, buf_idx);
                    }
                }
            }
        }
        else
        {
            OTEL_TRACE(NCCL_INIT, "Stopped KernelCh [eHandle=%p], parentChain=%s, duration=%.2f us", eHandle,
                       getParentChain(otel_event->parentObj, NULL), otel_event->endTs - otel_event->startTs);
        }
    }
    else
    {
        OTEL_TRACE(NCCL_INIT, "Stopped %s [eHandle=%p], parentChain=%s, duration=%.2f us",
                   getEventTypeName(otel_event->type), eHandle, getParentChain(otel_event->parentObj, NULL),
                   otel_event->endTs - otel_event->startTs);
    }

    return ncclSuccess;
}

/**
 * @brief Record event state transition.
 *
 * Called by NCCL to record state changes in ProxyStep and KernelCh events.
 *
 * @param[in] eHandle Event handle from profiler_otel_start_event_v5().
 * @param[in] eState Event state (e.g., ProxyStepSendWait, KernelChStop).
 * @param[in] eStateArgs State-specific arguments.
 *
 * @return ncclSuccess on success.
 *
 * @note For ProxyStep events, SendWait captures the timestamp used as the transfer start.
 * @note For KernelCh events, KernelChStop captures the GPU stop timestamp.
 */
OTEL_HIDDEN ncclResult_t profiler_otel_record_event_state_v5(void* eHandle, ncclProfilerEventState_v5_t eState,
                                                             ncclProfilerEventStateArgs_v5_t* eStateArgs)
{
    if (eHandle == NULL)
    {
        OTEL_WARN(NCCL_INIT, "profiler_otel_record_event_state skipped, eHandle is NULL");
        return ncclSuccess;
    }

    otelEventHandle_t* otel_event = (otelEventHandle_t*)eHandle;

    // Log the state transition based on event type
#ifdef PROFILER_OTEL_ENABLE_TRACE
    const char* state_name = "UNKNOWN";
    switch (eState)
    {
        // ProxyStep states
        case ncclProfilerProxyStepSendGPUWait:
            state_name = "ProxyStepSendGPUWait";
            break;
        case ncclProfilerProxyStepSendPeerWait_v4:
            state_name = "ProxyStepSendPeerWait";
            break;
        case ncclProfilerProxyStepSendWait:
            state_name = "ProxyStepSendWait";
            break;
        case ncclProfilerProxyStepRecvWait:
            state_name = "ProxyStepRecvWait";
            break;
        case ncclProfilerProxyStepRecvFlushWait:
            state_name = "ProxyStepRecvFlushWait";
            break;
        case ncclProfilerProxyStepRecvGPUWait:
            state_name = "ProxyStepRecvGPUWait";
            break;

        // ProxyOp states
        case ncclProfilerProxyOpInProgress_v4:
            state_name = "ProxyOpInProgress";
            break;

        // Kernel channel states
        case ncclProfilerKernelChStop:
            state_name = "KernelChStop";
            break;

        default:
            state_name = "UNKNOWN";
            break;
    }
#endif

    // ==========================================================================
    // KernelCh state handling: record GPU stop timestamp from KernelChStop
    // ==========================================================================
    if (otel_event->type == ncclProfileKernelCh && eState == ncclProfilerKernelChStop)
    {
        if (eStateArgs != NULL)
        {
            otel_event->kernelCh.pTimerStop = eStateArgs->kernelCh.pTimer;
            otel_event->kernelCh.hasStop    = true;
            OTEL_TRACE(NCCL_INIT, "RecordEventState [eHandle=%p] KernelCh: KernelChStop pTimer=%lu", eHandle,
                       (unsigned long)eStateArgs->kernelCh.pTimer);
        }
        return ncclSuccess;
    }

    // ==========================================================================
    // ProxyStep state handling (existing logic)
    // ==========================================================================
    if (eStateArgs != NULL)
    {
        if (otel_event->type == ncclProfileProxyStep && eStateArgs->proxyStep.transSize > 0)
        {
            otel_event->proxyStep.transSize = eStateArgs->proxyStep.transSize;

            // Capture timestamp when ProxyStepSendWait state occurs - this is the actual transfer start
            if (eState == ncclProfilerProxyStepSendWait)
            {
                otel_event->proxyStep.sendWaitTs  = gettime() - startTime;
                otel_event->proxyStep.hasSendWait = true;
                OTEL_TRACE(NCCL_INIT,
                           "RecordEventState [eHandle=%p] type=%lu, state=%s (%d), transSize=%zu [TRANSFER START @ "
                           "%.2f us]",
                           eHandle, (unsigned long)otel_event->type, state_name, eState,
                           eStateArgs->proxyStep.transSize, otel_event->proxyStep.sendWaitTs);
            }
            else
            {
                OTEL_TRACE(NCCL_INIT, "RecordEventState [eHandle=%p] type=%lu, state=%s (%d), transSize=%zu", eHandle,
                           (unsigned long)otel_event->type, state_name, eState, eStateArgs->proxyStep.transSize);
            }
        }
        else
        {
            OTEL_TRACE(NCCL_INIT, "RecordEventState [eHandle=%p] type=%lu, state=%s (%d), with args", eHandle,
                       (unsigned long)otel_event->type, state_name, eState);
        }
    }
    else
    {
        OTEL_TRACE(NCCL_INIT, "RecordEventState [eHandle=%p] type=%lu, state=%s (%d), no args", eHandle,
                   (unsigned long)otel_event->type, state_name, eState);
    }

    return ncclSuccess;
}

/**
 * @brief Finalize the profiler plugin for a communicator.
 *
 * Called by NCCL when a communicator is destroyed. Cleans up plugin context and
 * communicator state. Telemetry is cleaned up when the last communicator is finalized.
 *
 * @param[in] context Plugin context from profiler_otel_init_v5().
 *
 * @return ncclSuccess on success.
 *
 * @note Telemetry thread is stopped only when the last communicator is finalized.
 */
OTEL_HIDDEN ncclResult_t profiler_otel_finalize_v5(void* context)
{
    OTEL_TRACE(NCCL_INIT, "===> profiler_otel_finalize(context=%p)", context);
    struct eventContext* ctx = (struct eventContext*)context;

    // Destroy communicator state
    if (ctx && ctx->commState)
    {
        OTEL_INFO(NCCL_INIT, "Destroying communicator state: name=%s, hash=%lu", ctx->commState->comm_name,
                  ctx->commState->comm_hash);
        delete ctx->commState;
        ctx->commState = nullptr;
    }

    free(ctx);

    // Decrement active communicator count and cleanup telemetry only when last communicator is finalized
    int remaining = __atomic_sub_fetch(&active_communicators, 1, __ATOMIC_ACQ_REL);
    if (remaining == 0)
    {
        profiler_otel_telemetry_cleanup();
        // Reset telemetry_initialized so it can be re-initialized if needed
        __atomic_store_n(&telemetry_initialized, 0, __ATOMIC_RELEASE);
    }

    return ncclSuccess;
}
// end NCCL Profiler Plugin v5
