// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "profiler_gpu_metadata.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include "communicator_state.h"
#include "profiler_otel.h"

#if defined(GPU_PLATFORM_ROCM)
#include <hip/hip_runtime.h>

using gpuError_t    = hipError_t;
using gpuDeviceProp = hipDeviceProp_t;

struct gpuUUID_t
{
    char bytes[16];
};

#define gpuSuccess             hipSuccess
#define gpuGetDevice           hipGetDevice
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceGetPCIBusId   hipDeviceGetPCIBusId
#define gpuGetErrorString      hipGetErrorString
#define GPU_PLATFORM_NAME      "ROCm/HIP"

#else
#include <cuda_runtime.h>

using gpuError_t    = cudaError_t;
using gpuDeviceProp = cudaDeviceProp;
using gpuUUID_t     = cudaUUID_t;

#define gpuSuccess             cudaSuccess
#define gpuGetDevice           cudaGetDevice
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceGetPCIBusId   cudaDeviceGetPCIBusId
#define gpuGetErrorString      cudaGetErrorString
#define GPU_PLATFORM_NAME      "CUDA"

#endif  // GPU_PLATFORM_ROCM

static std::map<std::string, int> g_gpu_id_to_rank;
static std::mutex g_gpu_rank_map_mutex;

/**
 * @brief Convert a GPU UUID payload into the canonical string form.
 *
 * @param[in] uuid Raw GPU UUID structure.
 *
 * @return Lowercase UUID string, or "unknown" when formatting fails.
 */
static std::string gpuUuidToString(const gpuUUID_t& uuid)
{
    char uuid_str[64];
    const unsigned char* uuid_bytes = reinterpret_cast<const unsigned char*>(uuid.bytes);
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
 * @brief Populate GPU PCI bus ID and UUID metadata for a communicator state.
 *
 * @param[in,out] commState Communicator state to populate.
 */
void populateGpuMetadata(CommunicatorState* commState)
{
    int gpu_device_id  = -1;
    gpuError_t gpu_err = gpuGetDevice(&gpu_device_id);
    if (gpu_err == gpuSuccess && gpu_device_id >= 0)
    {
        gpuDeviceProp device_prop;
        gpu_err = gpuGetDeviceProperties(&device_prop, gpu_device_id);
        if (gpu_err == gpuSuccess)
        {
            char pci_bus_id_str[256];
            gpu_err = gpuDeviceGetPCIBusId(pci_bus_id_str, sizeof(pci_bus_id_str), gpu_device_id);
            if (gpu_err == gpuSuccess)
            {
                commState->gpu_pci_bus_id = std::string(pci_bus_id_str);
            }
            else
            {
                commState->gpu_pci_bus_id = "unknown";
                OTEL_WARN(NCCL_INIT, "Failed to get PCI Bus ID for device %d: %s", gpu_device_id,
                          gpuGetErrorString(gpu_err));
            }

#if defined(GPU_PLATFORM_ROCM)
            gpuUUID_t hip_uuid;
            memset(&hip_uuid, 0, sizeof(hip_uuid));
            if (commState->gpu_pci_bus_id != "unknown")
            {
                const char* pci_str = commState->gpu_pci_bus_id.c_str();
                for (size_t i = 0; i < 16 && pci_str[i] != '\0'; ++i)
                {
                    hip_uuid.bytes[i] = pci_str[i];
                }
            }
            commState->gpu_uuid = gpuUuidToString(hip_uuid);
#else
            commState->gpu_uuid = gpuUuidToString(device_prop.uuid);
#endif

            OTEL_TRACE(NCCL_INIT, GPU_PLATFORM_NAME " device: id=%d, PCI_BUS_ID=%s, UUID=%s", gpu_device_id,
                       commState->gpu_pci_bus_id.c_str(), commState->gpu_uuid.c_str());
            return;
        }

        commState->gpu_pci_bus_id = "unknown";
        commState->gpu_uuid       = "unknown";
        OTEL_WARN(NCCL_INIT, "Failed to get " GPU_PLATFORM_NAME " device properties for device %d: %s", gpu_device_id,
                  gpuGetErrorString(gpu_err));
        return;
    }

    commState->gpu_pci_bus_id = "unknown";
    commState->gpu_uuid       = "unknown";
    OTEL_TRACE(NCCL_INIT,
               GPU_PLATFORM_NAME " device not available: %s (this may be normal if GPU runtime is not initialized)",
               gpuGetErrorString(gpu_err));
}

/**
 * @brief Resolve the communicator local rank and communicator type.
 *
 * @param[in,out] commState Communicator state to populate.
 * @param[in] rank Global rank of the current process.
 * @param[in] nranks Number of ranks in the communicator.
 */
void resolveLocalRankAndCommType(CommunicatorState* commState, int rank, int nranks)
{
    if (nranks > 2)
    {
        commState->local_rank = rank;
        commState->comm_type  = CommunicatorState::CommType::COLLECTIVE;
        if (!commState->gpu_pci_bus_id.empty() && commState->gpu_pci_bus_id != "unknown")
        {
            std::lock_guard<std::mutex> lock(g_gpu_rank_map_mutex);
            g_gpu_id_to_rank[commState->gpu_pci_bus_id] = rank;
            OTEL_TRACE(NCCL_INIT, "COLLECTIVE: Cached GPU %s -> rank %d", commState->gpu_pci_bus_id.c_str(), rank);
        }
        OTEL_TRACE(NCCL_INIT, "COLLECTIVE (nranks=%d): local_rank = rank = %d", nranks, commState->local_rank);
        return;
    }

    commState->comm_type = CommunicatorState::CommType::P2P;

    bool found = false;
    if (!commState->gpu_pci_bus_id.empty() && commState->gpu_pci_bus_id != "unknown")
    {
        std::lock_guard<std::mutex> lock(g_gpu_rank_map_mutex);
        auto it = g_gpu_id_to_rank.find(commState->gpu_pci_bus_id);
        if (it != g_gpu_id_to_rank.end())
        {
            commState->local_rank = it->second;
            found                 = true;
            OTEL_TRACE(NCCL_INIT, "P2P: Found GPU %s -> rank %d from map", commState->gpu_pci_bus_id.c_str(),
                       commState->local_rank);
        }
    }

    if (!found)
    {
        commState->local_rank = rank;
        OTEL_TRACE(NCCL_INIT, "P2P: GPU ID not in map, using rank=%d as local_rank (GPU=%s)", rank,
                   commState->gpu_pci_bus_id.c_str());
    }
}

#ifdef UNIT_TESTING
/**
 * @brief Expose UUID string formatting to unit tests.
 *
 * @param[in] uuid_bytes Raw 16-byte UUID payload.
 *
 * @return Canonical UUID string for the supplied byte sequence.
 */
std::string test_gpuUuidToString(const unsigned char* uuid_bytes)
{
    gpuUUID_t uuid;
    memcpy(uuid.bytes, uuid_bytes, 16);
    return gpuUuidToString(uuid);
}
#endif  // UNIT_TESTING