// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef COMMUNICATOR_STATE_H_
#define COMMUNICATOR_STATE_H_

#include <atomic>
#include <cstdint>
#include <string>

#include "events.h"

#define BUFFER_SIZE             100000  // 100k elements per buffer
#define NUM_BUFFERS             4       // 4 circular buffers
#define WINDOW_TRIGGER_COUNT    50000   // Trigger window closing at 50k elements
#define WINDOW_TRIGGER_TIME_SEC 5       // Trigger window closing after 5 seconds

/**
 * Window states for the circular buffer state machine
 */
enum WindowState : uint8_t
{
    WINDOW_FILLING = 0,  // Window is actively being filled
    WINDOW_CLOSING,      // Window has reached trigger, closing in-progress operations
    WINDOW_PROCESSING,   // Window is being processed by background thread
    WINDOW_READY         // Window is cleared and ready to be reused
};

/**
 * Per-window metadata for tracking state and in-progress operations
 */
struct WindowMetadata
{
    std::atomic<WindowState> state;
    std::atomic<uint32_t> element_count;          // Number of elements in this window
    std::atomic<uint32_t> in_progress_count;      // Number of in-progress coll/p2p/transfers
    std::atomic<uint32_t> groups_in_progress;     // Number of in-progress Group operations
    std::atomic<uint32_t> proxy_ops_in_progress;  // Number of ProxyOps currently in-progress in this window
    std::atomic<uint32_t> kernel_ch_in_progress;  // Number of KernelCh events currently in-progress in this window
    std::atomic<uint32_t> pending_first_child;    // Coll/P2P events awaiting their first child (KernelCh/ProxyOp)
    double start_time;                            // Time when window started filling

    WindowMetadata()
        : state(WINDOW_READY),
          element_count(0),
          in_progress_count(0),
          groups_in_progress(0),
          proxy_ops_in_progress(0),
          kernel_ch_in_progress(0),
          pending_first_child(0),
          start_time(0.0)
    {
    }
};

/**
 * Communicator state managing circular buffers for event storage
 * Each communicator has its own set of 4 buffers that rotate independently
 *
 * Note: Buffers are heap-allocated to avoid stack overflow (~40 MB total)
 */
struct CommunicatorState
{
    // 4 circular buffer arrays (heap-allocated to avoid stack overflow)
    // Each buffer is NUM_BUFFERS x BUFFER_SIZE = 4 x 100k = 400k events
    // With ~100 bytes per event, this is ~40 MB total
    otelEventHandle_t** buffers;

    // Metadata for each window
    WindowMetadata windows[NUM_BUFFERS];

    // Active buffer index (0-3)
    std::atomic<uint8_t> active_buffer_idx;

    // Next element index within active buffer
    std::atomic<uint32_t> next_element_idx;

    // Active Group routing state.
    // While a Group is open, P2P sends that belong to its fan-out should stay in the
    // Group's window even if the active window rotates before those sends start.
    std::atomic<void*> active_group_handle;
    std::atomic<uint32_t> active_group_depth;

    // Communicator metadata
    const char* comm_name;
    uint64_t comm_hash;
    int rank;
    int nranks;
    int nNodes;

    // Rank and hostname information
    std::string hostname;        // Hostname of the node running this rank
    int local_rank;              // Local rank within the node
    std::string gpu_pci_bus_id;  // GPU PCI BUS ID (e.g., "00000000:01:00.0")
    std::string gpu_uuid;        // GPU UUID

    // Communicator type classification
    // P2P communicators always have exactly 2 ranks (point-to-point)
    // Collective communicators have more than 2 ranks
    enum class CommType
    {
        UNKNOWN = 0,
        P2P,        // Exactly 2 ranks - point-to-point inter-pipeline communication
        COLLECTIVE  // More than 2 ranks - collective operations (AllReduce, etc.)
    };
    CommType comm_type;  // Inferred from nranks

    // Scale-up execution mode classification.
    //
    // We assume a communicator is either CUDA-Graph-driven or not. Once determined,
    // the mode is persisted here and used to:
    //  - annotate exported OTEL metrics, and
    //  - select the appropriate scale-up aggregation path (CUDA Graph vs non-CUDA Graph).
    enum class ScaleUpExecMode : uint8_t
    {
        UNKNOWN = 0,
        NON_CUDA_GRAPH,
        CUDA_GRAPH
    };
    std::atomic<uint8_t> scaleUpExecMode;  // stores ScaleUpExecMode as uint8_t

    bool isScaleUpCudaGraphDriven() const
    {
        return scaleUpExecMode.load(std::memory_order_acquire) == static_cast<uint8_t>(ScaleUpExecMode::CUDA_GRAPH);
    }
    const char* getScaleUpExecModeString() const
    {
        auto mode = static_cast<ScaleUpExecMode>(scaleUpExecMode.load(std::memory_order_acquire));
        switch (mode)
        {
            case ScaleUpExecMode::NON_CUDA_GRAPH:
                return "non_cuda_graph";
            case ScaleUpExecMode::CUDA_GRAPH:
                return "cuda_graph";
            default:
                return "unknown";
        }
    }

    // Get human-readable communicator type string
    const char* getCommTypeString() const
    {
        switch (comm_type)
        {
            case CommType::P2P:
                return "P2P";
            case CommType::COLLECTIVE:
                return "COLLECTIVE";
            default:
                return "UNKNOWN";
        }
    }

    // Window management configuration
    double window_timeout_usec;  // Window closing timeout in microseconds

    // Compatibility aliases
    std::string commName;  // String version for easier use

    CommunicatorState();

    ~CommunicatorState();

    otelEventHandle_t* allocate_event_slot(void* parentObj = nullptr, double current_time = 0.0);

    bool should_close_window(uint8_t buffer_idx, double current_time);

    void set_window_start_time_if_needed(uint8_t buffer_idx, double current_time);

    void trigger_window_closing(uint8_t buffer_idx);

    void switch_to_next_buffer(uint8_t current_idx);

    void mark_operation_start(uint8_t buffer_idx);

    void mark_operation_complete(uint8_t buffer_idx);

    uint8_t get_active_buffer_idx() const;

    WindowMetadata* get_window_metadata(uint8_t buffer_idx);
};

otelEventHandle_t* get_next_event_handle(CommunicatorState* state, void* parentObj, double current_time);

#endif  // COMMUNICATOR_STATE_H_
