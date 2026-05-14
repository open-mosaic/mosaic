// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include "communicator_state.h"

#include <cstdlib>
#include <cstring>

#include "profiler_otel.h"
#include "telemetry.h"  // For profiler_otel_telemetry_notify_window_ready

/**
 * @brief Construct a CommunicatorState with initialized buffers.
 *
 * Initializes all 4 circular buffers and sets the first buffer to FILLING state.
 */
CommunicatorState::CommunicatorState()
    : buffers(nullptr),
      active_buffer_idx(0),
      next_element_idx(0),
      active_group_handle(nullptr),
      active_group_depth(0),
      comm_name(nullptr),
      comm_hash(0),
      rank(0),
      nranks(0),
      nNodes(0),
      hostname(""),
      local_rank(-1),
      gpu_pci_bus_id(""),
      gpu_uuid(""),
      comm_type(CommType::UNKNOWN),
      scaleUpExecMode(static_cast<uint8_t>(ScaleUpExecMode::UNKNOWN))
{
    // Heap-allocate buffers to avoid stack overflow (~40 MB total)
    // Allocate array of buffer pointers
    buffers = new otelEventHandle_t*[NUM_BUFFERS];
    for (int i = 0; i < NUM_BUFFERS; i++)
    {
        buffers[i] = new otelEventHandle_t[BUFFER_SIZE]();  // () zero-initializes
    }

    // Initialize window timeout (may be overridden in profiler_otel_init_v5)
    window_timeout_usec = WINDOW_TRIGGER_TIME_SEC * 1e6;  // Default 5 seconds in microseconds

    // Initialize window metadata
    for (int i = 0; i < NUM_BUFFERS; i++)
    {
        windows[i].state.store(WINDOW_READY, std::memory_order_release);
        windows[i].element_count.store(0, std::memory_order_release);
        windows[i].in_progress_count.store(0, std::memory_order_release);
        windows[i].groups_in_progress.store(0, std::memory_order_release);
        windows[i].proxy_ops_in_progress.store(0, std::memory_order_release);
        windows[i].kernel_ch_in_progress.store(0, std::memory_order_release);
        windows[i].pending_first_child.store(0, std::memory_order_release);
        windows[i].start_time = 0.0;
    }

    // Mark first buffer as FILLING
    windows[0].state.store(WINDOW_FILLING, std::memory_order_release);
}

/**
 * @brief Destructor for CommunicatorState.
 *
 * Frees heap-allocated buffers.
 */
CommunicatorState::~CommunicatorState()
{
    // Free heap-allocated buffers
    if (buffers)
    {
        for (int i = 0; i < NUM_BUFFERS; i++)
        {
            delete[] buffers[i];
        }
        delete[] buffers;
        buffers = nullptr;
    }
}

/**
 * @brief Get the window index containing a parent event handle.
 *
 * Searches all circular buffers to find which window contains the parentObj pointer.
 * Validates that the pointer is properly aligned to an event handle boundary.
 *
 * @param[in] parentObj Parent event handle pointer.
 * @param[in] commState Communicator state containing buffers.
 *
 * @return Window index (0-3) if found, UINT8_MAX if not found or invalid.
 */
static uint8_t get_parent_window_idx(void* parentObj, CommunicatorState* commState)
{
    if (!parentObj || !commState || !commState->buffers) return UINT8_MAX;  // Invalid window index

    // Verify the pointer is within our buffer range to avoid segfaults
    // Check all circular buffers
    uintptr_t ptr_addr = (uintptr_t)parentObj;
    for (uint8_t i = 0; i < NUM_BUFFERS; i++)
    {
        if (!commState->buffers[i]) continue;  // Buffer not allocated

        uintptr_t buffer_start = (uintptr_t)commState->buffers[i];
        uintptr_t buffer_end   = buffer_start + (BUFFER_SIZE * sizeof(otelEventHandle_t));

        if (ptr_addr >= buffer_start && ptr_addr < buffer_end)
        {
            // Check if it's properly aligned to an event handle boundary
            if ((ptr_addr - buffer_start) % sizeof(otelEventHandle_t) == 0)
            {
                // Valid pointer - cast and get buffer_idx
                otelEventHandle_t* parent_event = (otelEventHandle_t*)parentObj;
                return parent_event->buffer_idx;
            }
        }
    }

    // Pointer is not in our buffers - likely a test mock or invalid pointer
    return UINT8_MAX;
}

/**
 * @brief Allocate a slot in the circular buffer (lock-free).
 *
 * Routes events to the appropriate window based on parent object:
 * - If parentObj is in a CLOSING window, route to that window
 * - Otherwise route to the active FILLING window
 * - Checks for window closing triggers before allocation
 *
 * @param[in] parentObj Parent event handle (nullptr for root events).
 * @param[in] current_time Current time in microseconds (for time-based closing).
 *
 * @return Pointer to allocated event slot, or nullptr if allocation failed.
 *
 * @note Lock-free implementation using atomic operations.
 * @note Retries on race conditions or invalid window states.
 */
otelEventHandle_t* CommunicatorState::allocate_event_slot(void* parentObj, double current_time)
{
    // Retry loop to handle race conditions
    while (true)
    {
        uint8_t target_window = UINT8_MAX;

        // Determine target window based on parent
        if (parentObj)
        {
            // Event has a parent - check which window parent belongs to
            target_window = get_parent_window_idx(parentObj, this);
            if (target_window < NUM_BUFFERS)
            {
                WindowState parent_state = windows[target_window].state.load(std::memory_order_acquire);
                if (parent_state == WINDOW_CLOSING)
                {
                    // Parent is in CLOSING window - allocate to same CLOSING window
                    // This ensures ProxyOps and ProxySteps stay with their parent Coll/P2P
                    // Only events with parents in CLOSING window can be added to CLOSING window
                }
                else if (parent_state == WINDOW_FILLING)
                {
                    // Parent is in FILLING window - allocate to same FILLING window
                    // This ensures ProxyOps and ProxySteps stay with their parent Coll/P2P
                }
                else
                {
                    // Parent window is PROCESSING or READY - shouldn't happen, use active window
                    target_window = UINT8_MAX;
                }
            }
            else
            {
                // Invalid parent window, use active window
                target_window = UINT8_MAX;
            }
        }

        // If no parent or invalid parent window, use active window
        if (target_window == UINT8_MAX)
        {
            target_window = active_buffer_idx.load(std::memory_order_acquire);
        }

        // Check target window state
        WindowState target_state = windows[target_window].state.load(std::memory_order_acquire);

        // If target window is CLOSING, verify this event has a parent in that CLOSING window
        if (target_state == WINDOW_CLOSING)
        {
            if (!parentObj)
            {
                // New collective (no parent) cannot go to CLOSING window
                // Route to active window (which should be FILLING)
                target_window = active_buffer_idx.load(std::memory_order_acquire);
                target_state  = windows[target_window].state.load(std::memory_order_acquire);

                // Verify active window is FILLING
                // Race condition: window might be closing concurrently, so retry if not FILLING
                if (target_state != WINDOW_FILLING)
                {
                    // Check if this window is still the active window (might have switched)
                    uint8_t current_active = active_buffer_idx.load(std::memory_order_acquire);
                    if (target_window != current_active)
                    {
                        // Active window switched, retry with new active window
                        OTEL_TRACE(NCCL_INIT,
                                   "Active window switched from %u to %u during allocation, retrying (expected race)",
                                   target_window, current_active);
                        continue;
                    }
                    else
                    {
                        // Active window is not FILLING (should be rare - window closing race)
                        OTEL_TRACE(NCCL_INIT, "Active window %u is not FILLING (state=%u), retrying (race condition)",
                                   target_window, (uint8_t)target_state);
                        // Retry to get correct state
                        continue;
                    }
                }
            }
            else
            {
                // Event has parent - verify parent is actually in this CLOSING window
                uint8_t parent_window = get_parent_window_idx(parentObj, this);
                if (parent_window != target_window)
                {
                    // Parent is not in the CLOSING window - this shouldn't happen
                    // Route to parent's actual window instead
                    if (parent_window < NUM_BUFFERS)
                    {
                        WindowState parent_state = windows[parent_window].state.load(std::memory_order_acquire);
                        if (parent_state == WINDOW_FILLING)
                        {
                            // Parent is in FILLING window - route there
                            target_window = parent_window;
                            target_state  = WINDOW_FILLING;
                        }
                        else
                        {
                            // Parent is in unexpected state, use active window
                            target_window = active_buffer_idx.load(std::memory_order_acquire);
                            target_state  = windows[target_window].state.load(std::memory_order_acquire);

                            // Verify active window is in valid state (race condition protection)
                            if (target_state != WINDOW_FILLING && target_state != WINDOW_CLOSING)
                            {
                                // Window state is invalid, retry
                                continue;
                            }
                        }
                    }
                    else
                    {
                        // Invalid parent window, use active window
                        target_window = active_buffer_idx.load(std::memory_order_acquire);
                        target_state  = windows[target_window].state.load(std::memory_order_acquire);

                        // Verify active window is in valid state (race condition protection)
                        if (target_state != WINDOW_FILLING && target_state != WINDOW_CLOSING)
                        {
                            // Window state is invalid, retry
                            continue;
                        }
                    }
                }
                // else: parent is in CLOSING window, which is correct - continue with target_window
            }
        }

        // Verify target window is in valid state
        target_state = windows[target_window].state.load(std::memory_order_acquire);

        // Check if target window should close (for time-based closing)
        if (target_state == WINDOW_FILLING && current_time > 0.0)
        {
            if (should_close_window(target_window, current_time))
            {
                trigger_window_closing(target_window);
                target_state = windows[target_window].state.load(std::memory_order_acquire);
            }
        }

        if (target_state != WINDOW_FILLING && target_state != WINDOW_CLOSING)
        {
            // Window is not in valid state, retry
            continue;
        }

        // Check current element count BEFORE incrementing to prevent overflow
        // For CLOSING windows, be more strict - only allow if there's room
        uint32_t current_count = windows[target_window].element_count.load(std::memory_order_acquire);
        if (current_count >= BUFFER_SIZE)
        {
            // Buffer is full - cannot allocate more events
            if (target_state == WINDOW_CLOSING)
            {
                // CLOSING window is full - this should not happen often, but can occur
                // if many events with parents in CLOSING window arrive after window closed
                OTEL_WARN(NCCL_INIT, "CLOSING window %u is full (count=%u), rejecting event with parent in this window",
                          target_window, current_count);
            }
            else
            {
                // FILLING window is full - should have triggered closing earlier
                OTEL_WARN(NCCL_INIT, "FILLING window %u is full (count=%u), this should not happen", target_window,
                          current_count);
            }
            return nullptr;
        }

        // Use element_count as slot index (atomic increment gives us next slot)
        uint32_t slot_idx = windows[target_window].element_count.fetch_add(1, std::memory_order_acq_rel);

        // Double-check for buffer overflow (race condition protection)
        if (slot_idx >= BUFFER_SIZE)
        {
            OTEL_WARN(NCCL_INIT,
                      "Buffer overflow detected after increment: slot_idx=%u >= BUFFER_SIZE=%d, buffer_idx=%u",
                      slot_idx, BUFFER_SIZE, target_window);
            // Decrement count since we can't use this slot
            windows[target_window].element_count.fetch_sub(1, std::memory_order_acq_rel);
            return nullptr;
        }

        // Verify window state didn't change to invalid state
        WindowState verify_state = windows[target_window].state.load(std::memory_order_acquire);
        if (verify_state != WINDOW_FILLING && verify_state != WINDOW_CLOSING)
        {
            // Window state changed to invalid, retry
            continue;
        }

        // Check if we reached trigger threshold (for FILLING windows)
        if (verify_state == WINDOW_FILLING && slot_idx + 1 >= WINDOW_TRIGGER_COUNT)
        {
            trigger_window_closing(target_window);
        }

        // Set buffer_idx in the allocated slot so caller knows which window this belongs to
        otelEventHandle_t* event = &buffers[target_window][slot_idx];
        event->buffer_idx        = target_window;

        // Return the allocated slot
        return event;
    }
}

/**
 * @brief Check if a window should close.
 *
 * Windows close when either:
 * - Element count reaches WINDOW_TRIGGER_COUNT (10k), OR
 * - Time elapsed exceeds window_timeout_usec (configurable)
 *
 * @param[in] buffer_idx Window index to check (0-3).
 * @param[in] current_time Current time in microseconds.
 *
 * @return true if window should close, false otherwise.
 *
 * @note Only checks FILLING windows (other states return false).
 */
bool CommunicatorState::should_close_window(uint8_t buffer_idx, double current_time)
{
    WindowMetadata* window = &windows[buffer_idx];

    // Check if window is in FILLING state
    WindowState state = window->state.load(std::memory_order_acquire);
    if (state != WINDOW_FILLING)
    {
        return false;
    }

    // Check element count trigger
    uint32_t count = window->element_count.load(std::memory_order_acquire);
    if (count >= WINDOW_TRIGGER_COUNT)
    {
        OTEL_TRACE(NCCL_INIT, "Window %u count trigger: count=%u >= %d", buffer_idx, count, WINDOW_TRIGGER_COUNT);
        return true;
    }

    // Check time trigger
    if (window->start_time > 0.0)
    {
        double elapsed = current_time - window->start_time;

        // Log every 500 calls to see if we're checking and what the elapsed time is
        static std::atomic<int> check_counter{0};
        if (check_counter.fetch_add(1, std::memory_order_relaxed) % 500 == 0)
        {
            OTEL_TRACE(NCCL_INIT, "Window %u time check: elapsed=%.2f us (threshold=%.0f us), count=%u", buffer_idx,
                       elapsed, window_timeout_usec, count);
        }

        if (elapsed >= window_timeout_usec)
        {
            OTEL_TRACE(NCCL_INIT, "Window %u TIME TRIGGER: elapsed=%.2f us >= %.0f us", buffer_idx, elapsed,
                       window_timeout_usec);
            return true;
        }
    }

    return false;
}

/**
 * @brief Trigger window closing and switch to next buffer.
 *
 * Transitions a window from FILLING to CLOSING, switches active buffer to the
 * next window, and initializes the next window for FILLING. If no operations
 * are in-progress, immediately transitions to PROCESSING.
 *
 * @param[in] buffer_idx Window index to close (0-3).
 *
 * @note Thread-safe: uses compare-and-swap for state transitions.
 * @note If next window is not READY/PROCESSING, forces it to READY.
 */
void CommunicatorState::trigger_window_closing(uint8_t buffer_idx)
{
    WindowMetadata* window = &windows[buffer_idx];

    // Transition from FILLING to CLOSING
    WindowState expected = WINDOW_FILLING;
    if (window->state.compare_exchange_strong(expected, WINDOW_CLOSING, std::memory_order_acq_rel))
    {
        uint32_t in_progress = window->in_progress_count.load(std::memory_order_acquire);
        OTEL_TRACE(NCCL_INIT, "Window %u transitioning to CLOSING (count=%u, in_progress=%u)", buffer_idx,
                   window->element_count.load(std::memory_order_acquire), in_progress);

        // Immediately switch active buffer to next window and set it to FILLING
        // This ensures new collectives go to the new FILLING window
        uint8_t next_idx = (buffer_idx + 1) % NUM_BUFFERS;

        // Verify next window is READY (should be, unless processing is very slow)
        WindowState next_state = windows[next_idx].state.load(std::memory_order_acquire);
        if (next_state != WINDOW_READY && next_state != WINDOW_PROCESSING)
        {
            OTEL_WARN(NCCL_INIT, "Next window %u is not READY or PROCESSING (state=%u), forcing to READY", next_idx,
                      (uint8_t)next_state);
            // Force it to READY (processing thread should have finished)
            windows[next_idx].state.store(WINDOW_READY, std::memory_order_release);
        }

        // Initialize next buffer for FILLING
        windows[next_idx].state.store(WINDOW_FILLING, std::memory_order_release);
        windows[next_idx].element_count.store(0, std::memory_order_release);
        windows[next_idx].in_progress_count.store(0, std::memory_order_release);
        windows[next_idx].groups_in_progress.store(0, std::memory_order_release);
        windows[next_idx].proxy_ops_in_progress.store(0, std::memory_order_release);
        windows[next_idx].kernel_ch_in_progress.store(0, std::memory_order_release);
        windows[next_idx].pending_first_child.store(0, std::memory_order_release);
        windows[next_idx].start_time = 0.0;  // Will be set on first event

        // Switch active buffer atomically
        active_buffer_idx.store(next_idx, std::memory_order_release);

        OTEL_TRACE(NCCL_INIT, "Active buffer switched from %u to %u (new window is FILLING)", buffer_idx, next_idx);

        // Check if CLOSING window can immediately transition to PROCESSING
        uint32_t proxy_ops_pending = window->proxy_ops_in_progress.load(std::memory_order_acquire);
        uint32_t kernel_ch_pending = window->kernel_ch_in_progress.load(std::memory_order_acquire);
        uint32_t groups_active     = window->groups_in_progress.load(std::memory_order_acquire);

        if (in_progress == 0)
        {
            // No operations in progress, transition to PROCESSING immediately
            WindowState closing_state = WINDOW_CLOSING;
            if (window->state.compare_exchange_strong(closing_state, WINDOW_PROCESSING, std::memory_order_acq_rel))
            {
                OTEL_TRACE(NCCL_INIT, "Window %u transitioned to PROCESSING (no in-progress ops)", buffer_idx);
                profiler_otel_telemetry_notify_window_ready(this, buffer_idx);
            }
        }
        else if (proxy_ops_pending == 0 && kernel_ch_pending == 0 && groups_active == 0)
        {
            // All ProxyOps, KernelCh events, and Groups are done, but in_progress > 0.
            // The remaining count is from orphaned Coll/P2P +1s (Colls that received
            // neither ProxyOps nor KernelCh events).
            OTEL_TRACE(NCCL_INIT, "Window %u: %u orphaned in-progress ops (no pending ops/groups), forcing PROCESSING",
                       buffer_idx, in_progress);
            window->in_progress_count.store(0, std::memory_order_release);
            WindowState closing_state = WINDOW_CLOSING;
            if (window->state.compare_exchange_strong(closing_state, WINDOW_PROCESSING, std::memory_order_acq_rel))
            {
                OTEL_TRACE(NCCL_INIT, "Window %u transitioned to PROCESSING (forced)", buffer_idx);
                profiler_otel_telemetry_notify_window_ready(this, buffer_idx);
            }
        }
        else
        {
            OTEL_TRACE(
                NCCL_INIT,
                "Window %u will continue until %u in-progress ops complete (groups=%u, proxy_ops=%u, kernel_ch=%u)",
                buffer_idx, in_progress, groups_active, proxy_ops_pending, kernel_ch_pending);
        }
    }
}

/**
 * @brief Switch window from CLOSING to PROCESSING.
 *
 * Called when in_progress_count reaches 0. Transitions the window to PROCESSING
 * and notifies the telemetry thread. The active buffer was already switched
 * in trigger_window_closing().
 *
 * @param[in] current_idx Window index to transition (0-3).
 *
 * @note Thread-safe: uses compare-and-swap for state transition.
 */
void CommunicatorState::switch_to_next_buffer(uint8_t current_idx)
{
    WindowMetadata* window = &windows[current_idx];

    OTEL_TRACE(NCCL_INIT, "Window %u: All in-progress ops completed, transitioning to PROCESSING", current_idx);

    // Transition current window from CLOSING to PROCESSING
    // Note: The active buffer was already switched to next window in trigger_window_closing()
    WindowState closing_state = WINDOW_CLOSING;
    if (window->state.compare_exchange_strong(closing_state, WINDOW_PROCESSING, std::memory_order_acq_rel))
    {
        OTEL_TRACE(NCCL_INIT, "Window %u transitioned to PROCESSING", current_idx);
        // Notify telemetry thread
        profiler_otel_telemetry_notify_window_ready(this, current_idx);
    }
    else
    {
        OTEL_WARN(NCCL_INIT, "Window %u failed to transition to PROCESSING (state=%u)", current_idx,
                  (uint8_t)window->state.load(std::memory_order_acquire));
    }
}

/**
 * @brief Mark an operation as in-progress.
 *
 * Increments the window's in_progress_count. Called when a Coll/P2P starts
 * to track expected ProxyOp completions.
 *
 * @param[in] buffer_idx Window index (0-3).
 *
 * @note Thread-safe: uses atomic increment.
 */
void CommunicatorState::mark_operation_start(uint8_t buffer_idx)
{
    windows[buffer_idx].in_progress_count.fetch_add(1, std::memory_order_acq_rel);
}

/**
 * @brief Mark an operation as completed.
 *
 * Decrements the window's in_progress_count. Called when a ProxyOp completes.
 * If this was the last operation, triggers window transition to PROCESSING.
 *
 * @param[in] buffer_idx Window index (0-3).
 *
 * @note Thread-safe: uses atomic decrement.
 * @note Only processes CLOSING or FILLING windows (ignores PROCESSING/READY).
 * @note Prevents underflow by checking count before decrementing.
 */
void CommunicatorState::mark_operation_complete(uint8_t buffer_idx)
{
    // Check window state first - only allow completion on CLOSING windows
    // Windows in PROCESSING or READY state should not accept completions
    WindowState state = windows[buffer_idx].state.load(std::memory_order_acquire);
    if (state != WINDOW_CLOSING && state != WINDOW_FILLING)
    {
        // Window is already PROCESSING or READY - this ProxyOp completed too late
        // This can happen if a ProxyOp completes after the window has already been processed
        OTEL_TRACE(NCCL_INIT, "ProxyOp completed on window %u in state %u (already processed, ignoring)", buffer_idx,
                   (uint8_t)state);
        return;
    }

    // Atomically decrement in_progress_count using CAS loop to prevent underflow.
    // A plain load-check-then-fetch_sub has a TOCTOU race: two threads (e.g. main
    // thread stopping a Group and proxy thread stopping a ProxyOp) can both read
    // count=1, both pass the check, and both decrement — wrapping a uint32_t to
    // UINT32_MAX and permanently stalling the window in CLOSING state.
    uint32_t current = windows[buffer_idx].in_progress_count.load(std::memory_order_acquire);
    while (true)
    {
        if (current == 0)
        {
            OTEL_TRACE(
                NCCL_INIT,
                "ProxyOp completion on window %u with in_progress_count already 0 (state=%u) - expected if fewer "
                "ProxyOps created than nChannels",
                buffer_idx, (uint8_t)state);
            return;
        }

        if (windows[buffer_idx].in_progress_count.compare_exchange_weak(current, current - 1, std::memory_order_acq_rel,
                                                                        std::memory_order_acquire))
        {
            // Successfully decremented from 'current' to 'current - 1'
            if (current == 1)
            {
                // We just decremented from 1 to 0
                WindowState verify_state = windows[buffer_idx].state.load(std::memory_order_acquire);
                if (verify_state == WINDOW_CLOSING)
                {
                    OTEL_TRACE(NCCL_INIT, "Window %u: Last operation completed, triggering buffer switch", buffer_idx);
                    switch_to_next_buffer(buffer_idx);
                }
            }
            return;
        }
        // CAS failed: 'current' was updated to the actual value, retry
    }
}

/**
 * @brief Get the current active buffer index.
 *
 * @return Active buffer index (0-3).
 *
 * @note Thread-safe: uses atomic load.
 */
uint8_t CommunicatorState::get_active_buffer_idx() const
{
    return active_buffer_idx.load(std::memory_order_acquire);
}

/**
 * @brief Get window metadata for a buffer.
 *
 * @param[in] buffer_idx Window index (0-3).
 *
 * @return Pointer to WindowMetadata, or nullptr if buffer_idx is invalid.
 */
WindowMetadata* CommunicatorState::get_window_metadata(uint8_t buffer_idx)
{
    if (buffer_idx >= NUM_BUFFERS)
    {
        return nullptr;
    }
    return &windows[buffer_idx];
}

/**
 * @brief Set window start time if not already set.
 *
 * Sets the start_time for time-based window closing. Only sets if:
 * - start_time is 0.0 (unset), AND
 * - element_count > 0 (window has events)
 *
 * @param[in] buffer_idx Window index (0-3).
 * @param[in] current_time Current time in microseconds.
 */
void CommunicatorState::set_window_start_time_if_needed(uint8_t buffer_idx, double current_time)
{
    // Set start_time on first event in this window (for time-based closing)
    // Only set if not already set (0.0 indicates unset)
    if (windows[buffer_idx].start_time == 0.0 && windows[buffer_idx].element_count.load(std::memory_order_acquire) > 0)
    {
        windows[buffer_idx].start_time = current_time;
        OTEL_TRACE(NCCL_INIT, "Window %u start_time set to %.2f us (element_count=%u)", buffer_idx, current_time,
                   windows[buffer_idx].element_count.load(std::memory_order_acquire));
    }
}

/**
 * @brief Helper function to get next event handle from circular buffer.
 *
 * Wrapper around CommunicatorState::allocate_event_slot() for easier use.
 *
 * @param[in] state Communicator state containing buffers.
 * @param[in] parentObj Parent event handle (nullptr for root events).
 * @param[in] current_time Current time in microseconds.
 *
 * @return Pointer to allocated event slot, or nullptr if allocation failed.
 */
otelEventHandle_t* get_next_event_handle(CommunicatorState* state, void* parentObj, double current_time)
{
    if (!state)
    {
        return nullptr;
    }
    return state->allocate_event_slot(parentObj, current_time);
}
