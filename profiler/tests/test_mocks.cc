// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

// Shared mock implementations for all test files
// This file provides mock implementations of external dependencies

#include <atomic>
#include <mutex>
#include <utility>
#include <vector>

#include "../communicator_state.h"

// Global tracking for window ready notifications (for testing)
static std::atomic<int> g_notify_window_ready_call_count{0};
static std::vector<std::pair<CommunicatorState*, int>> g_notify_window_ready_calls;
static std::mutex g_notify_window_ready_mutex;  // Protect the vector from concurrent access
static std::atomic<int> g_unregister_communicator_call_count{0};
static CommunicatorState* g_last_unregistered_communicator = nullptr;

// Helper functions to access mock state (for tests)
void reset_notify_window_ready_tracking()
{
    std::lock_guard<std::mutex> lock(g_notify_window_ready_mutex);
    g_notify_window_ready_call_count.store(0);
    g_notify_window_ready_calls.clear();
}

int get_notify_window_ready_call_count()
{
    return g_notify_window_ready_call_count.load();
}

void reset_unregister_communicator_tracking()
{
    g_unregister_communicator_call_count.store(0);
    g_last_unregistered_communicator = nullptr;
}

int get_unregister_communicator_call_count()
{
    return g_unregister_communicator_call_count.load();
}

CommunicatorState* get_last_unregistered_communicator()
{
    return g_last_unregistered_communicator;
}

std::vector<std::pair<CommunicatorState*, int>> get_notify_window_ready_calls()
{
    // Return a copy to avoid race conditions when accessing the vector
    std::lock_guard<std::mutex> lock(g_notify_window_ready_mutex);
    return g_notify_window_ready_calls;
}

// Mock telemetry functions
// Note: These must match the linkage in telemetry.h (C++ linkage, not extern "C")
void profiler_otel_telemetry_init()
{
    // Mock implementation - do nothing
}

void profiler_otel_telemetry_cleanup()
{
    // Mock implementation - do nothing
}

void profiler_otel_telemetry_notify_window_ready(struct CommunicatorState* commState, int window_idx)
{
    g_notify_window_ready_call_count.fetch_add(1, std::memory_order_relaxed);

    // Thread-safe push to vector
    std::lock_guard<std::mutex> lock(g_notify_window_ready_mutex);
    g_notify_window_ready_calls.push_back(std::make_pair(commState, window_idx));
}

void profiler_otel_telemetry_unregister_communicator(struct CommunicatorState* commState)
{
    g_unregister_communicator_call_count.fetch_add(1, std::memory_order_relaxed);
    g_last_unregistered_communicator = commState;
}

// Mock gettime function
// Returns incrementing timestamp to simulate time passing
double gettime()
{
    static double time = 1234567890.0 * 1e6;  // Start at fixed timestamp
    double current     = time;
    time += 1.0;  // Increment by 1 microsecond each call
    return current;
}
