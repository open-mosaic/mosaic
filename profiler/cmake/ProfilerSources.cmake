# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

set(PROFILER_CORE_SOURCE_FILES
    aggregation.cc
    aggregation_scale_up.cc
    communicator_state.cc
    linear_regression.cc
    nccl_plugin.cc
    profiler_gpu_metadata.cc
    profiler_otel.cc
    profiler_otel_lifecycle.cc
    profiler_runtime_state.cc
    profiler_v4_compat.cc
    scale_up_inference.cc)

set(PROFILER_TELEMETRY_SOURCE_FILES
    telemetry.cc
    telemetry_runtime.cc
    telemetry_export.cc
    telemetry_emit_views.cc
    telemetry_primer.cc
    window_processor.cc)

set(PROFILER_TEST_SOURCE_FILES
    test_main.cc
    test_mocks.cc
    support/event_descr_builders.cc
    support/event_handle_builders.cc
    unit/test_profiler_otel.cc
    unit/test_profiler_utils.cc
    unit/test_params.cc
    unit/test_communicator_state.cc
    unit/test_linear_regression.cc
    unit/aggregation/test_aggregated_transfer.cc
    unit/aggregation/test_aggregated_operation.cc
    unit/aggregation/test_window_aggregator_basic.cc
    unit/aggregation/test_window_aggregator_proxy_linking.cc
    unit/aggregation/test_window_aggregator_scale_up.cc
    unit/aggregation/test_window_aggregator_alltoall.cc
    unit/profiler_otel/test_start_event.cc
    unit/profiler_otel/test_stop_event.cc
    unit/profiler_otel/test_record_event_state.cc
    unit/profiler_otel/test_group_routing.cc
    unit/test_edge_cases.cc
    unit/test_race_conditions.cc
    unit/test_scale_up_inference.cc
    unit/test_v4_compat.cc)

set(PROFILER_TELEMETRY_TEST_SOURCE_FILES
    test_main.cc
    test_mocks.cc
    mocks/param_mocks.cc
    mocks/telemetry_export_mocks.cc
    support/event_handle_builders.cc
    unit/telemetry/test_emit_views.cc
    unit/telemetry/test_primer_state_machine.cc
    unit/telemetry/test_window_processor.cc)

set(PROFILER_TELEMETRY_TEST_PLUGIN_SOURCE_FILES
    aggregation.cc
    aggregation_scale_up.cc
    communicator_state.cc
    linear_regression.cc
    profiler_runtime_state.cc
    scale_up_inference.cc
    telemetry_emit_views.cc
    telemetry_primer.cc
    window_processor.cc)

function(profiler_prefix_sources out_var base_dir)
    set(prefixed_sources)
    foreach(source_file IN LISTS ARGN)
        list(APPEND prefixed_sources "${base_dir}/${source_file}")
    endforeach()
    set(${out_var} ${prefixed_sources} PARENT_SCOPE)
endfunction()