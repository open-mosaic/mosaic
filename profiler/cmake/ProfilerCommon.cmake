# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

function(profiler_set_platform_defaults)
    if(NOT DEFINED GPU_PLATFORM)
        set(GPU_PLATFORM "AUTO" PARENT_SCOPE)
    endif()

    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "/opt/rocm" PARENT_SCOPE)
    endif()
endfunction()

function(_profiler_find_cuda_runtime log_prefix out_found out_type out_include_dirs out_compile_defs out_cuda_found out_cuda_lib_dir)
    set(cuda_found FALSE)
    set(cuda_include_dirs)
    set(cuda_compile_defs GPU_PLATFORM_CUDA)
    set(cuda_library_dir "")

    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "${log_prefix}CUDA found at: ${CUDAToolkit_INCLUDE_DIRS}")
        set(cuda_found TRUE)
        set(cuda_include_dirs ${CUDAToolkit_INCLUDE_DIRS})
        set(cuda_library_dir "${CUDAToolkit_LIBRARY_DIR}")
        set(cuda_toolkit_found TRUE)
    else()
        set(cuda_toolkit_found FALSE)
        set(cuda_paths
            "/usr/local/cuda/include"
            "/usr/local/cuda-13.0/include"
            "/usr/local/cuda-13/include"
            "/usr/local/cuda-12.9/include"
            "/usr/local/cuda-12/include"
            "/usr/local/cuda-11/include"
            "/opt/cuda/include"
            "/usr/include/cuda")

        foreach(cuda_path ${cuda_paths})
            if(EXISTS "${cuda_path}/cuda_runtime.h")
                message(STATUS "${log_prefix}CUDA headers found at: ${cuda_path}")
                set(cuda_found TRUE)
                set(cuda_include_dirs "${cuda_path}")
                break()
            endif()
        endforeach()
    endif()

    set(${out_found} ${cuda_found} PARENT_SCOPE)
    set(${out_type} "CUDA" PARENT_SCOPE)
    set(${out_include_dirs} ${cuda_include_dirs} PARENT_SCOPE)
    set(${out_compile_defs} ${cuda_compile_defs} PARENT_SCOPE)
    set(${out_cuda_found} ${cuda_toolkit_found} PARENT_SCOPE)
    set(${out_cuda_lib_dir} "${cuda_library_dir}" PARENT_SCOPE)
endfunction()

function(_profiler_find_rocm_runtime log_prefix out_found out_type out_include_dirs out_compile_defs out_rocm_path)
    set(local_rocm_path "${ROCM_PATH}")
    set(rocm_found FALSE)
    set(rocm_include_dirs)
    set(rocm_compile_defs GPU_PLATFORM_ROCM __HIP_PLATFORM_AMD__)

    set(rocm_include_path "${local_rocm_path}/include")
    set(hip_include_path "${local_rocm_path}/include/hip")

    if(EXISTS "${rocm_include_path}/hip/hip_runtime.h" OR EXISTS "${hip_include_path}/hip_runtime.h")
        message(STATUS "${log_prefix}ROCm/HIP found at: ${local_rocm_path}")
        list(APPEND rocm_include_dirs "${rocm_include_path}")
        if(EXISTS "${hip_include_path}")
            list(APPEND rocm_include_dirs "${hip_include_path}")
        endif()
        set(rocm_found TRUE)
    else()
        set(rocm_paths
            "/opt/rocm"
            "/opt/rocm-6.0"
            "/opt/rocm-5.7"
            "/opt/rocm-5.6"
            "/usr/local/rocm")

        foreach(rocm_search_path ${rocm_paths})
            if(EXISTS "${rocm_search_path}/include/hip/hip_runtime.h")
                message(STATUS "${log_prefix}ROCm/HIP headers found at: ${rocm_search_path}")
                set(local_rocm_path "${rocm_search_path}")
                set(rocm_found TRUE)
                set(rocm_include_dirs "${rocm_search_path}/include")
                break()
            endif()
        endforeach()
    endif()

    set(${out_found} ${rocm_found} PARENT_SCOPE)
    set(${out_type} "ROCM" PARENT_SCOPE)
    set(${out_include_dirs} ${rocm_include_dirs} PARENT_SCOPE)
    set(${out_compile_defs} ${rocm_compile_defs} PARENT_SCOPE)
    set(${out_rocm_path} "${local_rocm_path}" PARENT_SCOPE)
endfunction()

function(profiler_detect_gpu_runtime log_prefix)
    set(gpu_runtime_found FALSE)
    set(gpu_runtime_type "")
    set(gpu_include_dirs)
    set(gpu_compile_definitions)
    set(local_rocm_path "${ROCM_PATH}")
    set(local_cuda_toolkit_found FALSE)
    set(local_cuda_library_dir "")

    if(GPU_PLATFORM STREQUAL "CUDA")
        message(STATUS "${log_prefix}GPU Platform: CUDA (forced)")
        _profiler_find_cuda_runtime("${log_prefix}" gpu_runtime_found gpu_runtime_type gpu_include_dirs
                                    gpu_compile_definitions local_cuda_toolkit_found local_cuda_library_dir)
        if(NOT gpu_runtime_found)
            message(FATAL_ERROR "${log_prefix}CUDA platform requested but CUDA headers not found.")
        endif()
    elseif(GPU_PLATFORM STREQUAL "ROCM")
        message(STATUS "${log_prefix}GPU Platform: ROCm (forced)")
        _profiler_find_rocm_runtime("${log_prefix}" gpu_runtime_found gpu_runtime_type gpu_include_dirs
                                    gpu_compile_definitions local_rocm_path)
        if(NOT gpu_runtime_found)
            message(FATAL_ERROR "${log_prefix}ROCm platform requested but ROCm/HIP headers not found at ${local_rocm_path}")
        endif()
    else()
        message(STATUS "${log_prefix}GPU Platform: AUTO (detecting...)")
        _profiler_find_cuda_runtime("${log_prefix}" gpu_runtime_found gpu_runtime_type gpu_include_dirs
                                    gpu_compile_definitions local_cuda_toolkit_found local_cuda_library_dir)
        if(NOT gpu_runtime_found)
            _profiler_find_rocm_runtime("${log_prefix}" gpu_runtime_found gpu_runtime_type gpu_include_dirs
                                        gpu_compile_definitions local_rocm_path)
        endif()

        if(NOT gpu_runtime_found)
            message(WARNING "${log_prefix}No GPU runtime (CUDA or ROCm) found. Build may fail.")
        else()
            message(STATUS "${log_prefix}Auto-detected GPU Platform: ${gpu_runtime_type}")
        endif()
    endif()

    set(GPU_RUNTIME_FOUND ${gpu_runtime_found} PARENT_SCOPE)
    set(GPU_RUNTIME_TYPE "${gpu_runtime_type}" PARENT_SCOPE)
    set(PROFILER_GPU_INCLUDE_DIRS ${gpu_include_dirs} PARENT_SCOPE)
    set(PROFILER_GPU_COMPILE_DEFINITIONS ${gpu_compile_definitions} PARENT_SCOPE)
    set(CUDAToolkit_FOUND ${local_cuda_toolkit_found} PARENT_SCOPE)
    set(CUDAToolkit_LIBRARY_DIR "${local_cuda_library_dir}" PARENT_SCOPE)
    set(ROCM_PATH "${local_rocm_path}" PARENT_SCOPE)
endfunction()

function(profiler_configure_trace log_prefix enabled_message disabled_message)
    option(TRACE "Enable trace logging (PROFILER_OTEL_TRACE active)" OFF)
    if(TRACE)
        set(PROFILER_TRACE_COMPILE_DEFINITIONS PROFILER_OTEL_ENABLE_TRACE PARENT_SCOPE)
        message(STATUS "${log_prefix}${enabled_message}")
    else()
        set(PROFILER_TRACE_COMPILE_DEFINITIONS "" PARENT_SCOPE)
        message(STATUS "${log_prefix}${disabled_message}")
    endif()
endfunction()

function(profiler_configure_sampling_profile log_prefix)
    option(PROFILER_ENABLE_SAMPLING_PROFILE
           "Build with frame pointers and debug info for low-overhead perf sampling"
           OFF)
    if(PROFILER_ENABLE_SAMPLING_PROFILE)
        add_compile_options(-fno-omit-frame-pointer -fno-optimize-sibling-calls -g)
        message(STATUS "${log_prefix}Sampling profiler support enabled (-g -fno-omit-frame-pointer)")
    else()
        message(STATUS "${log_prefix}Sampling profiler support disabled")
    endif()
endfunction()

function(_profiler_require_existing_path description path_value)
    if(NOT EXISTS "${path_value}")
        message(FATAL_ERROR "${description} not found: ${path_value}")
    endif()
endfunction()

function(profiler_resolve_nccl_paths log_prefix require_nccl)
    set(local_nccl_path "${NCCL_PATH}")
    if(NOT local_nccl_path)
        set(local_nccl_path "$ENV{NCCL_PATH}")
    endif()

    if(NOT local_nccl_path)
        if(require_nccl)
            message(FATAL_ERROR "${log_prefix}NCCL_PATH not set. Please set environment variable NCCL_PATH or pass -DNCCL_PATH=<path> to cmake")
        else()
            message(WARNING "${log_prefix}NCCL_PATH not set. Tests may not compile correctly.")
            set(PROFILER_NCCL_INCLUDE_DIRS "" PARENT_SCOPE)
            set(NCCL_PATH "" PARENT_SCOPE)
            return()
        endif()
    endif()

    message(STATUS "${log_prefix}NCCL_PATH=${local_nccl_path}")
    _profiler_require_existing_path("${log_prefix}NCCL root" "${local_nccl_path}")

    set(nccl_src_include_path "${local_nccl_path}/src/include")
    set(nccl_build_include_path "${local_nccl_path}/build/include")
    set(nccl_plugin_include_path "${local_nccl_path}/src/include/plugin")

    _profiler_require_existing_path("${log_prefix}NCCL source include path" "${nccl_src_include_path}")
    _profiler_require_existing_path("${log_prefix}NCCL build include path" "${nccl_build_include_path}")
    _profiler_require_existing_path("${log_prefix}NCCL plugin include path" "${nccl_plugin_include_path}")

    set(PROFILER_NCCL_INCLUDE_DIRS
        "${nccl_src_include_path}"
        "${nccl_build_include_path}"
        "${nccl_plugin_include_path}"
        PARENT_SCOPE)
    set(NCCL_PATH "${local_nccl_path}" PARENT_SCOPE)
endfunction()