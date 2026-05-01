# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)

project(nccl-profiler-otel-plugin VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build configuration
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# Linux-only build: define NCCL_OS_LINUX unconditionally.
# Required because nccl_profiler.h includes os.h, which gates
# ncclSocketDescriptor / ncclAffinity behind NCCL_OS_LINUX.
add_compile_definitions(NCCL_OS_LINUX)

# GPU Platform selection: CUDA, ROCM, or AUTO (default: AUTO)
if(NOT DEFINED GPU_PLATFORM)
    set(GPU_PLATFORM "AUTO")
endif()

# ROCm path (default: /opt/rocm)
if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm")
endif()

# Platform detection and setup
set(GPU_RUNTIME_FOUND FALSE)

# Helper function to find CUDA
function(find_cuda_runtime)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found at: ${CUDAToolkit_INCLUDE_DIRS}")
        include_directories(${CUDAToolkit_INCLUDE_DIRS})
        set(GPU_RUNTIME_FOUND TRUE PARENT_SCOPE)
        set(GPU_RUNTIME_TYPE "CUDA" PARENT_SCOPE)
        add_definitions(-DGPU_PLATFORM_CUDA)
    else()
        # Fallback to common CUDA paths
        set(CUDA_PATHS
            "/usr/local/cuda/include"
            "/usr/local/cuda-13.0/include"
            "/usr/local/cuda-13/include"
            "/usr/local/cuda-12.9/include"
            "/usr/local/cuda-12/include"
            "/usr/local/cuda-11/include"
            "/opt/cuda/include"
            "/usr/include/cuda"
        )

        foreach(CUDA_PATH ${CUDA_PATHS})
            if(EXISTS "${CUDA_PATH}/cuda_runtime.h")
                message(STATUS "CUDA headers found at: ${CUDA_PATH}")
                include_directories(${CUDA_PATH})
                set(GPU_RUNTIME_FOUND TRUE PARENT_SCOPE)
                set(GPU_RUNTIME_TYPE "CUDA" PARENT_SCOPE)
                add_definitions(-DGPU_PLATFORM_CUDA)
                return()
            endif()
        endforeach()
    endif()
endfunction()

# Helper function to find ROCm/HIP
function(find_rocm_runtime)
    # Check for ROCm HIP runtime
    set(ROCM_INCLUDE_PATH "${ROCM_PATH}/include")
    set(HIP_INCLUDE_PATH "${ROCM_PATH}/include/hip")

    if(EXISTS "${ROCM_INCLUDE_PATH}/hip/hip_runtime.h" OR EXISTS "${HIP_INCLUDE_PATH}/hip_runtime.h")
        message(STATUS "ROCm/HIP found at: ${ROCM_PATH}")
        include_directories(${ROCM_INCLUDE_PATH})
        if(EXISTS "${HIP_INCLUDE_PATH}")
            include_directories(${HIP_INCLUDE_PATH})
        endif()
        set(GPU_RUNTIME_FOUND TRUE PARENT_SCOPE)
        set(GPU_RUNTIME_TYPE "ROCM" PARENT_SCOPE)
        add_definitions(-DGPU_PLATFORM_ROCM)
        add_definitions(-D__HIP_PLATFORM_AMD__)
    else()
        # Try alternative ROCm paths
        set(ROCM_PATHS
            "/opt/rocm"
            "/opt/rocm-6.0"
            "/opt/rocm-5.7"
            "/opt/rocm-5.6"
            "/usr/local/rocm"
        )

        foreach(ROCM_SEARCH_PATH ${ROCM_PATHS})
            if(EXISTS "${ROCM_SEARCH_PATH}/include/hip/hip_runtime.h")
                message(STATUS "ROCm/HIP headers found at: ${ROCM_SEARCH_PATH}")
                include_directories("${ROCM_SEARCH_PATH}/include")
                set(GPU_RUNTIME_FOUND TRUE PARENT_SCOPE)
                set(GPU_RUNTIME_TYPE "ROCM" PARENT_SCOPE)
                set(ROCM_PATH "${ROCM_SEARCH_PATH}" PARENT_SCOPE)
                add_definitions(-DGPU_PLATFORM_ROCM)
                add_definitions(-D__HIP_PLATFORM_AMD__)
                return()
            endif()
        endforeach()
    endif()
endfunction()

# Detect GPU runtime based on GPU_PLATFORM setting
if(GPU_PLATFORM STREQUAL "CUDA")
    message(STATUS "GPU Platform: CUDA (forced)")
    find_cuda_runtime()
    if(NOT GPU_RUNTIME_FOUND)
        message(FATAL_ERROR "CUDA platform requested but CUDA headers not found.")
    endif()
elseif(GPU_PLATFORM STREQUAL "ROCM")
    message(STATUS "GPU Platform: ROCm (forced)")
    find_rocm_runtime()
    if(NOT GPU_RUNTIME_FOUND)
        message(FATAL_ERROR "ROCm platform requested but ROCm/HIP headers not found at ${ROCM_PATH}")
    endif()
else()
    # AUTO detection: try CUDA first, then ROCm
    message(STATUS "GPU Platform: AUTO (detecting...)")
    find_cuda_runtime()
    if(NOT GPU_RUNTIME_FOUND)
        find_rocm_runtime()
    endif()

    if(NOT GPU_RUNTIME_FOUND)
        message(WARNING "No GPU runtime (CUDA or ROCm) found. Build may fail.")
    else()
        message(STATUS "Auto-detected GPU Platform: ${GPU_RUNTIME_TYPE}")
    endif()
endif()

# Compile-time TRACE gating (mirror NCCL's TRACE=1 behavior)
# Enable by passing -DTRACE=ON or -DTRACE=1 to CMake
option(TRACE "Enable trace logging (PROFILER_OTEL_TRACE active)" OFF)
if(TRACE)
  add_definitions(-DPROFILER_OTEL_ENABLE_TRACE)
  message(STATUS "PROFILER OTEL TRACE enabled (TRACE=1)")
else()
  message(STATUS "PROFILER OTEL TRACE disabled (build without TRACE=1)")
endif()

# NCCL path detection
function(DETECT_PATH name path)
if (NOT DEFINED ${path})
  message(FATAL_ERROR "${name} undefined, define ${path}")
elseif(NOT EXISTS ${${path}})
  message(FATAL_ERROR "${name} not found, ${path}=${${path}}")
else()
  message(STATUS "${name} found ${path}=${${path}}")
endif()
endfunction()

# Set NCCL path from environment or command line
if(NOT DEFINED NCCL_PATH)
  set(NCCL_PATH $ENV{NCCL_PATH})
endif()

if(NOT NCCL_PATH)
  message(FATAL_ERROR "NCCL_PATH not set. Please set environment variable NCCL_PATH or pass -DNCCL_PATH=<path> to cmake")
endif()

DETECT_PATH(NCCL NCCL_PATH)

# NCCL include paths
set(NCCL_SRC_INCLUDE_PATH ${NCCL_PATH}/src/include)
set(NCCL_BUILD_INCLUDE_PATH ${NCCL_PATH}/build/include)
set(NCCL_PLUGIN_INCLUDE_PATH ${NCCL_PATH}/src/include/plugin)

# Check if paths exist
if(NOT EXISTS ${NCCL_SRC_INCLUDE_PATH})
  message(FATAL_ERROR "NCCL source include path not found: ${NCCL_SRC_INCLUDE_PATH}")
endif()

if(NOT EXISTS ${NCCL_BUILD_INCLUDE_PATH})
  message(FATAL_ERROR "NCCL build include path not found: ${NCCL_BUILD_INCLUDE_PATH}")
endif()

if(NOT EXISTS ${NCCL_PLUGIN_INCLUDE_PATH})
  message(FATAL_ERROR "NCCL plugin include path not found: ${NCCL_PLUGIN_INCLUDE_PATH}")
endif()

# Include NCCL directories
include_directories(${NCCL_SRC_INCLUDE_PATH})
include_directories(${NCCL_BUILD_INCLUDE_PATH})
include_directories(${NCCL_PLUGIN_INCLUDE_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# OpenTelemetry C++ SDK path (optional, for custom installations)
# Can be set via -DOTEL_PATH=/path/to/opentelemetry-build
if(NOT DEFINED OTEL_PATH)
    set(OTEL_PATH "")
endif()

# Find OpenTelemetry C++ SDK
# First, check if user specified a custom path
if(OTEL_PATH AND EXISTS "${OTEL_PATH}/include/opentelemetry/sdk/metrics/meter.h")
    message(STATUS "OpenTelemetry C++ SDK found at user-specified path: ${OTEL_PATH}")
    include_directories("${OTEL_PATH}/include")
    link_directories("${OTEL_PATH}/lib" "${OTEL_PATH}/lib64")
    set(OTEL_FOUND TRUE)
    set(OTEL_INSTALL_PATH "${OTEL_PATH}")
    add_definitions(-DENABLE_OTEL)
else()
    if(OTEL_PATH)
        message(WARNING "OpenTelemetry C++ SDK not found at specified OTEL_PATH: ${OTEL_PATH}. Will try to find automatically.")
    endif()

    # Try pkg-config
    find_package(PkgConfig QUIET)
    if (PkgConfig_FOUND)
      pkg_check_modules(OTEL opentelemetry-cpp)
      if (OTEL_FOUND)
        message(STATUS "OpenTelemetry C++ SDK found via pkg-config")
        include_directories(${OTEL_INCLUDE_DIRS})
        link_directories(${OTEL_LIBRARY_DIRS})
        add_definitions(-DENABLE_OTEL)
      else()
        message(WARNING "OpenTelemetry C++ SDK not found via pkg-config. Will try to find manually.")
      endif()
    endif()

    # Manual OpenTelemetry C++ SDK paths if pkg-config fails
    if(NOT OTEL_FOUND)
      set(OTEL_SEARCH_PATHS
        "/usr/local/include"
        "/usr/include"
        "/opt/opentelemetry-build/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/opentelemetry-build/include"
      )

      foreach(OTEL_SEARCH_PATH ${OTEL_SEARCH_PATHS})
        if(EXISTS "${OTEL_SEARCH_PATH}/opentelemetry/sdk/metrics/meter.h")
          message(STATUS "OpenTelemetry headers found at: ${OTEL_SEARCH_PATH}")
          include_directories(${OTEL_SEARCH_PATH})
          # Derive lib path from include path
          get_filename_component(OTEL_BASE_PATH "${OTEL_SEARCH_PATH}" DIRECTORY)
          set(OTEL_INSTALL_PATH "${OTEL_BASE_PATH}")
          set(OTEL_FOUND TRUE)
          add_definitions(-DENABLE_OTEL)
          break()
        endif()
      endforeach()

      if(NOT OTEL_FOUND)
        message(WARNING "OpenTelemetry C++ SDK not found. Telemetry will be disabled.")
        add_definitions(-DDISABLE_OTEL)
      endif()
    endif()
endif()


# Source files
SET(OTEL_SOURCES
  aggregation.cc
  communicator_state.cc
  linear_regression.cc
  nccl_plugin.cc
  profiler_otel.cc
  profiler_v4_compat.cc
  telemetry.cc
  telemetry_primer.cc
)

# Create shared library
add_library(nccl-profiler-otel SHARED ${OTEL_SOURCES})

# Link OpenTelemetry libraries
if(OTEL_FOUND)
    message(STATUS "OpenTelemetry SDK found, linking...")

    set(_otel_search_dirs /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64 ${CMAKE_CURRENT_SOURCE_DIR}/opentelemetry-build/lib)
    # Add user-specified OTEL_PATH lib directories
    if(OTEL_INSTALL_PATH)
        list(INSERT _otel_search_dirs 0 "${OTEL_INSTALL_PATH}/lib" "${OTEL_INSTALL_PATH}/lib64")
    endif()
    if(DEFINED PROFILER_OTEL_LIB_DIRS)
        list(APPEND _otel_search_dirs ${PROFILER_OTEL_LIB_DIRS})
    endif()

    set(_otel_lib_names
        opentelemetry_metrics
        opentelemetry_common
        opentelemetry_resources
        opentelemetry_version
        opentelemetry_logs
        opentelemetry_trace
        opentelemetry_otlp_recordable
        opentelemetry_exporter_otlp_http_metric
        opentelemetry_exporter_otlp_http
        opentelemetry_exporter_otlp_http_client
        opentelemetry_http_client_curl
        opentelemetry_proto)

    set(_otel_libs)
    foreach(_name IN LISTS _otel_lib_names)
        string(REPLACE ":" "_" _var_name ${_name})
        string(REPLACE "/" "_" _var_name ${_var_name})
        set(_find_var "FIND_${_var_name}")
        find_library(${_find_var} NAMES ${_name} HINTS ${_otel_search_dirs})
        if(${_find_var})
            list(APPEND _otel_libs ${${_find_var}})
        endif()
    endforeach()

    find_package(Protobuf QUIET)

    # Skip find_package(CURL) due to broken CMake config in /usr/local/lib/cmake/CURL/
    # Go directly to manual library discovery
    if(TARGET protobuf::libprotobuf)
        list(APPEND _otel_libs protobuf::libprotobuf)
    else()
        find_library(PROTOBUF_LIB protobuf HINTS ${_otel_search_dirs} /usr/lib/x86_64-linux-gnu)
        if(PROTOBUF_LIB)
            list(APPEND _otel_libs ${PROTOBUF_LIB})
        endif()
    endif()

    # Manual CURL search - skip the broken CMake package
    find_library(CURL_LIB NAMES curl libcurl.so.4 libcurl.so.4.8.0 HINTS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib ${_otel_search_dirs})
    if(CURL_LIB)
        list(APPEND _otel_libs ${CURL_LIB})
        message(STATUS "Found CURL manually: ${CURL_LIB}")
    else()
        message(WARNING "CURL library not found")
    endif()

    if(_otel_libs)
        add_definitions(-DENABLE_OTEL)
        # Use plain signature consistently (it is already used above for CUDA)
        target_link_libraries(nccl-profiler-otel ${_otel_libs})

        # Ensure exporter objects are kept
        set(_force_names
            opentelemetry_exporter_otlp_http_metric
            opentelemetry_exporter_otlp_http
            opentelemetry_http_client_curl
            opentelemetry_proto
            opentelemetry_otlp_recordable
            opentelemetry_trace
            opentelemetry_logs
            opentelemetry_common)
        set(_force_libs)
        foreach(n IN LISTS _force_names)
            string(REPLACE ":" "_" _var_name ${n})
            string(REPLACE "/" "_" _var_name ${_var_name})
            set(_find_var "FIND_${_var_name}")
            if(DEFINED ${_find_var} AND ${_find_var})
                list(APPEND _force_libs ${${_find_var}})
            endif()
        endforeach()
        if(_force_libs)
            target_link_options(nccl-profiler-otel PRIVATE -Wl,--whole-archive ${_force_libs} -Wl,--no-whole-archive)
        endif()

        message(STATUS "Linked OpenTelemetry libs: ${_otel_libs}")
    else()
        message(WARNING "OpenTelemetry libraries not found; telemetry will fall back to logs")
    endif()
else()
    message(STATUS "OpenTelemetry SDK not found at build time, will use runtime detection")
endif()

# Set library properties
set_target_properties(nccl-profiler-otel PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "profiler_otel.h"
)

# Install library
install(TARGETS nccl-profiler-otel
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

# Print build information
message(STATUS "Building NCCL OTEL Profiler Plugin")
message(STATUS "GPU Platform: ${GPU_RUNTIME_TYPE}")
if(GPU_RUNTIME_TYPE STREQUAL "ROCM")
    message(STATUS "ROCm Path: ${ROCM_PATH}")
endif()
message(STATUS "NCCL/RCCL Path: ${NCCL_PATH}")
if(OTEL_FOUND)
    message(STATUS "OpenTelemetry: enabled")
    if(OTEL_INSTALL_PATH)
        message(STATUS "OpenTelemetry Path: ${OTEL_INSTALL_PATH}")
    endif()
else()
    message(STATUS "OpenTelemetry: disabled")
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Output: libnccl-profiler-otel.so")
