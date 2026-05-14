# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)

project(nccl-profiler-otel-plugin VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ProfilerCommon.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ProfilerSources.cmake)

# Build configuration
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

profiler_set_platform_defaults()
profiler_detect_gpu_runtime("")
profiler_configure_trace("" "PROFILER OTEL TRACE enabled (TRACE=1)"
                         "PROFILER OTEL TRACE disabled (build without TRACE=1)")
profiler_configure_sampling_profile("")
profiler_resolve_nccl_paths("" TRUE)

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


profiler_prefix_sources(OTEL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}"
                                                ${PROFILER_CORE_SOURCE_FILES}
                                                ${PROFILER_TELEMETRY_SOURCE_FILES})

# Create shared library
add_library(nccl-profiler-otel SHARED ${OTEL_SOURCES})

target_include_directories(nccl-profiler-otel PRIVATE
    ${PROFILER_GPU_INCLUDE_DIRS}
    ${PROFILER_NCCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(nccl-profiler-otel PRIVATE
    NCCL_OS_LINUX
    ${PROFILER_GPU_COMPILE_DEFINITIONS}
    ${PROFILER_TRACE_COMPILE_DEFINITIONS})

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
