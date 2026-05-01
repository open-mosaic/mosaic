# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)

project(nccl-profiler-otel-tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Linux-only build: define NCCL_OS_LINUX unconditionally.
# Required because nccl_profiler.h includes os.h, which gates
# ncclSocketDescriptor / ncclAffinity / ncclPid_t behind NCCL_OS_LINUX.
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
function(find_cuda_runtime_tests)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "Tests: CUDA found at: ${CUDAToolkit_INCLUDE_DIRS}")
        include_directories(${CUDAToolkit_INCLUDE_DIRS})
        set(GPU_RUNTIME_FOUND TRUE PARENT_SCOPE)
        set(GPU_RUNTIME_TYPE "CUDA" PARENT_SCOPE)
        set(CUDAToolkit_FOUND TRUE PARENT_SCOPE)
        set(CUDAToolkit_LIBRARY_DIR "${CUDAToolkit_LIBRARY_DIR}" PARENT_SCOPE)
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
                message(STATUS "Tests: CUDA headers found at: ${CUDA_PATH}")
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
function(find_rocm_runtime_tests)
    # Check for ROCm HIP runtime
    set(ROCM_INCLUDE_PATH "${ROCM_PATH}/include")
    set(HIP_INCLUDE_PATH "${ROCM_PATH}/include/hip")

    if(EXISTS "${ROCM_INCLUDE_PATH}/hip/hip_runtime.h" OR EXISTS "${HIP_INCLUDE_PATH}/hip_runtime.h")
        message(STATUS "Tests: ROCm/HIP found at: ${ROCM_PATH}")
        include_directories(${ROCM_INCLUDE_PATH})
        if(EXISTS "${HIP_INCLUDE_PATH}")
            include_directories(${HIP_INCLUDE_PATH})
        endif()
        set(GPU_RUNTIME_FOUND TRUE PARENT_SCOPE)
        set(GPU_RUNTIME_TYPE "ROCM" PARENT_SCOPE)
        set(ROCM_PATH "${ROCM_PATH}" PARENT_SCOPE)
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
                message(STATUS "Tests: ROCm/HIP headers found at: ${ROCM_SEARCH_PATH}")
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
    message(STATUS "Tests: GPU Platform: CUDA (forced)")
    find_cuda_runtime_tests()
    if(NOT GPU_RUNTIME_FOUND)
        message(FATAL_ERROR "Tests: CUDA platform requested but CUDA headers not found.")
    endif()
elseif(GPU_PLATFORM STREQUAL "ROCM")
    message(STATUS "Tests: GPU Platform: ROCm (forced)")
    find_rocm_runtime_tests()
    if(NOT GPU_RUNTIME_FOUND)
        message(FATAL_ERROR "Tests: ROCm platform requested but ROCm/HIP headers not found at ${ROCM_PATH}")
    endif()
else()
    # AUTO detection: try CUDA first, then ROCm
    message(STATUS "Tests: GPU Platform: AUTO (detecting...)")
    find_cuda_runtime_tests()
    if(NOT GPU_RUNTIME_FOUND)
        find_rocm_runtime_tests()
    endif()

    if(NOT GPU_RUNTIME_FOUND)
        message(WARNING "Tests: No GPU runtime (CUDA or ROCm) found. Build may fail.")
    else()
        message(STATUS "Tests: Auto-detected GPU Platform: ${GPU_RUNTIME_TYPE}")
    endif()
endif()

# Code coverage support. Enable with -DCODE_COVERAGE=ON.
option(CODE_COVERAGE "Build tests with gcov coverage instrumentation" OFF)
if(CODE_COVERAGE)
    message(STATUS "Tests: Code coverage enabled (--coverage)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()

# Optional TRACE toggle for tests (mirrors top-level). Enable with -DTRACE=ON.
option(TRACE "Enable trace logging (OTEL_TRACE active) for test builds" OFF)
if(TRACE)
    add_definitions(-DPROFILER_OTEL_ENABLE_TRACE)
    message(STATUS "Tests: OTEL TRACE enabled (TRACE=1)")
else()
    message(STATUS "Tests: OTEL TRACE disabled")
endif()

# NCCL path detection (same as main CMakeLists.txt)
if(NOT DEFINED NCCL_PATH)
    set(NCCL_PATH $ENV{NCCL_PATH})
endif()

if(NCCL_PATH)
    message(STATUS "Tests: NCCL_PATH=${NCCL_PATH}")
    # NCCL include paths
    set(NCCL_SRC_INCLUDE_PATH ${NCCL_PATH}/src/include)
    set(NCCL_BUILD_INCLUDE_PATH ${NCCL_PATH}/build/include)
    set(NCCL_PLUGIN_INCLUDE_PATH ${NCCL_PATH}/src/include/plugin)

    # Include NCCL directories
    include_directories(${NCCL_SRC_INCLUDE_PATH})
    include_directories(${NCCL_BUILD_INCLUDE_PATH})
    include_directories(${NCCL_PLUGIN_INCLUDE_PATH})
else()
    message(WARNING "Tests: NCCL_PATH not set. Tests may not compile correctly.")
endif()

set(CMAKE_SKIP_BUILD_RPATH TRUE)
if(GPU_RUNTIME_TYPE STREQUAL "ROCM")
    set(CMAKE_BUILD_RPATH "${ROCM_PATH}/lib;${ROCM_PATH}/lib64;/usr/lib/x86_64-linux-gnu;/usr/lib")
    set(CMAKE_INSTALL_RPATH "${ROCM_PATH}/lib;${ROCM_PATH}/lib64;/usr/lib/x86_64-linux-gnu;/usr/lib")
else()
    set(CMAKE_BUILD_RPATH "/usr/lib/x86_64-linux-gnu;/usr/lib")
    set(CMAKE_INSTALL_RPATH "/usr/lib/x86_64-linux-gnu;/usr/lib")
endif()
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

# Find GoogleTest and GoogleMock
find_package(GTest QUIET)
if (NOT GTest_FOUND)
    # try to find it via pkg-config
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GTEST gtest)
        pkg_check_modules(GMOCK gmock)
    endif()

    if(NOT GTEST_FOUND)
        message(WARNING "GoogleTest not found. Attempting to download...")
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.12.1
        )
        FetchContent_MakeAvailable(googletest)
        include(GoogleTest)
        set(GTEST_LIBRARIES gtest gtest_main)
        set(GMOCK_LIBRARIES gmock gmock_main)
    endif()
endif()

# If GTest was found but GMock headers are missing, fetch googletest to provide GMock
find_library(GMOCK_LIB_NAMES gmock)
find_path(GMOCK_INCLUDE_DIR NAMES gmock/gmock.h)
if(NOT TARGET GTest::gmock AND NOT GMOCK_LIB AND NOT GMOCK_INCLUDE_DIR)
    message(WARNING "GMock headers not found. Attempting to download googletest for gmock...")
    include(FetchContent)
    FetchContent_Declare(
        googletest_missing_gmock
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest_missing_gmock)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mocks)

# Test sources
set(TEST_SOURCES
    test_main.cc
    test_mocks.cc
    unit/test_profiler_otel.cc
    unit/test_profiler_utils.cc
    unit/test_params.cc
    unit/test_communicator_state.cc
    unit/test_linear_regression.cc
    unit/test_aggregation.cc
    unit/test_profiler_events.cc
    unit/test_edge_cases.cc
    unit/test_race_conditions.cc
    unit/test_scale_up_inference.cc
    unit/test_v4_compat.cc
)

# Plugin sources needed for tests
# Note: telemetry.cc is excluded because we provide mocks in test_mocks.cc
set(PLUGIN_TEST_SOURCES
    ../nccl_plugin.cc
    ../profiler_otel.cc
    ../profiler_v4_compat.cc
    ../communicator_state.cc
    ../linear_regression.cc
    ../aggregation.cc
)

# Try to locate standalone gmock if CMake package lacks an imported target (re-run after fetch)
find_library(GMOCK_MAIN_LIB NAMES gmock_main)

# Find GPU runtime library (CUDA or ROCm/HIP)
if(GPU_RUNTIME_TYPE STREQUAL "CUDA")
    if(CUDAToolkit_FOUND)
        # Use CUDAToolkit imported targets if available
        find_library(GPU_RUNTIME_LIB cudart
            PATHS ${CUDAToolkit_LIBRARY_DIR}
            NO_DEFAULT_PATH)
        if(GPU_RUNTIME_LIB)
            message(STATUS "Tests: CUDA runtime library found: ${GPU_RUNTIME_LIB}")
        endif()
    else()
        # Fallback: search common CUDA library paths
        set(CUDA_LIB_PATHS
            "/usr/local/cuda/lib64"
            "/usr/local/cuda/lib"
            "/usr/local/cuda-13.0/lib64"
            "/usr/local/cuda-13/lib64"
            "/usr/local/cuda-12.9/lib64"
            "/usr/local/cuda-12/lib64"
            "/usr/local/cuda-11/lib64"
            "/opt/cuda/lib64"
            "/usr/lib/x86_64-linux-gnu"
        )
        find_library(GPU_RUNTIME_LIB cudart PATHS ${CUDA_LIB_PATHS})
        if(GPU_RUNTIME_LIB)
            message(STATUS "Tests: CUDA runtime library found: ${GPU_RUNTIME_LIB}")
        else()
            message(WARNING "Tests: CUDA runtime library (cudart) not found. Tests may fail to link.")
        endif()
    endif()
elseif(GPU_RUNTIME_TYPE STREQUAL "ROCM")
    # Search for HIP/ROCm runtime library
    set(ROCM_LIB_PATHS
        "${ROCM_PATH}/lib"
        "${ROCM_PATH}/lib64"
        "/opt/rocm/lib"
        "/opt/rocm/lib64"
        "/opt/rocm-6.0/lib"
        "/opt/rocm-5.7/lib"
        "/usr/lib/x86_64-linux-gnu"
    )
    # Try amdhip64 first (newer ROCm), then hip_hcc (older ROCm)
    find_library(GPU_RUNTIME_LIB NAMES amdhip64 hip_hcc PATHS ${ROCM_LIB_PATHS})
    if(GPU_RUNTIME_LIB)
        message(STATUS "Tests: ROCm/HIP runtime library found: ${GPU_RUNTIME_LIB}")
    else()
        message(WARNING "Tests: ROCm/HIP runtime library not found. Tests may fail to link.")
    endif()
endif()

# Create test executable with plugin sources
add_executable(nccl_profiler_otel_tests ${TEST_SOURCES} ${PLUGIN_TEST_SOURCES})

# Add compile definition for unit testing to disable dlopen
target_compile_definitions(nccl_profiler_otel_tests PRIVATE UNIT_TESTING=1)

# Link libraries to test executable
target_link_libraries(nccl_profiler_otel_tests pthread dl)
if(GPU_RUNTIME_LIB)
    target_link_libraries(nccl_profiler_otel_tests ${GPU_RUNTIME_LIB})
endif()

if(TARGET gtest AND TARGET gmock)
    target_link_libraries(nccl_profiler_otel_tests gtest gmock)
elseif(TARGET GTest::gtest)
    target_link_libraries(nccl_profiler_otel_tests GTest::gtest)
    if (TARGET GTest::gmock)
        target_link_libraries(nccl_profiler_otel_tests GTest::gmock)
    elseif(GMOCK_LIB)
        target_link_libraries(nccl_profiler_otel_tests ${GMOCK_LIB})
    else()
        message(WARNING "GMock not found as imported target or library; gmock-based tests may fail")
    endif()
elseif(GTEST_LIBRARIES)
    target_link_libraries(nccl_profiler_otel_tests ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES})
else()
    target_link_libraries(nccl_profiler_otel_tests gtest gmock)
endif()

# Add test
enable_testing()
add_test(NAME nccl_profiler_otel_tests COMMAND nccl_profiler_otel_tests)

# Print configuration
message(STATUS "Building NCCL OTEL Profiler Plugin Tests")
message(STATUS "Tests: GPU Platform: ${GPU_RUNTIME_TYPE}")
if(GPU_RUNTIME_TYPE STREQUAL "ROCM")
    message(STATUS "Tests: ROCm Path: ${ROCM_PATH}")
endif()
message(STATUS "Test executable: nccl_profiler_otel_tests")
