# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)

project(nccl-profiler-otel-tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ProfilerCommon.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ProfilerSources.cmake)

profiler_set_platform_defaults()
profiler_detect_gpu_runtime("Tests: ")
profiler_configure_sampling_profile("Tests: ")

# Code coverage support. Enable with -DCODE_COVERAGE=ON.
option(CODE_COVERAGE "Build tests with gcov coverage instrumentation" OFF)
if(CODE_COVERAGE)
    message(STATUS "Tests: Code coverage enabled (--coverage)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()

# Optional TRACE toggle for tests (mirrors top-level). Enable with -DTRACE=ON.
profiler_configure_trace("Tests: " "OTEL TRACE enabled (TRACE=1)" "OTEL TRACE disabled")
profiler_resolve_nccl_paths("Tests: " FALSE)

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

profiler_prefix_sources(TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}" ${PROFILER_TEST_SOURCE_FILES})
profiler_prefix_sources(TELEMETRY_TEST_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}" ${PROFILER_TELEMETRY_TEST_SOURCE_FILES})

# Plugin sources needed for tests. telemetry.cc is excluded because test_mocks.cc provides the public mocks.
profiler_prefix_sources(PLUGIN_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/.." ${PROFILER_CORE_SOURCE_FILES})
profiler_prefix_sources(TELEMETRY_TEST_PLUGIN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/.."
    ${PROFILER_TELEMETRY_TEST_PLUGIN_SOURCE_FILES})

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

add_executable(nccl_profiler_otel_telemetry_tests ${TELEMETRY_TEST_SOURCE_FILES} ${TELEMETRY_TEST_PLUGIN_SOURCES})

target_include_directories(nccl_profiler_otel_tests PRIVATE
    ${PROFILER_GPU_INCLUDE_DIRS}
    ${PROFILER_NCCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mocks)

target_include_directories(nccl_profiler_otel_telemetry_tests PRIVATE
    ${PROFILER_GPU_INCLUDE_DIRS}
    ${PROFILER_NCCL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/mocks)

# Add compile definition for unit testing to disable dlopen
target_compile_definitions(nccl_profiler_otel_tests PRIVATE
    UNIT_TESTING=1
    NCCL_OS_LINUX
    ${PROFILER_GPU_COMPILE_DEFINITIONS}
    ${PROFILER_TRACE_COMPILE_DEFINITIONS})

target_compile_definitions(nccl_profiler_otel_telemetry_tests PRIVATE
    UNIT_TESTING=1
    ENABLE_OTEL=1
    NCCL_OS_LINUX
    ${PROFILER_GPU_COMPILE_DEFINITIONS}
    ${PROFILER_TRACE_COMPILE_DEFINITIONS})

# Link libraries to test executable
target_link_libraries(nccl_profiler_otel_tests pthread dl)
target_link_libraries(nccl_profiler_otel_telemetry_tests pthread dl)
if(GPU_RUNTIME_LIB)
    target_link_libraries(nccl_profiler_otel_tests ${GPU_RUNTIME_LIB})
    target_link_libraries(nccl_profiler_otel_telemetry_tests ${GPU_RUNTIME_LIB})
endif()

if(TARGET gtest AND TARGET gmock)
    target_link_libraries(nccl_profiler_otel_tests gtest gmock)
    target_link_libraries(nccl_profiler_otel_telemetry_tests gtest gmock)
elseif(TARGET GTest::gtest)
    target_link_libraries(nccl_profiler_otel_tests GTest::gtest)
    target_link_libraries(nccl_profiler_otel_telemetry_tests GTest::gtest)
    if (TARGET GTest::gmock)
        target_link_libraries(nccl_profiler_otel_tests GTest::gmock)
        target_link_libraries(nccl_profiler_otel_telemetry_tests GTest::gmock)
    elseif(GMOCK_LIB)
        target_link_libraries(nccl_profiler_otel_tests ${GMOCK_LIB})
        target_link_libraries(nccl_profiler_otel_telemetry_tests ${GMOCK_LIB})
    else()
        message(WARNING "GMock not found as imported target or library; gmock-based tests may fail")
    endif()
elseif(GTEST_LIBRARIES)
    target_link_libraries(nccl_profiler_otel_tests ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES})
    target_link_libraries(nccl_profiler_otel_telemetry_tests ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES})
else()
    target_link_libraries(nccl_profiler_otel_tests gtest gmock)
    target_link_libraries(nccl_profiler_otel_telemetry_tests gtest gmock)
endif()

# Add test
enable_testing()
add_test(NAME nccl_profiler_otel_tests COMMAND nccl_profiler_otel_tests)
add_test(NAME nccl_profiler_otel_telemetry_tests COMMAND nccl_profiler_otel_telemetry_tests)

# Print configuration
message(STATUS "Building NCCL OTEL Profiler Plugin Tests")
message(STATUS "Tests: GPU Platform: ${GPU_RUNTIME_TYPE}")
if(GPU_RUNTIME_TYPE STREQUAL "ROCM")
    message(STATUS "Tests: ROCm Path: ${ROCM_PATH}")
endif()
message(STATUS "Test executable: nccl_profiler_otel_tests")
message(STATUS "Test executable: nccl_profiler_otel_telemetry_tests")
