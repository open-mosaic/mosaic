# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

# NCCL OTEL Profiler Plugin Makefile
# Uses CMake for actual building
# Supports both NVIDIA (CUDA) and AMD (ROCm) GPUs

# Default target
.PHONY: all
all: build

# Build directories
BUILD_DIR = build
TEST_BUILD_DIR = tests/build

# CMake configuration
CMAKE_FLAGS ?= -DCMAKE_BUILD_TYPE=Release

# GPU Platform selection: CUDA, ROCM, or AUTO (default: AUTO)
# AUTO will try to detect the available platform
GPU_PLATFORM ?= AUTO

# Allow `make TRACE=1` to toggle CMake TRACE option
TRACE ?= 0
ifeq ($(filter 1 ON on true TRUE yes YES,$(TRACE)),)
  TRACE_FLAG := OFF
else
  TRACE_FLAG := ON
endif

# Allow `make COVERAGE=1` to build tests with gcov coverage instrumentation
COVERAGE ?= 0
ifeq ($(filter 1 ON on true TRUE yes YES,$(COVERAGE)),)
  COVERAGE_FLAG := OFF
else
  COVERAGE_FLAG := ON
endif

# Allow `make PROFILE=1` to add frame pointers and debug info for perf sampling.
PROFILE ?= 0
ifeq ($(filter 1 ON on true TRUE yes YES,$(PROFILE)),)
	PROFILE_FLAG := OFF
else
	PROFILE_FLAG := ON
endif

COVERAGE_BUILD_DIR = tests/build-coverage

# ROCm path (default: /opt/rocm)
ROCM_PATH ?= /opt/rocm

# OpenTelemetry C++ SDK path (optional, for custom installations)
OTEL_PATH ?=

# Only pass ROCM_PATH to CMake when relevant, to avoid CMake warnings on CUDA builds.
# Treat ROCM_PATH as user-provided only when set via environment or command line.
ROCM_PATH_ORIGIN := $(origin ROCM_PATH)
ROCM_PATH_USER_SET := $(filter environment command line,$(ROCM_PATH_ORIGIN))
ifeq ($(GPU_PLATFORM),ROCM)
  ROCM_PATH_CMAKE_ARG := -DROCM_PATH=$(ROCM_PATH)
else ifneq ($(strip $(ROCM_PATH_USER_SET)),)
  ROCM_PATH_CMAKE_ARG := -DROCM_PATH=$(ROCM_PATH)
else
  ROCM_PATH_CMAKE_ARG :=
endif


# Configure and build using CMake
.PHONY: configure
configure: $(BUILD_DIR)/Makefile

$(BUILD_DIR)/Makefile: CMakeLists.txt
	@echo "Configuring NCCL OTEL Profiler Plugin..."
	@mkdir -p $(BUILD_DIR)
	@if [ -z "$(NCCL_PATH)" ]; then \
		echo "WARNING: NCCL_PATH is not set. Set NCCL_PATH to your NCCL/RCCL source/build tree (or source config.local)."; \
	fi
	@if [ -n "$(OTEL_PATH)" ]; then \
		cd $(BUILD_DIR) && cmake .. $(CMAKE_FLAGS) -DNCCL_PATH=$(NCCL_PATH) -DTRACE=$(TRACE_FLAG) -DPROFILER_ENABLE_SAMPLING_PROFILE=$(PROFILE_FLAG) -DGPU_PLATFORM=$(GPU_PLATFORM) $(ROCM_PATH_CMAKE_ARG) -DOTEL_PATH=$(OTEL_PATH); \
	else \
		cd $(BUILD_DIR) && cmake .. $(CMAKE_FLAGS) -DNCCL_PATH=$(NCCL_PATH) -DTRACE=$(TRACE_FLAG) -DPROFILER_ENABLE_SAMPLING_PROFILE=$(PROFILE_FLAG) -DGPU_PLATFORM=$(GPU_PLATFORM) $(ROCM_PATH_CMAKE_ARG); \
	fi

.PHONY: build
build: configure
	@echo "Building NCCL OTEL Profiler Plugin..."
	@$(MAKE) -C $(BUILD_DIR) -j$(shell nproc)
	@echo "Build complete: $(BUILD_DIR)/libnccl-profiler-otel.so"

.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -rf $(TEST_BUILD_DIR)
	@rm -rf $(COVERAGE_BUILD_DIR)
	@echo "Clean complete"

# Install the library
.PHONY: install
install: build
	@echo "Installing NCCL OTEL Profiler Plugin..."
	@$(MAKE) -C $(BUILD_DIR) install

# Test targets
.PHONY: test-configure
test-configure: $(TEST_BUILD_DIR)/Makefile

$(TEST_BUILD_DIR)/Makefile: tests/CMakeLists.txt
	@echo "Configuring NCCL OTEL Profiler Plugin Tests..."
	@mkdir -p $(TEST_BUILD_DIR)
	@if [ -z "$(NCCL_PATH)" ]; then \
		echo "WARNING: NCCL_PATH is not set. Tests may not compile correctly (or source config.local)."; \
	fi
	@if [ -n "$(NCCL_PATH)" ]; then \
		cd $(TEST_BUILD_DIR) && cmake .. -DNCCL_PATH=$(NCCL_PATH) -DTRACE=$(TRACE_FLAG) -DPROFILER_ENABLE_SAMPLING_PROFILE=$(PROFILE_FLAG) -DGPU_PLATFORM=$(GPU_PLATFORM) $(ROCM_PATH_CMAKE_ARG); \
	else \
		cd $(TEST_BUILD_DIR) && cmake .. -DTRACE=$(TRACE_FLAG) -DPROFILER_ENABLE_SAMPLING_PROFILE=$(PROFILE_FLAG) -DGPU_PLATFORM=$(GPU_PLATFORM) $(ROCM_PATH_CMAKE_ARG); \
	fi

.PHONY: test-build
test-build: test-configure
	@echo "Building NCCL OTEL Profiler Plugin Tests..."
	@$(MAKE) -C $(TEST_BUILD_DIR) -j$(shell nproc)
	@echo "Test build complete: $(TEST_BUILD_DIR)/nccl_profiler_otel_tests"
	@echo "Test build complete: $(TEST_BUILD_DIR)/nccl_profiler_otel_telemetry_tests"

.PHONY: test
test: test-build
	@echo "Running NCCL OTEL Profiler Plugin Tests..."
	@cd $(TEST_BUILD_DIR) && ./nccl_profiler_otel_tests
	@cd $(TEST_BUILD_DIR) && ./nccl_profiler_otel_telemetry_tests
	@echo "Tests complete."

.PHONY: test-verbose
test-verbose: test-build
	@echo "Running NCCL OTEL Profiler Plugin Tests (verbose)..."
	@$(TEST_BUILD_DIR)/nccl_profiler_otel_tests
	@$(TEST_BUILD_DIR)/nccl_profiler_otel_telemetry_tests
	@echo "Tests complete"

# Coverage targets (uses a separate build directory to avoid mixing with regular test builds)
.PHONY: coverage-configure
coverage-configure: $(COVERAGE_BUILD_DIR)/Makefile

$(COVERAGE_BUILD_DIR)/Makefile: tests/CMakeLists.txt
	@echo "Configuring tests with coverage instrumentation..."
	@mkdir -p $(COVERAGE_BUILD_DIR)
	@if [ -n "$(NCCL_PATH)" ]; then \
		cd $(COVERAGE_BUILD_DIR) && cmake .. -DNCCL_PATH=$(NCCL_PATH) -DTRACE=$(TRACE_FLAG) -DGPU_PLATFORM=$(GPU_PLATFORM) $(ROCM_PATH_CMAKE_ARG) -DCODE_COVERAGE=ON; \
	else \
		cd $(COVERAGE_BUILD_DIR) && cmake .. -DTRACE=$(TRACE_FLAG) -DGPU_PLATFORM=$(GPU_PLATFORM) $(ROCM_PATH_CMAKE_ARG) -DCODE_COVERAGE=ON; \
	fi

.PHONY: coverage-build
coverage-build: coverage-configure
	@echo "Building tests with coverage..."
	@$(MAKE) -C $(COVERAGE_BUILD_DIR) -j$(shell nproc)

.PHONY: coverage
coverage: coverage-build
	@echo "Running tests with coverage..."
	@cd $(COVERAGE_BUILD_DIR) && ./nccl_profiler_otel_tests
	@cd $(COVERAGE_BUILD_DIR) && ./nccl_profiler_otel_telemetry_tests
	@echo "Collecting coverage data..."
	@if command -v lcov > /dev/null 2>&1; then \
		lcov --capture --directory $(COVERAGE_BUILD_DIR) --output-file $(COVERAGE_BUILD_DIR)/coverage.info --rc branch_coverage=0 --ignore-errors mismatch,inconsistent,negative,unused --quiet; \
		lcov --remove $(COVERAGE_BUILD_DIR)/coverage.info '*/tests/*' '*/mocks/*' '/usr/*' '*/googletest/*' \
			--output-file $(COVERAGE_BUILD_DIR)/coverage.info --rc branch_coverage=0 --ignore-errors mismatch,inconsistent,negative,unused --quiet; \
		echo ""; \
		echo "=== Coverage Summary ==="; \
		lcov --list $(COVERAGE_BUILD_DIR)/coverage.info --rc branch_coverage=0; \
		echo ""; \
		echo "Coverage info: $(COVERAGE_BUILD_DIR)/coverage.info"; \
		if command -v genhtml > /dev/null 2>&1; then \
			genhtml $(COVERAGE_BUILD_DIR)/coverage.info --output-directory $(COVERAGE_BUILD_DIR)/coverage_html --rc branch_coverage=0 --ignore-errors mismatch,inconsistent,negative,unused --quiet; \
			echo "HTML report:   $(COVERAGE_BUILD_DIR)/coverage_html/index.html"; \
		fi; \
	elif command -v python > /dev/null 2>&1 && python -c "import gcovr" > /dev/null 2>&1; then \
		mkdir -p $(COVERAGE_BUILD_DIR)/coverage_html; \
		python -m gcovr --root . --object-directory $(COVERAGE_BUILD_DIR) --exclude '^tests/' \
			--gcov-ignore-parse-errors=negative_hits.warn_once_per_file --txt-summary \
			--lcov $(COVERAGE_BUILD_DIR)/coverage.info; \
		python -m gcovr --root . --object-directory $(COVERAGE_BUILD_DIR) --exclude '^tests/' \
			--gcov-ignore-parse-errors=negative_hits.warn_once_per_file \
			--html-details $(COVERAGE_BUILD_DIR)/coverage_html/index.html > /dev/null; \
		echo "Coverage info: $(COVERAGE_BUILD_DIR)/coverage.info"; \
		echo "HTML report:   $(COVERAGE_BUILD_DIR)/coverage_html/index.html"; \
	else \
		echo "ERROR: neither lcov nor python -m gcovr is available in the current environment."; \
		exit 1; \
	fi

.PHONY: coverage-clean
coverage-clean:
	@echo "Cleaning coverage build artifacts..."
	@rm -rf $(COVERAGE_BUILD_DIR)
	@echo "Coverage clean complete"

# Help target
.PHONY: help
help:
	@echo "NCCL OTEL Profiler Plugin Build System"
	@echo ""
	@echo "Supports both NVIDIA (CUDA) and AMD (ROCm) GPUs"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build the plugin (default)"
	@echo "  build          - Build the plugin"
	@echo "  configure      - Configure CMake build"
	@echo "  test           - Build and run unit tests"
	@echo "  test-build     - Build unit tests only"
	@echo "  coverage       - Build and run tests with gcov coverage, generate lcov report"
	@echo "  coverage-clean - Clean coverage build artifacts"
	@echo "  clean          - Clean build artifacts (including tests and coverage)"
	@echo "  install        - Install the library"
	@echo "  help           - Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  NCCL_PATH      - Path to NCCL/RCCL installation (default: $(NCCL_PATH))"
	@echo "  GPU_PLATFORM   - GPU platform: CUDA, ROCM, or AUTO (default: $(GPU_PLATFORM))"
	@echo "  ROCM_PATH      - Path to ROCm installation (default: $(ROCM_PATH))"
	@echo "  OTEL_PATH      - Path to OpenTelemetry C++ SDK installation (optional)"
	@echo "  CMAKE_FLAGS    - Additional CMake flags (default: $(CMAKE_FLAGS))"
	@echo "  TRACE          - Enable trace logging (default: $(TRACE), 0=off, 1=on)"
	@echo "  COVERAGE       - Enable test coverage instrumentation (default: $(COVERAGE), 0=off, 1=on)"
	@echo "  PROFILE        - Enable frame pointers/debug info for perf sampling (default: $(PROFILE), 0=off, 1=on)"
	@echo ""
	@echo "Examples:"
	@echo "  make                                      # Build with auto-detected platform"
	@echo "  make NCCL_PATH=/path/to/nccl              # Build for NVIDIA with custom NCCL path"
	@echo "  make GPU_PLATFORM=ROCM NCCL_PATH=/opt/rocm/rccl  # Build for AMD ROCm"
	@echo "  make GPU_PLATFORM=CUDA                    # Force CUDA platform"
	@echo "  make OTEL_PATH=/opt/opentelemetry-build   # Use custom OpenTelemetry installation"
	@echo "  make CMAKE_FLAGS=-DCMAKE_BUILD_TYPE=Debug # Debug build"
	@echo "  make PROFILE=1 build test                 # Build plugin/tests for perf profiling"
	@echo "  make test                                 # Build and run unit tests"
	@echo "  make coverage                             # Build and run tests with coverage report"
	@echo "  make clean                                # Clean all build artifacts"
