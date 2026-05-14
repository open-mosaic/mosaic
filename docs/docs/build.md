---
icon: fontawesome/solid/code
title: Develop
---

<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# Build Profiler Plugin

## Build Options

| Aspect | Option 1: Build with Docker | Option 2: Build with Make |
| --- | --- | --- |
| **Use case** | Production / integration.<br>Ready to run with Ray and vLLM | Local development and custom workloads |
| **Workload** | Reference Workload (Ray & vLLM) | Any NCCL workload (e.g. `nccl_test`) |
| **Ready to use** | Yes - image includes plugin, NCCL,<br>OTel SDK, and runtime | No - You build the plugin,<br>and install NCCL, CUDA, OTel SDK on the host |
| **Portability** | Self-contained image.<br>Same environment everywhere | Depends on host<br>NCCL/CUDA/OTel versions |
| **Customization** | Rebuild Docker image<br>with other workload | Full control over paths,<br>compiler, and dependencies |

## Option 1: Build with Docker

### Prerequisites

- **NVIDIA Container Toolkit**

### Build Docker image

```bash
cd profiler
docker build -t openmosaic/mosaic-vllm:dev -f Dockerfile .
```

Run Docker image

```bash
docker run -it --rm openmosaic/mosaic-vllm:dev ray ...
```

## Option 2: Build with Make

### Prerequisites

- **NVIDIA CUDA** 13.1 or compatible versions
- **NCCL** 2.29.2-1 or compatible versions
- The following apt packages:

    ``` bash
    sudo apt update
    sudo apt install build-essential cmake git \
      libgrpc++-dev libprotobuf-dev libssl-dev libcurl4-openssl-dev \
      protobuf-compiler protobuf-compiler-grpc
    ```

- Additional packages if you want to run NCCL tests later:

    ``` bash
    sudo apt install openmpi-bin libopenmpi-dev
    ```

### Build OpenTelemetry C++ SDK

The plugin requires the OpenTelemetry C++ SDK. Install it as follows:

!!! note
    Update `CMAKE_INSTALL_PREFIX` to the desired installation location as needed

``` bash
# Clone the C++ SDK from GitHub
git clone --branch v1.25.0 --depth 1 https://github.com/open-telemetry/opentelemetry-cpp.git

# Change to the OpenTelemetry C++ SDK directory
cd opentelemetry-cpp

# Create a build directory and navigate into it
mkdir build && cd build

# Configure CMake with required options:
cmake \
    -DCMAKE_INSTALL_PREFIX=/tmp/opentelemetry-build
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DWITH_OTLP_GRPC=ON \
    -DWITH_OTLP_HTTP=ON \
    -DBUILD_TESTING=OFF \
    -DOPENTELEMETRY_INSTALL=ON \
    -DWITH_PROMETHEUS=ON \
    -DWITH_BENCHMARK=OFF \
    ..

# Build the SDK
cmake --build . -j$(nproc)

# Install OpenTelemetry C++ into the `opentelemetry-build` directory:
cmake --install .

# Copy OpenTelemetry headers and library files to system include directory
sudo cp -r /tmp/opentelemetry-build/include/* /usr/local/include/
sudo cp -r /tmp/opentelemetry-build/lib/* /usr/local/lib/

# Refresh the dynamic linker run-time bindings
sudo ldconfig
```

### Build profiler plugin

The project includes a Makefile that wraps CMake for easier building:

```bash
# Basic build
make

# Build with debug trace logging enabled
make TRACE=1

# Build with custom NCCL path
make NCCL_PATH=/path/to/nccl

# build with custom OTEL path
make OTEL_PATH=/path/to/opentelemetry-build

# Clean build artifacts
make clean

# Install the plugin
make install
```

The build process creates:

- `build/libnccl-profiler-otel.so` - The main plugin shared library
- `build/libnccl-profiler-otel.so.1` - Versioned symlink
- `build/libnccl-profiler-otel.so.1.0.0` - Full versioned library

# Deploy Profiler Plugin

This section includes instruction to deploy profiler plugin with LGTM stack + Pipeline Analyzer.

The `deployments/` directory contains a ready-to-run local observability stack:

- **LGTM stack** via `grafana/otel-lgtm` (Grafana + OTel Collector + Prometheus)
- **Pipeline Analyzer**: a small HTTP service that reads Prometheus metrics and helps analyze pipeline traffic

## Deploy with Docker Compose

From the repo root:

```bash
cd deployments

# First time (or after changing pipeline-analyzer image requirements)
docker compose build

# Start the stack
docker compose up
```

Once it is ready, you can access the following components:

- **Grafana UI**: `http://localhost:3000`
    - Dashboards are auto-provisioned from `deployments/dashboards/` (see `deployments/dashboards/dashboards.yml`)
- **OTLP endpoints** (for the profiler plugin):
    - HTTP: `localhost:4318`
- **Prometheus (debug)**:
    - Metrics endpoint: `http://localhost:8889/metrics`
    - Web UI: `http://localhost:9090`
- **Pipeline Analyzer API**: `http://localhost:55001`

## Configure OpenTelemetry Collector

By default the plugin exports OTLP/HTTP to `http://localhost:4318`, which matches the compose stack.
If you need to set it explicitly:

```bash
export NCCL_PROFILER_OTEL_TELEMETRY_ENDPOINT=http://localhost:4318
```

# Environment Variables

The plugin uses several environment variables for configuration. All variables use the `NCCL_` prefix to follow NCCL conventions.

## Required

- `NCCL_PROFILER_PLUGIN`
    - **Description**: Specifies which profiler plugin to use
    - **Value**: `otel`
    - **Example**: `export NCCL_PROFILER_PLUGIN=otel`

    !!! note
        The NCCL_PROFILER_PLUGIN environment variable must match the name of the profiler shared library.
        See [Plugin name and supporting multiple profiler plugins](https://github.com/NVIDIA/nccl/tree/master/ext-profiler#plugin-name-and-supporting-multiple-profiler-plugins)
        for more details

## Optional

### Telemetry Endpoint Configuration
- **`NCCL_PROFILER_OTEL_TELEMETRY_ENDPOINT`** (default: `http://localhost:4318`)
    - OpenTelemetry collector base URL (the plugin automatically appends `/v1/metrics`)
    - Format: `http://host:port` or `https://host:port`
    - Example: `export NCCL_PROFILER_OTEL_TELEMETRY_ENDPOINT=http://otel-collector:4318`

### Plugin Control
- **`NCCL_PROFILER_OTEL_ENABLE`** (default: `1`)
    - Enable/disable the plugin entirely
    - Set to `0` to disable the plugin
    - Example: `export NCCL_PROFILER_OTEL_ENABLE=1`

### Telemetry Configuration
- **`NCCL_PROFILER_OTEL_TELEMETRY_ENABLE`** (default: `1`)
    - Enable/disable telemetry collection
    - Set to `0` to disable telemetry while keeping plugin active
    - Example: `export NCCL_PROFILER_OTEL_TELEMETRY_ENABLE=1`

- **`NCCL_PROFILER_OTEL_TELEMETRY_INTERVAL_SEC`** (default: `5`)
    - Reporting interval in seconds
    - Controls how often metrics are exported to the collector
    - Example: `export NCCL_PROFILER_OTEL_TELEMETRY_INTERVAL_SEC=10`

- **`NCCL_PROFILER_OTEL_TELEMETRY_BATCH_TIMEOUT_MS`** (default: `3000`)
    - Batch timeout in milliseconds
    - Maximum time to wait before sending a batch of metrics
    - Example: `export NCCL_PROFILER_OTEL_TELEMETRY_BATCH_TIMEOUT_MS=5000`

### Linear Regression Configuration
- **`NCCL_PROFILER_LINEAR_REGRESSION_MODE`** (default: `AVG`)
    - Controls how latency/rate are estimated from transfer size/time samples
    - Supported values:
        - `AVG`: use all points as observed
        - `MIN`: for each transfer size, keep the minimum time observed (more robust to jitter)
    - Example: `export NCCL_PROFILER_LINEAR_REGRESSION_MODE=AVG`

### Event Filtering
- **`NCCL_PROFILE_EVENT_MASK`** (auto-configured to `0x1E` if not set)
    - Bitmask to control which NCCL events are profiled
    - Available event types:
        - `0x02`: Collective operations (ncclProfileColl)
        - `0x04`: Point-to-point operations (ncclProfileP2p)
        - `0x08`: Proxy operations (ncclProfileProxyOp)
        - `0x10`: Proxy step operations (ncclProfileProxyStep)
    - Default auto-configuration: `0x1E` (Coll + P2P + ProxyOp + ProxyStep)
    - Example: `export NCCL_PROFILE_EVENT_MASK=0x1E`
    - Note: The plugin automatically configures this mask unless explicitly set. `ncclProfileProxyCtrl` (0x20) is not tracked as it doesn't contribute to metrics. `ncclProfileGroup` (0x01) is supported (a handle is returned for correct parent-chain behavior) but it is not exported as a metric.

### Library Path Configuration
- **`LD_LIBRARY_PATH`**
    - Add the plugin directory to the library search path
    - Example: `export LD_LIBRARY_PATH=/usr/local/plugins/lib:$LD_LIBRARY_PATH`

# Usage Examples

## Basic Configuration

1. **Set up environment variables**:
   ```bash
   export NCCL_PROFILER_PLUGIN=otel
   # Optional: customize endpoint (defaults to http://localhost:4318)
   # export NCCL_PROFILER_OTEL_TELEMETRY_ENDPOINT=http://otel-collector:4318
   export LD_LIBRARY_PATH=/usr/local/plugins/lib:$LD_LIBRARY_PATH
   ```

2. **Run your NCCL application**:
   ```bash
   ./your_nccl_application
   ```

## Advanced Configuration

**High-frequency monitoring**:
```bash
export NCCL_PROFILER_PLUGIN=otel
export NCCL_PROFILER_OTEL_TELEMETRY_ENDPOINT=http://otel-collector:4318
export NCCL_PROFILER_OTEL_TELEMETRY_INTERVAL_SEC=1
export LD_LIBRARY_PATH=/usr/local/plugins/lib:$LD_LIBRARY_PATH
```

**Debug mode with trace logging**:
```bash
# Build with trace logging enabled
make TRACE=1

# Set environment variables
export NCCL_PROFILER_PLUGIN=otel
export NCCL_PROFILER_OTEL_TELEMETRY_ENDPOINT=http://localhost:4318
export NCCL_PROFILER_OTEL_ENABLE=1
export NCCL_DEBUG=INFO
export LD_LIBRARY_PATH=/usr/local/plugins/lib:$LD_LIBRARY_PATH
```

# Unit Tests

The plugin includes unit tests in the `profiler/tests/` directory:

- `test_aggregation.cc`: Tests for event aggregation logic
- `test_communicator_state.cc`: Tests for circular buffer management
- `test_linear_regression.cc`: Tests for linear regression calculations
- `test_profiler_otel.cc`: Tests for plugin interface functions
- `test_profiler_utils.cc`: Tests for utility functions (ncclTypeSize, gpuUuidToString)
- `test_params.cc`: Tests for environment variable parameter loading
- `test_profiler_events.cc`: Tests for event handling (Coll, P2P, ProxyOp, ProxyStep, Group)
- `test_edge_cases.cc`: Tests for edge cases and error handling
- `test_race_conditions.cc`: Tests for thread safety

Run tests with:
```bash
make test
```

# Documentation

All functions in the codebase are documented with Doxygen-style comments, including:

- Function descriptions
- Parameter documentation (`@param[in]`, `@param[out]`)
- Return value documentation (`@return`)
- Notes about thread safety, usage, and implementation details

To generate documentation, use Doxygen:
```bash
doxygen Doxyfile  # If Doxyfile exists
# Or use default Doxygen configuration
doxygen -g Doxyfile
doxygen Doxyfile
```

# Contribution Tips
  When adding new features or modifying existing code:

  1. Add Doxygen documentation headers to all new functions
  2. Update this README if architecture or usage changes
  3. Add unit tests for new functionality
  4. Ensure thread safety for lock-free operations
  5. Follow the existing code style and conventions
