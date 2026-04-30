---
icon: fontawesome/solid/circle-check
title: Validation
---

<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# Test Strategy

Mosaic uses a **layered test strategy** to balance fast developer feedback with end-to-end confidence. We separate **unit tests** (fast, GPU-free) from **integration tests** (real deployment, GPU-required).

## Unit Tests
Unit tests act as **merge guardrails** and do not require GPUs.

=== "What"
    - Profiler plugin builds and passes unit tests (`gtest`)
    - Container images builds
    - Code quality checks
        - Formatting (`clang-format`)
        - License headers (`reuse`)

=== "When"
    - Every pull request
    - Every commit to `main`

=== "Where"
    - Standard GitHub runners (no GPU required)

## Integration Tests

Integration tests validate **real-world, end-to-end behavior**.

**Initial focus**: Minimal 2-GPU setup with NVIDIA GPUs

=== "What"
    - Provision the test environment on the self-hosted GPU runner
    - Deploy Mosaic observability stack (LGTM + dashboards) with Docker compose
    - Run a reference workload (vLLM + profiler plugin) with Docker compose
    - Verify metrics are emitted from profiler plugin and received by LGTM stack
        ``` bash title="Query should return positive array size"
        MIMIR_IP="x.x.x.x"
        NOW=$(date +%s)
        METRIC="nccl_profiler_collective_count_sum"

        curl -s "http://${MIMIR_IP}:9090/api/v1/query_range?"\
        "query=${METRIC}&start=$((NOW-30))&end=${NOW}&step=15" |\
        jq '.data.result | length'
        ```

=== "When"
    - Nightly
    - Maintainer-triggered

=== "Where"
    - Self-hosted GPU runners (GPU and container runtime required)

# Run tests

## Unit Tests

```bash title="Run profiler unit test"
mkdir -p tests/build
cd tests/build
cmake .. \
  -DNCCL_PATH=$NCCL_PATH \
  -DGPU_PLATFORM=AUTO \
  -DTRACE=OFF
cmake --build . --parallel $(nproc)
./nccl_profiler_otel_tests
```

## Integration Tests

The integration tests validate the complete integration of all Open Mosaic components.
They use the [Production Test Framework](https://github.com/open-mosaic/production-test-framework) to control the test environment and execute the tests.

### Prerequisites

- **make** — Required to run the Makefile targets.
- **Docker** and **Docker Compose** — Required to run the stack and test container.
- **production-test-framework Docker image** — This image will be automatically pulled from Docker Hub or it can be built from the [Production Test Framework](https://github.com/open-mosaic/production-test-framework) project.

### Integration test commands

```bash title="Run the integration tests"
cd tests
# results are in JUnitXML and located at ${HOME}/tmp/mosaic/test_results.xml
make test
```

```bash title="Stand up the container stack without running tests"
make setup
```

```bash title="Run tests against an already running container stack"
make run-tests
```

```bash title="Shut down and clean up the container stack"
make cleanup
```

### Advanced usage

#### Interactive Shell

The production test framework container has an interactive shell that can be used for running one-off tests and troubleshooting problems with the tests or framework.
To enter the shell, remove the `RUN_MAKE_TARGET` environment variable from `tests/docker-compose.yml`:
```diff
  production-test-framework:
    profiles: ["test"]
    image: production-test-framework:latest
    container_name: production-test-framework
    environment:
      - MOSAIC_PATH=${MOSAIC_PATH}
-     - RUN_MAKE_TARGET=profiler-otel-test-only
      - PYTEST_ADDOPTS="--junitxml=/app/results/results.xml"
```

Running the `make test` command will start the container stack,
and you will be dropped into a shell in the `production-test-framework` container that displays a help message for the most common testing targets.

#### Docker Privilege

The host's Docker socket can be mounted into the container to run `docker` commands or perform other container stack analysis by adding a bind mount to `tests/docker-compose.yml`:

```diff
  production-test-framework:
    profiles: ["test"]
    image: production-test-framework:latest
    container_name: production-test-framework
    environment:
      - MOSAIC_PATH=${MOSAIC_PATH}
      - PYTEST_ADDOPTS="--junitxml=/app/results/results.xml"
    network_mode: "host"
    volumes:
      - ./suites:/app/tests:ro
      - test-results:/app/results
+     - /var/run/docker.sock:/var/run/docker.sock
```

When the container stack is started, you will be able to run docker commands as `sudo` from inside the `production-test-framework` shell prompt:

```bash
/app/framework $ sudo docker ps --format "{{.Names}}"
production-test-framework
mosaic-vllm
mosaic-pipeline-analyzer
mosaic-otel-lgtm
/app/framework $
```
