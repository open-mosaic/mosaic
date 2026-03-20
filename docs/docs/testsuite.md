---
icon: fontawesome/solid/list-check
title: Test Suites
---

<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# NCCL Profiler OpenTelemetry Tests Suite

The NCCL Profiler Open Telemetry test suite tests the NCCL Profiler OpenTelemetry plugin's telemetry export functionality. These tests verify that:
1. The LGTM stack (Loki, Grafana, Tempo, Mimir) and OTel Collector are accessible
2. vLLM inference triggers NCCL operations
3. The NCCL profiler exports metrics to Prometheus via OpenTelemetry

## Hardware Requirements

**Minimum 2 NVIDIA GPUs required.**
These tests validate NCCL profiler metrics, which are only generated when multi-GPU parallelism (tensor or pipeline) triggers inter-GPU communication.
A single GPU will not produce NCCL metrics.

```bash title="Verify GPU availability"
nvidia-smi
```

## Software Requirements

Before running these tests, ensure the following are running:

1. **LGTM Stack and OTel Collector**
2. **vLLM** with NCCL Profiler plugin enabled and tensor parallelism (`--tensor-parallel-size 2`) or pipeline parallelism (`--pipeline-parallel-size 2`)

Using the `production-test-framework` container will automate a lot of the managing the lifecycle of these services, but they can also be managed manually.

## Test Structure

### Shared fixtures

The `grafana_url` fixture is defined in `tests/suites/conftest.py` and is used by both the profiler OTEL and Grafana dashboards suites. It uses `GRAFANA_HOST` and `GRAFANA_PORT` (default: `http://localhost:3000`).

### Profiler OTEL `conftest.py`

Provides fixtures and constants for the profiler OTEL tests:

| Fixture | Description |
|---------|-------------|
| `prometheus_url` | Prometheus API endpoint (default: `http://localhost:9090`) |
| `grafana_url` | Grafana endpoint (shared; default: `http://localhost:3000`) |
| `vllm_client` | Client for vLLM inference API |
| `vllm_ready` | Waits for vLLM to be healthy |
| `inference_completed` | Runs inference to trigger NCCL operations |
| `nccl_profiler_metrics` | List of expected Prometheus metric names |

### `test_profiler_metrics.py`

Contains the test class `TestNCCLProfilerTelemetry`:

| Test | Description |
|------|-------------|
| `test_otel_collector_accessible` | Verifies Prometheus endpoint is reachable |
| `test_grafana_accessible` | Verifies Grafana dashboard is reachable |
| `test_nccl_metrics_exported_after_inference` | Runs inference and validates all expected NCCL metrics appear in Prometheus |

## Expected Metrics

The tests validate that the following metrics are exported to Prometheus (defined in `telemetry.cc`):

### Collective Metrics
- `nccl_profiler_collective_bytes_total` - Total bytes in collective ops
- `nccl_profiler_collective_time_microseconds_sum` - Time spent in collective ops
- `nccl_profiler_collective_count_sum` - Number of collective ops

    !!! notes
        Collective metrics appear with tensor parallelism (`--tensor-parallel-size`).

### P2P Metrics
- `nccl_profiler_p2p_bytes_bytes_sum` - Bytes in P2P ops
- `nccl_profiler_p2p_time_microseconds_sum` - Time in P2P ops

    !!! notes
        P2P metrics only appear when using pipeline parallelism (`--pipeline-parallel-size`).

### Rank/Transfer Metrics
- `nccl_profiler_rank_bytes_total` - Bytes transferred between ranks
- `nccl_profiler_rank_latency_microseconds_sum` - Latency between ranks
- `nccl_profiler_transfer_size_bytes_sum` - Transfer sizes per channel

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_HOST` | `localhost` | vLLM server host |
| `VLLM_PORT` | `8080` | vLLM server port |
| `PROMETHEUS_HOST` | `localhost` | Prometheus host |
| `PROMETHEUS_PORT` | `9090` | Prometheus port |
| `GRAFANA_HOST` | `localhost` | Grafana host |
| `GRAFANA_PORT` | `3000` | Grafana port |

## Troubleshooting

### No metrics appearing in Prometheus

1. Check vLLM logs for NCCL profiler initialization:
   ```bash
   make profiler-otel-logs
   ```

2. Verify the OTEL endpoint is correct:
   ```bash
   # Should be http://nccl-profiler-otel-lgtm:4318 (HTTP, not gRPC)
   docker inspect nccl-profiler-vllm | grep OTEL
   ```

3. Check Prometheus targets:
   ```text
   http://localhost:9090/targets
   ```

### Tests timeout waiting for vLLM

vLLM needs time to download and load the model. The default timeout is 5 minutes. Check logs:
```bash
docker logs nccl-profiler-vllm -f
```

### Missing P2P metrics

P2P metrics require pipeline parallelism. Modify `docker-compose.yml`:
```yaml
command: ["vllm serve ... --pipeline-parallel-size 2"]
```

---

## Grafana Dashboards Tests Suite

The Grafana dashboards test suite validates that the dashboards in the repository stay in sync with the running Grafana instance. It does not require vLLM or GPUs.

### What is tested

1. **YAML vs folder**: Every `*.json` file in `deployments/dashboards` is listed in `dashboards.yml` (no dashboard file is left out of provisioning).
2. **Dashboard presence in Grafana**: The dashboards listed in `dashboards.yml` are available in Grafana
3. **Each dashboard loads**: For each dashboard returned by Grafana’s search API, the dashboard UID API returns HTTP 200.

### Requirements

- Grafana (LGTM stack) must be running. When running via `make test`, the production-test-framework container mounts the repo’s `deployments/dashboards` directory at `/mnt/dashboards` so the tests can read `dashboards.yml` and the list of JSON files.
- The suite uses the same Grafana URL as the profiler OTEL suite (`GRAFANA_HOST`, `GRAFANA_PORT`; default `http://localhost:3000`).

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DASHBOARDS_DIR` | `/mnt/dashboards` | Path to the dashboards directory (set by mount when run in container) |
