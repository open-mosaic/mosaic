# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
NCCL Profiler OTEL test fixtures.

This conftest provides fixtures and constants specific to NCCL profiler OTEL testing.
"""

import os
import time

import pytest
import requests

from production_test_framework.vllm import VllmClient, VllmConfig
from production_test_framework.workload.prompt_workload import PromptWorkload
from production_test_framework.workload.inferencex_workload import InferencexWorkload
from production_test_framework.workload.nccl_workload import NcclWorkload


# =============================================================================
# Constants
# =============================================================================

# Inferencex workload Docker exec timeout (seconds)
DOCKER_EXEC_TIMEOUT = 1200

# PROMPT = "Explain briefly the different LLM parallelization techniques."
PROMPT = "How many oceans are there in the world?"

# Default vLLM configuration
DEFAULT_VLLM_HOST = "localhost"
DEFAULT_VLLM_PORT = 8080
VLLM_READY_TIMEOUT = 300  # 5 minutes for model download and loading

# Default OTEL stack configuration
DEFAULT_PROMETHEUS_HOST = "localhost"
DEFAULT_PROMETHEUS_PORT = 9090

# Scrape cadence of the 'otel-collector' job in deployments/prometheus.yaml. A poll has to
# be slower than this for two identical readings to mean "no new scrape landed" rather than
# "we looked twice between scrapes".
METRICS_SCRAPE_INTERVAL = 5  # seconds

# Quiescing before a workload: poll the metric totals until they stop moving, so the
# baseline we snapshot cannot contain samples still in flight from the previous workload.
# The collector keeps republishing every series it has seen (Prometheus exporter
# metric_expiration, 5m by default), so without this the previous workload's trailing
# scrapes land inside the next workload's window -- the gap between workloads in a
# parametrized run is under a second, far less than one scrape interval.
QUIESCE_POLL_INTERVAL = METRICS_SCRAPE_INTERVAL + 1  # seconds
QUIESCE_STABLE_POLLS = 2  # consecutive unchanged readings required
QUIESCE_TIMEOUT = 90  # seconds

NCCL_PROFILER_METRICS = [
    # Collective Information metrics
    "nccl_profiler_collective_bytes_total",  # Counter (bytes)
    "nccl_profiler_collective_time_microseconds_sum",  # Histogram (us)
    "nccl_profiler_collective_count_sum",  # Histogram (count)
    "nccl_profiler_collective_num_transfers_count_sum",  # Histogram (count)
    "nccl_profiler_collective_transfer_size_bytes_sum",  # Histogram (bytes)
    "nccl_profiler_collective_transfer_time_microseconds_sum",  # Histogram (us)
    # P2P Information metrics
    "nccl_profiler_p2p_bytes_sum",  # Histogram (bytes)
    "nccl_profiler_p2p_time_microseconds_sum",  # Histogram (us)
    "nccl_profiler_p2p_num_transfers_count_sum",  # Histogram (count)
    "nccl_profiler_p2p_transfer_size_bytes_sum",  # Histogram (bytes)
    "nccl_profiler_p2p_transfer_time_microseconds_sum",  # Histogram (us)
    # Rank Information metrics
    "nccl_profiler_rank_bytes_total",  # Counter (bytes)
    "nccl_profiler_rank_latency_microseconds_sum",  # Histogram (us)
    "nccl_profiler_rank_rate_sum",  # Histogram (MB/s)
    # Transfer Information metrics
    "nccl_profiler_transfer_size_bytes_sum",  # Histogram (bytes)
    "nccl_profiler_transfer_time_microseconds_sum",  # Histogram (us)
    "nccl_profiler_transfer_latency_microseconds_sum",  # Histogram (us)
]

# Expected metrics based on workload type.
NCCL_PROFILER_METRICS_EXPECTED_PROMPT_WORKLOAD = [
    "nccl_profiler_collective_bytes_total",
    "nccl_profiler_collective_time_microseconds_sum",
    "nccl_profiler_collective_count_sum",
    "nccl_profiler_collective_transfer_size_bytes_sum",
    "nccl_profiler_collective_transfer_time_microseconds_sum",
    "nccl_profiler_rank_bytes_total",
    "nccl_profiler_transfer_size_bytes_sum",
    "nccl_profiler_transfer_time_microseconds_sum",
]

NCCL_PROFILER_METRICS_EXPECTED_INFERENCEX_WORKLOAD = [
    "nccl_profiler_collective_bytes_total",
    "nccl_profiler_collective_time_microseconds_sum",
    "nccl_profiler_collective_count_sum",
    "nccl_profiler_collective_transfer_size_bytes_sum",
    "nccl_profiler_collective_transfer_time_microseconds_sum",
    "nccl_profiler_rank_bytes_total",
    "nccl_profiler_rank_latency_microseconds_sum",
    "nccl_profiler_transfer_size_bytes_sum",
    "nccl_profiler_transfer_time_microseconds_sum",
    "nccl_profiler_transfer_latency_microseconds_sum",
]

NCCL_PROFILER_METRICS_EXPECTED_NCCL_WORKLOAD = [
    "nccl_profiler_collective_bytes_total",
    "nccl_profiler_collective_time_microseconds_sum",
    "nccl_profiler_collective_count_sum",
    "nccl_profiler_collective_num_transfers_count_sum",
    "nccl_profiler_collective_transfer_size_bytes_sum",
    "nccl_profiler_collective_transfer_time_microseconds_sum",
    "nccl_profiler_rank_bytes_total",
    "nccl_profiler_transfer_size_bytes_sum",
]


def expected_nccl_profiler_metrics(workload) -> list[str]:
    """Return the NCCL profiler metric names we assert on for this workload type."""
    if isinstance(workload, PromptWorkload):
        return NCCL_PROFILER_METRICS_EXPECTED_PROMPT_WORKLOAD
    if isinstance(workload, InferencexWorkload):
        return NCCL_PROFILER_METRICS_EXPECTED_INFERENCEX_WORKLOAD
    if isinstance(workload, NcclWorkload):
        return NCCL_PROFILER_METRICS_EXPECTED_NCCL_WORKLOAD
    raise TypeError(f"Unsupported workload type: {type(workload)!r}")


# =============================================================================
# Prometheus helpers
#
# These metrics are cumulative (counters ending _total, histogram _sum/_count), and the
# collector republishes each series every scrape whether or not any NCCL op occurred. So
# the presence of a sample proves only that the series exists -- it is true forever after
# the first NCCL op anywhere in the session, and would pass for a workload that did
# nothing. Everything below is built to assert on an *increase* instead.
# =============================================================================


def query_prometheus(prometheus_url: str, promql: str, timeout: int = 10) -> list:
    """Run an instant query and return its result series (empty on any failure)."""
    try:
        response = requests.get(
            f"{prometheus_url}/api/v1/query",
            params={"query": promql},
            timeout=timeout,
        )
    except requests.exceptions.RequestException:
        return []
    if response.status_code != 200:
        return []
    data = response.json()
    if data.get("status") != "success":
        return []
    return data.get("data", {}).get("result", [])


def metric_total(prometheus_url: str, metric_name: str) -> float | None:
    """
    Sum a metric across all of its series, or None when the series does not exist.

    Summing keeps the comparison stable when the profiler reports per-rank or per-collective
    series, and distinguishes "no series at all" (None) from "series present, value 0".
    """
    series = query_prometheus(prometheus_url, f"sum({metric_name})")
    if not series:
        return None
    try:
        return float(series[0]["value"][1])
    except (KeyError, IndexError, ValueError):
        return None


def snapshot_metric_totals(prometheus_url: str, metric_names: list[str]) -> dict[str, float | None]:
    """Current total for each metric, for use as a before/after baseline."""
    return {name: metric_total(prometheus_url, name) for name in metric_names}


def metric_increased(baseline: float | None, current: float | None) -> bool:
    """
    True when *current* represents NCCL work done since *baseline* was taken.

    Three cases, matching how these series actually behave:

    * no series before, one now -- a fresh exporter (e.g. a workload in its own
      container) starts its counters at zero, so any positive value is new work.
    * value went down -- the exporter or its process restarted and the counter reset.
      Treat it the way PromQL ``increase()`` does and count up from zero.
    * value went up -- new work, the ordinary case.
    """
    if current is None:
        return False
    if baseline is None or current < baseline:
        return current > 0
    return current > baseline


def wait_for_metrics_quiesced(
    prometheus_url: str,
    metric_names: list[str],
    *,
    timeout: float = QUIESCE_TIMEOUT,
    poll_interval: float = QUIESCE_POLL_INTERVAL,
    stable_polls: int = QUIESCE_STABLE_POLLS,
) -> tuple[dict[str, float | None], bool]:
    """
    Poll until every metric total stops changing, then return ``(totals, settled)``.

    The returned totals are the baseline for the next workload: once the numbers hold
    still across consecutive scrapes, nothing from the previous workload is still landing,
    so any later increase is attributable to the workload we are about to run.

    ``settled`` is False if the timeout hit first -- the caller gets a usable (if less
    trustworthy) baseline rather than an exception, and can say so in the log.
    """
    deadline = time.monotonic() + timeout
    previous: dict[str, float | None] | None = None
    unchanged = 0

    while True:
        current = snapshot_metric_totals(prometheus_url, metric_names)
        unchanged = unchanged + 1 if current == previous else 0
        previous = current

        if unchanged >= stable_polls:
            return current, True
        if time.monotonic() >= deadline:
            return current, False
        time.sleep(poll_interval)


# =============================================================================
# OTEL Stack Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def prometheus_url() -> str:
    """
    Provide the Prometheus URL.
    """
    host = os.getenv("PROMETHEUS_HOST", DEFAULT_PROMETHEUS_HOST)
    port = os.getenv("PROMETHEUS_PORT", str(DEFAULT_PROMETHEUS_PORT))
    return f"http://{host}:{port}"


# =============================================================================
# vLLM Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def vllm_config() -> VllmConfig:
    """
    Provide vLLM configuration.
    """
    host = os.getenv("VLLM_HOST", DEFAULT_VLLM_HOST)
    port = int(os.getenv("VLLM_PORT", str(DEFAULT_VLLM_PORT)))
    return VllmConfig(host=host, port=port)


@pytest.fixture(scope="session")
def vllm_client(vllm_config: VllmConfig) -> VllmClient:
    """
    Provide a vLLM client instance.
    """
    return VllmClient(vllm_config)


@pytest.fixture(scope="session")
def vllm_ready(vllm_client: VllmClient) -> bool:
    """
    Wait for vLLM to be ready and return status.
    """
    is_ready = vllm_client.wait_for_ready(timeout=VLLM_READY_TIMEOUT)
    if not is_ready:
        pytest.fail("vLLM server not ready within timeout")
    return True


@pytest.fixture(scope="session")
def prompt_workload():
    """
    Provide a prompt workload.
    """
    return PromptWorkload(prompt=PROMPT)


@pytest.fixture(scope="session")
def inferencex_workload():
    """
    Provide an inferencex workload.
    """
    return InferencexWorkload(docker_exec_timeout=DOCKER_EXEC_TIMEOUT)

@pytest.fixture(scope="session")
def nccl_workload():
    """
    Provide an NCCL workload.
    """
    return NcclWorkload(max_bytes="128M", gpus="device=0,1", gpus_per_host=2)


@pytest.fixture
def workload(request):
    """
    Dispatch to ``prompt_workload`` or ``inferencex_workload`` via indirect parametrization.
    """
    return request.getfixturevalue(request.param)