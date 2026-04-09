# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
NCCL Profiler OTEL test fixtures.

This conftest provides fixtures and constants specific to NCCL profiler OTEL testing.
"""

import os
import time

import pytest

from framework.vllm import VllmClient, VllmConfig, InferenceResult
from framework.workload.prompt_workload import PromptWorkload
from framework.workload.inferencex_workload import InferencexWorkload


# =============================================================================
# Constants
# =============================================================================

# PROMPT = "Explain briefly the different LLM parallelization techniques."
PROMPT = "How many oceans are there in the world?"

# Default vLLM configuration
DEFAULT_VLLM_HOST = "localhost"
DEFAULT_VLLM_PORT = 8080
VLLM_READY_TIMEOUT = 300  # 5 minutes for model download and loading

# Default OTEL stack configuration
DEFAULT_PROMETHEUS_HOST = "localhost"
DEFAULT_PROMETHEUS_PORT = 9090

# All NCCL profiler metrics defined in telemetry.cc initializeOtelMetrics()
# Prometheus-transformed metric names from telemetry.cc initializeOtelMetrics()
# Counters get _total suffix, Histograms get _bucket/_count/_sum suffixes,
# and units are expanded (us -> microseconds, bytes -> bytes).
# Using _total for counters and _sum for histograms as primary validation.
# NCCL_PROFILER_METRICS = [
#     # Collective Information metrics
#     "nccl_profiler_collective_bytes_total",  # Counter (bytes)
#     "nccl_profiler_collective_time_microseconds_sum",  # Histogram (us)
#     "nccl_profiler_collective_count_sum",  # Histogram (count)
#     "nccl_profiler_collective_num_transfers_sum",  # Histogram (count)
#     "nccl_profiler_collective_transfer_size_bytes_sum",  # Histogram (bytes)
#     "nccl_profiler_collective_transfer_time_microseconds_sum",  # Histogram (us)
#     # P2P Information metrics
#     "nccl_profiler_p2p_bytes_bytes_sum",  # Histogram (bytes)
#     "nccl_profiler_p2p_time_microseconds_sum",  # Histogram (us)
#     "nccl_profiler_p2p_num_transfers_sum",  # Histogram (count)
#     "nccl_profiler_p2p_transfer_size_bytes_sum",  # Histogram (bytes)
#     "nccl_profiler_p2p_transfer_time_microseconds_sum",  # Histogram (us)
#     # Rank Information metrics
#     "nccl_profiler_rank_bytes_total",  # Counter (bytes)
#     "nccl_profiler_rank_latency_microseconds_sum",  # Histogram (us)
#     "nccl_profiler_rank_rate_sum",  # Histogram (MB/s)
#     # Transfer Information metrics
#     "nccl_profiler_transfer_size_bytes_sum",  # Histogram (bytes)
#     "nccl_profiler_transfer_time_microseconds_sum",  # Histogram (us)
#     "nccl_profiler_transfer_latency_microseconds_sum",  # Histogram (us)
# ]

NCCL_PROFILER_METRICS = [
    # Collective Information metrics
    "nccl_profiler_collective_bytes_total",  # Counter (bytes)
    "nccl_profiler_collective_time_microseconds_sum",  # Histogram (us)
    "nccl_profiler_collective_count_sum",  # Histogram (count)
]


# =============================================================================
# OTEL Stack Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def nccl_profiler_metrics() -> list[str]:
    """
    Provide the list of NCCL profiler metric names.

    These are all metrics defined in telemetry.cc initializeOtelMetrics().
    """
    return NCCL_PROFILER_METRICS


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
    return InferencexWorkload()


@pytest.fixture
def workload(request):
    """
    Dispatch to ``prompt_workload`` or ``inferencex_workload`` via indirect parametrization.
    """
    return request.getfixturevalue(request.param)