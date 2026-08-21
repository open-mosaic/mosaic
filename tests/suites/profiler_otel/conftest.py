# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
NCCL Profiler OTEL test fixtures.

This conftest provides fixtures and constants specific to NCCL profiler OTEL testing.
"""

import math
import os
import time
from urllib.parse import urlparse

import pytest
import requests

from production_test_framework.vllm import VllmClient, VllmConfig
from production_test_framework.docker import remove_containers_by_label
from production_test_framework.workload.prompt_workload import PromptWorkload
from production_test_framework.workload.inferencex_workload import InferencexWorkload
from production_test_framework.workload.nccl_workload import NcclWorkload

from profiler_otel import profiles


# =============================================================================
# Constants
# =============================================================================

# PROMPT = "Explain briefly the different LLM parallelization techniques."
PROMPT = "How many oceans are there in the world?"

# Sent once before any measurement. The content is irrelevant -- it exists to get the server
# and the telemetry pipeline out of their cold state, not to test anything.
WARMUP_PROMPT = "Warm up."

# Metric used to decide the telemetry pipeline is producing real values. A counter is the most
# reliable of the family, and this one is expected for every profile.
WARMUP_PROBE_METRIC = "nccl_profiler_collective_bytes_total"

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


# Expert parallelism adds all-to-all traffic that a dense tensor-parallel deployment never
# generates, so the MoE tiers should show the p2p family on top of the dense set.
# Note NIXL KV transfer does not go through NCCL: p2p on the disagg tiers comes from EP
# all-to-all and cross-node sendrecv, not from the prefill/decode KV path.
NCCL_PROFILER_METRICS_EXPECTED_AGGREGATED_MOE = list(
    NCCL_PROFILER_METRICS_EXPECTED_INFERENCEX_WORKLOAD
)
NCCL_PROFILER_METRICS_EXPECTED_DISAGG_MOE = list(
    NCCL_PROFILER_METRICS_EXPECTED_INFERENCEX_WORKLOAD
)

#: Keyed by a profile's ``expected_metrics`` field.
METRIC_SETS_BY_NAME = {
    "aggregated_dense": NCCL_PROFILER_METRICS_EXPECTED_INFERENCEX_WORKLOAD,
    "aggregated_moe": NCCL_PROFILER_METRICS_EXPECTED_AGGREGATED_MOE,
    "disagg_moe": NCCL_PROFILER_METRICS_EXPECTED_DISAGG_MOE,
}


def expected_nccl_profiler_metrics(
    workload, profile: profiles.Profile | None = None
) -> list[str]:
    """
    Return the NCCL profiler metric names we assert on for this workload.

    The InferenceX workload serves every tier, aggregated and disaggregated alike, so its
    expected set cannot be derived from the class -- it comes from the profile. The prompt and
    NCCL workloads are still distinguishable by type.
    """
    if isinstance(workload, PromptWorkload):
        return NCCL_PROFILER_METRICS_EXPECTED_PROMPT_WORKLOAD
    if isinstance(workload, NcclWorkload):
        return NCCL_PROFILER_METRICS_EXPECTED_NCCL_WORKLOAD
    if isinstance(workload, InferencexWorkload):
        if profile is None:
            raise ValueError("a profile is required to pick the InferenceX metric set")
        return METRIC_SETS_BY_NAME[profile.expected_metrics]
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
        value = float(series[0]["value"][1])
    except (KeyError, IndexError, ValueError):
        return None
    # Prometheus legitimately returns NaN for some series. float() accepts it, and every
    # comparison against NaN is False, so a NaN total would silently read as "did not
    # increase" -- indistinguishable from a profiler that produced nothing.
    return value if math.isfinite(value) else None


def metric_totals_by(
    prometheus_url: str, metric_name: str, label: str = "hostname"
) -> dict[str, float]:
    """
    Total for *metric_name* broken down by *label*, e.g. ``{"gpu-node-03": 1234.0}``.

    :func:`metric_total` sums across every series, so one reporting rank makes the total rise
    and the assertion passes while every other rank is silent. On a 96-GPU cluster that is the
    difference between a real check and no check at all. The profiler labels every series with
    ``hostname``, ``rank``, ``local_rank`` and ``communicator``, so the breakdown is free.

    Returns an empty dict when the series does not exist.
    """
    series = query_prometheus(prometheus_url, f"sum by ({label}) ({metric_name})")
    totals: dict[str, float] = {}
    for item in series:
        key = item.get("metric", {}).get(label)
        if key is None:
            continue
        try:
            value = float(item["value"][1])
        except (KeyError, IndexError, ValueError):
            continue
        # See metric_total: a NaN would read as a flat host rather than an absent one.
        if math.isfinite(value):
            totals[key] = value
    return totals


def reporting_ranks(prometheus_url: str, metric_name: str) -> set[tuple[str, str]]:
    """
    The ``(hostname, rank)`` pairs currently exporting *metric_name*.

    Used to check that every rank we expect is present, and to name the ones that are not --
    "3 ranks did not report" is not actionable without knowing which node they were on.
    """
    series = query_prometheus(prometheus_url, f"count by (hostname, rank) ({metric_name})")
    pairs: set[tuple[str, str]] = set()
    for item in series:
        labels = item.get("metric", {})
        host, rank = labels.get("hostname"), labels.get("rank")
        if host is not None and rank is not None:
            pairs.add((host, rank))
    return pairs


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


@pytest.fixture(scope="session", autouse=True)
def sweep_orphaned_workload_containers():
    """
    Remove any workload container still running when the session ends.
    """
    yield

    removed = remove_containers_by_label()
    if removed:
        print(f"\n  Removed {len(removed)} orphaned workload container(s): {removed}")


@pytest.fixture(scope="session")
def workload_profile(request) -> profiles.Profile:
    """
    The hardware profile under test, from ``--workload-profile``.

    Session-scoped: one profile describes the deployment the whole run targets.
    """
    name = request.config.getoption("--workload-profile")
    extra_dirs = request.config.getoption("profile_dirs")
    try:
        profile = profiles.load(name, profiles.profile_dirs(extra_dirs))
    except profiles.ProfileError as exc:
        # Fail outside the except block so pytest reports the message alone rather than
        # chaining it onto a traceback nobody needs.
        error = str(exc)
        profile = None
    else:
        error = None
    if error is not None:
        pytest.fail(error, pytrace=False)
    print(f"\n  Profile: {profile.name} ({profile.path})")
    print(f"    {profile.description}")
    print(
        f"    expecting {profile.coverage.hosts} host(s), {profile.coverage.ranks} rank(s), "
        f"{profile.coverage.communicators} communicator(s)"
    )
    return profile


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


def _endpoint_host_port(profile: profiles.Profile) -> tuple[str, int] | None:
    """
    ``(host, port)`` for a profile reached that way, or None for a base_url profile.

    A disaggregated cluster sits behind a single frontend URL, which host+port cannot express.
    The model guarantees exactly one of the two forms is set.
    """
    if profile.endpoint.base_url is not None:
        return None
    return profile.endpoint.host, profile.endpoint.port


@pytest.fixture(scope="session")
def vllm_config(workload_profile: profiles.Profile) -> VllmConfig:
    """
    Provide vLLM configuration, defaulting to the profile's endpoint.

    VLLM_HOST/VLLM_PORT keep the override behaviour documented in docs/testsuite.md; only the
    fallback moves from a module constant to the profile. Without that, a disaggregated profile
    would point the InferenceX workload at the frontend while this fixture -- and so
    ``vllm_ready`` and ``prompt_workload`` -- still probed localhost:8080.
    """
    host_port = _endpoint_host_port(workload_profile)
    if host_port is None:
        base_url = workload_profile.endpoint.base_url
        parsed = urlparse(base_url)
        if not parsed.hostname:
            pytest.fail(f"profile endpoint base_url is not a usable URL: {base_url!r}")
        # VllmConfig is host/port only and hardcodes http://. Fine for a plain frontend; a
        # TLS-terminated or path-prefixed one needs a base_url field on VllmConfig upstream.
        if parsed.scheme not in ("", "http"):
            pytest.fail(
                f"profile endpoint base_url uses scheme {parsed.scheme!r}; VllmConfig only "
                "builds http:// URLs from host/port. Add base_url support to VllmConfig."
            )
        default_host, default_port = parsed.hostname, parsed.port or 80
    else:
        default_host, default_port = host_port

    host = os.getenv("VLLM_HOST", default_host)
    port = int(os.getenv("VLLM_PORT", str(default_port)))
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
def profiler_pipeline_warm(
    workload_profile, vllm_ready, vllm_client: VllmClient, prometheus_url: str
):
    """
    Drive one throwaway inference and wait for its telemetry to arrive, before any measurement.

    Without this the first workload of a session measures a cold server *and* an empty metrics
    pipeline, neither of which is what the test is about. Both were seen to fail a run on a
    4-GPU box:

    * the first inference after start-up took 65s against a few seconds in steady state, long
      enough to eat into the window allowed for metrics to arrive;
    * with no series in Prometheus yet, ``wait_for_metrics_quiesced`` settles immediately on
      "nothing exists" and hands back a baseline of absent for every metric. The assertion then
      reduces to "the first sample ever published must already exceed zero", which a profiler
      that registers its instruments before recording anything fails by publishing a flat zero.

    Warming up first makes the baseline a real number and every measurement a steady-state one.
    Re-running the same workload against an already-warm stack passed, which is what identified
    this.
    """
    print("\n  Warming up the server and the telemetry pipeline before measuring...")
    started = time.monotonic()
    result = vllm_client.complete(WARMUP_PROMPT)
    if not result.success:
        pytest.fail(
            f"warm-up inference failed, so nothing measured after it can be trusted: {result.error}"
        )
    print(f"    Warm-up inference took {time.monotonic() - started:.1f}s")

    # Waiting for the inference to return is not enough: export is chunky, and a whole
    # workload's worth of bytes can land in a single scrape well after the request finished.
    # Wait for a value to actually appear, so the first baseline is taken against a live series.
    deadline = time.monotonic() + workload_profile.timeouts.metrics_available
    while True:
        total = metric_total(prometheus_url, WARMUP_PROBE_METRIC)
        if total:
            print(f"    Telemetry pipeline live: {WARMUP_PROBE_METRIC}={total:.6g}")
            return
        if time.monotonic() >= deadline:
            # Not fatal. The tests assert on an increase and will fail on their own terms with
            # better diagnostics than this fixture could produce.
            print(
                f"    WARNING: {WARMUP_PROBE_METRIC} still absent or zero "
                f"{workload_profile.timeouts.metrics_available}s after the warm-up inference"
            )
            return
        time.sleep(QUIESCE_POLL_INTERVAL)


@pytest.fixture(scope="session")
def prompt_workload(vllm_config: VllmConfig, profiler_pipeline_warm):
    """
    Provide a prompt workload aimed at the profile's endpoint.
    """
    return PromptWorkload(prompt=PROMPT, host=vllm_config.host, port=vllm_config.port)


@pytest.fixture(scope="session")
def inferencex_workload(workload_profile: profiles.Profile, profiler_pipeline_warm):
    """
    Provide an InferenceX workload configured by the profile.

    ``benchmark_options`` is handed to the workload verbatim: it converts keys to
    benchmark_serving.py flags generically, so a profile can set any flag that script accepts
    without a change here. The endpoint is layered underneath, where ``None`` means "omit the
    flag" -- which is how a base_url profile drops the default host/port.
    """
    options = dict(workload_profile.benchmark_options)
    host_port = _endpoint_host_port(workload_profile)
    if host_port is None:
        options.setdefault("base_url", workload_profile.endpoint.base_url)
        options.setdefault("host", None)
        options.setdefault("port", None)
    else:
        options.setdefault("host", host_port[0])
        options.setdefault("port", host_port[1])
    options.setdefault("model", workload_profile.serving.model)

    # container_name is left to the workload, which generates one unique to this process and
    # removes it when the run ends however it ends.
    return InferencexWorkload(
        benchmark_options=options,
        docker_exec_timeout=workload_profile.timeouts.workload,
    )


@pytest.fixture(scope="session")
def nccl_workload(workload_profile: profiles.Profile, profiler_pipeline_warm):
    """
    Provide an NCCL workload sized to the profile's machine.
    """
    gpus_per_machine = workload_profile.hardware.gpus_per_machine
    device_list = ",".join(str(i) for i in range(gpus_per_machine))
    return NcclWorkload(
        max_bytes="128M",
        gpus=f"device={device_list}",
        gpus_per_host=gpus_per_machine,
    )


@pytest.fixture
def workload(request):
    """
    Dispatch to ``prompt_workload`` or ``inferencex_workload`` via indirect parametrization.
    """
    return request.getfixturevalue(request.param)
