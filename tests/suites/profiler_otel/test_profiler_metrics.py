# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
Tests for vLLM inference with NCCL Profiler OTEL.

These tests validate that the NCCL profiler is exporting telemetry correctly.
"""

import time

import pytest
import requests
from production_test_framework.vllm import InferenceResult
from production_test_framework.workload.inferencex_workload import InferencexBenchmarkResult
from production_test_framework.workload.workload import WorkloadStatus

from profiler_otel.conftest import (
    NCCL_PROFILER_METRICS_EXPECTED_PROMPT_WORKLOAD,
    expected_nccl_profiler_metrics,
    metric_increased,
    metric_total,
    metric_totals_by,
    metric_totals_by_gpu,
    wait_for_metrics_quiesced,
)
from profiler_otel.environment import benchmark_option_rows
from profiler_otel.reporting import (
    CoverageStatus,
    MetricStatus,
    format_delta,
    format_duration,
    format_number,
    format_value,
)

# =============================================================================
# NCCL Profiler Telemetry Tests
# =============================================================================
#
# Timeouts come from the active profile (``timeouts:`` in its YAML), because they differ by an
# order of magnitude between a 2-GPU CI box and a 96-GPU cluster: how long a workload runs, how
# long its metrics take to reach Prometheus, and how long the totals take to settle.

METRICS_POLL_INTERVAL = 2  # seconds


def _settle_and_snapshot(reporter, prometheus_url, metric_names, quiesce_timeout):
    """
    Wait for the metric totals to stop moving, then return them as a baseline.
    """
    reporter.note("Waiting for NCCL metric totals to settle before taking a baseline...")
    baseline, settled = wait_for_metrics_quiesced(prometheus_url, metric_names, timeout=quiesce_timeout)
    if not settled:
        reporter.note(
            f"WARNING: totals still moving after {quiesce_timeout}s; baseline may "
            "include samples from a previous workload"
        )
    return baseline


def _run_workload(reporter, workload, timeout):
    """Run *workload* to completion and return its result, failing on timeout or error."""
    workload.start()

    is_done = workload.wait_for_completion(timeout=timeout)
    assert is_done is True, f"Workload did not complete within timeout of {timeout} seconds"

    workload_result = workload.get_result()

    name = getattr(workload, "workload_name", type(workload).__name__)
    match workload_result.result:
        case str() as text:
            reporter.note(f"Workload result -- {name}: {text}")
        case InferenceResult() as inference:
            reporter.table(
                ["measure", "value"],
                [
                    ["characters generated", str(len(inference.text))],
                    ["usage", str(inference.usage)],
                    ["runtime", format_duration(workload_result.runtime)],
                    ["text", inference.text],
                ],
                title=f"Workload result -- {name}",
                left={1},
            )
        case InferencexBenchmarkResult() as bench:
            latency = bench.latency_ms

            reporter.table(
                ["option", "value", "benchmark_serving.py flag"],
                benchmark_option_rows(workload.benchmark_options),
                title=f"Workload configuration -- {name}",
                left={2},
            )

            reporter.table(
                ["measure", "value", "unit"],
                [
                    ["requests completed", format_number(bench.successful_requests, ",d"), ""],
                    ["benchmark duration (timed requests)", format_duration(bench.duration_seconds), ""],
                    ["tokens in (prompt)", format_number(bench.total_input_tokens, ",d"), "tokens"],
                    ["tokens out (generated)", format_number(bench.total_generated_tokens, ",d"), "tokens"],
                    ["throughput (requests)", format_number(bench.request_throughput), "req/s"],
                    [
                        "throughput (prompt+generated)",
                        format_number(bench.total_token_throughput),
                        "tok/s",
                    ],
                    [
                        "throughput (generated only)",
                        format_number(bench.output_token_throughput),
                        "tok/s",
                    ],
                    ["TTFT mean", format_number(latency.get("mean_ttft")), "ms"],
                    ["TTFT p99", format_number(latency.get("p99_ttft")), "ms"],
                    ["TPOT mean", format_number(latency.get("mean_tpot")), "ms"],
                    ["TPOT p99", format_number(latency.get("p99_tpot")), "ms"],
                    ["container wall time (incl. startup/teardown)", format_duration(workload_result.runtime), ""],
                ],
                title=f"Workload result -- {name}",
                left={2},
            )
        case None:
            reporter.note(f"Workload result -- {name}: no result (stopped, or no parseable output)")

    assert workload_result.status == WorkloadStatus.COMPLETED, (
        f"Inference must succeed before checking metrics; status was "
        f"{workload_result.status}. Result: {workload_result.result!r}"
    )
    return workload_result


def _poll_until_increased(prometheus_url, metric_names, baseline, result, timeout):
    """
    Poll each metric until its total rises above *baseline*, returning those that did.

    Metrics take time to reach Prometheus after a workload finishes, so rather than sleeping a
    fixed amount this records each metric as soon as it moves and returns once they all have or
    the timeout elapses.
    """
    increased: dict[str, tuple[float, float]] = {}
    deadline = time.monotonic() + timeout
    while True:
        for metric_name in metric_names:
            if metric_name in increased:
                continue
            current = metric_total(prometheus_url, metric_name)
            if metric_increased(baseline[metric_name], current):
                increased[metric_name] = (current, time.time() - result.end_time)

        if len(increased) == len(metric_names) or time.monotonic() >= deadline:
            return increased
        time.sleep(METRICS_POLL_INTERVAL)


@pytest.mark.profiler_otel
class TestNCCLProfilerTelemetry:
    """Tests for NCCL profiler telemetry export."""

    def test_otel_collector_accessible(self, prometheus_url: str):
        """
        :title: Connectivity - Prometheus endpoint accessible
        :suite: profiler_otel
        :description: Verify OTEL collector (Prometheus endpoint) is accessible.
        """
        url = f"{prometheus_url}/api/v1/status/buildinfo"
        retries = 3
        poll_interval = 2

        for i in range(retries):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    print(f"\n  Prometheus endpoint accessible at {prometheus_url}")
                    return
            except requests.exceptions.RequestException:
                pass

            if i < retries - 1:
                time.sleep(poll_interval)

        pytest.skip(f"Prometheus endpoint not accessible at {prometheus_url}")

    def test_grafana_accessible(self, grafana_url: str):
        """
        :title: Connectivity - Grafana dashboard accessible
        :suite: profiler_otel
        :description: Verify Grafana dashboard is accessible via health endpoint.
        """
        url = f"{grafana_url}/api/health"
        retries = 5
        poll_interval = 5

        for i in range(retries):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    print(f"\n  Grafana accessible at {grafana_url}")
                    return
            except requests.exceptions.RequestException:
                pass

            if i < retries - 1:
                time.sleep(poll_interval)

        pytest.skip(f"Grafana not accessible at {grafana_url}")

    @pytest.mark.parametrize(
        "workload",
        [
            "prompt_workload",
            "inferencex_workload",
            "nccl_workload",
        ],
        indirect=True,
        ids=["prompt_workload", "inferencex_workload", "nccl_workload"],
    )
    def test_nccl_metrics_exported_after_inference(
        self,
        workload,
        workload_profile,
        prometheus_url: str,
        reporter,
    ):
        """
        :title: Telemetry - NCCL metrics exported after inference
        :suite: profiler_otel
        :description: Verify NCCL profiler metrics are exported to Prometheus.
            Waits for the metric totals to settle, snapshots them.
            Runs the workload, then requires each expected metric for the active profile
            to increase above that baseline.
        """
        timeouts = workload_profile.timeouts
        nccl_profiler_metrics = expected_nccl_profiler_metrics(workload, workload_profile)

        baseline = _settle_and_snapshot(reporter, prometheus_url, nccl_profiler_metrics, timeouts.quiesce)
        reporter.table(
            ["metric", "total"],
            [[name, format_value(value)] for name, value in baseline.items()],
            title=f"Baseline ({len(baseline)} metrics)",
        )

        workload_result = _run_workload(reporter, workload, timeouts.workload)

        reporter.note(
            f"Workload window: start={workload_result.start_time:.3f} end={workload_result.end_time:.3f}"
        )
        increased_metrics = _poll_until_increased(
            prometheus_url,
            nccl_profiler_metrics,
            baseline,
            workload_result,
            timeouts.metrics_available,
        )

        # One row per expected metric, in the profile's order, so the table is the whole story:
        # what each total was, what it became, and for anything that did not move, which of the
        rows = []
        statuses: dict[str, MetricStatus] = {}
        for metric_name in nccl_profiler_metrics:
            before = baseline[metric_name]
            rose = metric_name in increased_metrics
            if rose:
                current, seen_after = increased_metrics[metric_name]
                seen = f"+{seen_after:.1f}s"
            else:
                current = metric_total(prometheus_url, metric_name)
                seen = "-"
            status = MetricStatus.for_metric(rose, current)
            statuses[metric_name] = status
            rows.append(
                [
                    metric_name,
                    format_value(before),
                    format_value(current),
                    format_delta(before, current),
                    seen,
                    status,
                ]
            )

        missing_metrics = [name for name, status in statuses.items() if status.is_failure]
        reporter.table(
            ["metric", "baseline", "current", "delta", "seen", "status"],
            rows,
            title=(
                f"NCCL profiler metrics -- {len(increased_metrics)}/{len(nccl_profiler_metrics)} "
                f"increased within {timeouts.metrics_available}s of the workload ending"
            ),
            left={5},
            status_column=5,
        )

        if missing_metrics:
            reporter.table(
                ["metric", "status", "what to check"],
                [[name, statuses[name], statuses[name].remedy] for name in missing_metrics],
                title=f"Did not increase ({len(missing_metrics)} of {len(nccl_profiler_metrics)})",
                left={1, 2},
                status_column=1,
            )

        assert not missing_metrics, (
            f"Expected {len(nccl_profiler_metrics)} metrics to increase, "
            f"{len(increased_metrics)} did within {timeouts.metrics_available}s. "
            "Did not increase: "
            + ", ".join(f"{name} ({statuses[name]})" for name in missing_metrics)
        )

    def test_every_node_reports_nccl_metrics(
        self,
        inferencex_workload,
        workload_profile,
        prometheus_url: str,
        reporter,
    ):
        """
        :title: Telemetry - every node, GPU and communicator does work
        :suite: profiler_otel
        :description: Verify the profiler reports NCCL activity from every expected host, GPU
            and communicator, not merely from somewhere. The other tests sum a metric across
            all of its series, so one healthy GPU makes the total rise and they pass while
            every other one is silent. Expected counts come from the profile's coverage.
        """
        coverage = workload_profile.coverage
        timeouts = workload_profile.timeouts
        metrics = expected_nccl_profiler_metrics(inferencex_workload, workload_profile)

        # One metric is enough to establish who did work, and keeps the query cheap at high
        # GPU counts. A counter is the most reliable of the family.
        probe = "nccl_profiler_collective_bytes_total"
        if probe not in metrics:
            probe = metrics[0]

        # Counted by *increase*, not by presence. Every containerised workload exports under
        # these same metric names, and the collector keeps republishing a series long after its
        # container is gone, so simply being present says nothing about this workload.
        baseline_by_gpu = metric_totals_by_gpu(prometheus_url, probe)
        baseline_by_comm = metric_totals_by(prometheus_url, probe, label="communicator")
        _settle_and_snapshot(reporter, prometheus_url, [probe], timeouts.quiesce)

        workload_result = _run_workload(reporter, inferencex_workload, timeouts.workload)

        def active_participants():
            """GPUs, hosts and communicators whose totals rose during this workload."""
            by_gpu = metric_totals_by_gpu(prometheus_url, probe)
            gpus = {key for key, total in by_gpu.items() if metric_increased(baseline_by_gpu.get(key), total)}
            by_comm = metric_totals_by(prometheus_url, probe, label="communicator")
            comms = {name for name, total in by_comm.items() if metric_increased(baseline_by_comm.get(name), total)}
            return gpus, {host for host, _ in gpus}, comms

        # Give the last GPUs' samples time to land before judging who is missing; export is
        # chunky, and a whole workload's traffic can appear in a single scrape.
        deadline = time.monotonic() + timeouts.metrics_available
        while True:
            gpus, hosts, comms = active_participants()
            enough = (
                len(hosts) >= coverage.hosts and len(gpus) >= coverage.gpus and len(comms) >= coverage.communicators
            )
            if enough or time.monotonic() >= deadline:
                break
            time.sleep(METRICS_POLL_INTERVAL)

        gpus_by_host: dict[str, list[str]] = {}
        for host, gpu in sorted(gpus):
            gpus_by_host.setdefault(host, []).append(gpu)

        reporter.table(
            ["participant", "seen", "expected", "status"],
            [
                [label, str(len(seen)), str(expected), CoverageStatus.for_counts(len(seen), expected)]
                for label, seen, expected in (
                    ("hosts", hosts, coverage.hosts),
                    ("GPUs", gpus, coverage.gpus),
                    ("communicators", comms, coverage.communicators),
                )
            ],
            title=(
                f"Did work within {time.time() - workload_result.end_time:.1f}s of the "
                f"workload ending ({probe})"
            ),
            left={3},
            status_column=3,
        )

        reporter.table(
            ["host", "GPUs", "PCI addresses"],
            [[host, str(len(host_gpus)), ", ".join(host_gpus)] for host, host_gpus in sorted(gpus_by_host.items())]
            or [["(none)", "0", ""]],
            title="GPUs that did work, per host",
            left={2},
        )
        reporter.note(f"communicators: {sorted(comms)}")

        problems: list[str] = []
        if len(hosts) != coverage.hosts:
            problems.append(f"expected {coverage.hosts} host(s) doing work, saw {len(hosts)}: {sorted(hosts)}")
        if len(gpus) != coverage.gpus:
            problems.append(f"expected {coverage.gpus} GPU(s) doing work, saw {len(gpus)}, per host: {gpus_by_host}")
        # Prefill and decode pools form separate communicators; a global total rises even when
        # only one of them is alive.
        if len(comms) < coverage.communicators:
            problems.append(
                f"expected at least {coverage.communicators} NCCL communicator(s) doing work, "
                f"saw {len(comms)}: {sorted(comms)}"
            )

        if problems:
            # A GPU present but flat is a different fault from one missing entirely, and
            # they need different fixes, so name which one this is.
            idle = sorted(set(metric_totals_by_gpu(prometheus_url, probe)) - gpus)
            if idle:
                reporter.table(
                    ["host", "GPU", "status"],
                    [[host, gpu, MetricStatus.FLAT] for host, gpu in idle],
                    title="Reporting a series but flat across this workload",
                    left={2},
                    status_column=2,
                )

        assert not problems, (
            f"NCCL profiler coverage does not match profile '{workload_profile.name}' "
            f"({probe}):\n  - " + "\n  - ".join(problems)
        )

    def test_metrics_do_not_increase_without_a_workload(self, workload_profile, prometheus_url: str, reporter):
        """
        :title: Telemetry - NCCL metrics do not increase while idle
        :suite: profiler_otel
        :description: Falsifiability guard for the metric assertions. The collector
            republishes every series it has seen on each scrape, so any check based on a
            sample being present passes even when nothing ran. This test runs no workload
            and requires the totals to stay flat -- if it fails, the assertions in this
            file have stopped distinguishing real NCCL activity from a live exporter.
            Checked per host as well as globally, since a global sum can hide one leaky node.
        """
        metrics = NCCL_PROFILER_METRICS_EXPECTED_PROMPT_WORKLOAD
        timeouts = workload_profile.timeouts
        quiesce_timeout = timeouts.quiesce
        # Watch for as long as a real workload gets for its metrics to land, so a leak has
        # the same opportunity to show up here as there.
        observation_period = timeouts.metrics_available

        baseline, settled = wait_for_metrics_quiesced(prometheus_url, metrics, timeout=quiesce_timeout)
        if not settled:
            pytest.fail(f"NCCL metric totals still moving after {quiesce_timeout}s; cannot establish an idle baseline")
        baseline_by_host = {m: metric_totals_by(prometheus_url, m) for m in metrics}

        time.sleep(observation_period)

        increased: dict[str, tuple[float | None, float | None]] = {}
        leaky_hosts: dict[str, list[str]] = {}
        currents: dict[str, float | None] = {}
        for name in metrics:
            current = metric_total(prometheus_url, name)
            currents[name] = current
            if metric_increased(baseline[name], current):
                increased[name] = (baseline[name], current)
            for host, total in metric_totals_by(prometheus_url, name).items():
                if metric_increased(baseline_by_host[name].get(host), total):
                    leaky_hosts.setdefault(host, []).append(name)

        reporter.table(
            ["metric", "baseline", "current", "delta", "status"],
            [
                [
                    name,
                    format_value(baseline[name]),
                    format_value(currents[name]),
                    format_delta(baseline[name], currents[name]),
                    # A metric that moved while idle is the failure here
                    MetricStatus.ROSE if name in increased else MetricStatus.FLAT,
                ]
                for name in metrics
            ],
            title=(
                f"Idled {observation_period}s after the totals settled -- "
                f"{len(increased)}/{len(metrics)} metrics moved (none should)"
            ),
            left={4},
            status_column=4,
        )
        if leaky_hosts:
            reporter.table(
                ["host", "metrics that moved"],
                [[host, ", ".join(names)] for host, names in sorted(leaky_hosts.items())],
                title="Per-host movement while idle",
                left={1},
            )

        assert not increased and not leaky_hosts, (
            "NCCL metric totals rose with no workload running: "
            + ", ".join(
                f"{name} {format_value(before)} -> {format_value(after)}"
                for name, (before, after) in increased.items()
            )
            + (f"; per-host movement: {leaky_hosts}" if leaky_hosts else "")
            + ". Either something else is driving NCCL ops, or the quiesce period is too "
            "short and the assertions in this file cannot tell activity from a live exporter."
        )
