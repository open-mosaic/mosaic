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

# =============================================================================
# NCCL Profiler Telemetry Tests
# =============================================================================
#
# Timeouts come from the active profile (``timeouts:`` in its YAML), because they differ by an
# order of magnitude between a 2-GPU CI box and a 96-GPU cluster: how long a workload runs, how
# long its metrics take to reach Prometheus, and how long the totals take to settle.

METRICS_POLL_INTERVAL = 2  # seconds


def _format_value(value: float | None) -> str:
    """Render a metric total for logs, distinguishing an absent series from a zero one."""
    return "absent" if value is None else f"{value:.6g}"


def _format_totals(totals: dict[str, float | None]) -> str:
    return ", ".join(f"{name}={_format_value(value)}" for name, value in totals.items())


def _settle_and_snapshot(prometheus_url, metric_names, quiesce_timeout):
    """
    Wait for the metric totals to stop moving, then return them as a baseline.
    """
    print("\n  Waiting for NCCL metric totals to settle before taking a baseline...")
    baseline, settled = wait_for_metrics_quiesced(
        prometheus_url, metric_names, timeout=quiesce_timeout
    )
    if not settled:
        print(
            f"  WARNING: totals still moving after {quiesce_timeout}s; baseline may "
            "include samples from a previous workload"
        )
    return baseline


def _run_workload(workload, timeout):
    """Run *workload* to completion and return its result, failing on timeout or error."""
    workload.start()

    is_done = workload.wait_for_completion(timeout=timeout)
    assert is_done is True, f"Workload did not complete within timeout of {timeout} seconds"

    workload_result = workload.get_result()

    print("Workload result:")
    match workload_result.result:
        case str():
            print(workload_result.result)
        case InferenceResult():
            print(f"  Generated {len(workload_result.result.text)} characters")
            print(f"  Usage: {workload_result.result.usage}")
            print(f"  Text: {workload_result.result.text}")
        case InferencexBenchmarkResult() as bench:
            print(f"  Completed {bench.successful_requests} requests in {bench.duration_seconds}s")
            print(f"  Throughput: {bench.total_token_throughput} tok/s total, "
                  f"{bench.output_token_throughput} tok/s output")
            latency = bench.latency_ms
            print(f"  TTFT mean/p99 (ms): {latency.get('mean_ttft')} / {latency.get('p99_ttft')}")
            print(f"  TPOT mean/p99 (ms): {latency.get('mean_tpot')} / {latency.get('p99_tpot')}")
        case None:
            print("  (no result -- workload was stopped or produced no parseable output)")

    print(f"  Workload runtime: {workload_result.runtime:.1f}s")

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
    increased: dict[str, float] = {}
    deadline = time.monotonic() + timeout
    while True:
        for metric_name in metric_names:
            if metric_name in increased:
                continue
            current = metric_total(prometheus_url, metric_name)
            if metric_increased(baseline[metric_name], current):
                increased[metric_name] = current
                print(
                    f"    Increased: {metric_name} "
                    f"{_format_value(baseline[metric_name])} -> {_format_value(current)} "
                    f"(+{time.time() - result.end_time:.1f}s after workload end)"
                )

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

        baseline = _settle_and_snapshot(
            prometheus_url, nccl_profiler_metrics, timeouts.quiesce
        )
        print(f"  Baseline ({len(baseline)} metrics): {_format_totals(baseline)}")

        workload_result = _run_workload(workload, timeouts.workload)

        print(
            f"  Workload window: start={workload_result.start_time:.3f} "
            f"end={workload_result.end_time:.3f}"
        )
        increased_metrics = _poll_until_increased(
            prometheus_url,
            nccl_profiler_metrics,
            baseline,
            workload_result,
            timeouts.metrics_available,
        )

        missing_metrics = [m for m in nccl_profiler_metrics if m not in increased_metrics]

        print(
            f"\n  {len(increased_metrics)}/{len(nccl_profiler_metrics)} NCCL profiler "
            "metrics increased during the workload"
        )
        if missing_metrics:
            print(f"  Metrics that did not increase: {missing_metrics}")
            # Separate the failure modes. Because the baseline was taken after the
            # totals settled, a flat total means the profiler genuinely produced nothing
            # for this workload. It is no longer possible to confuse that with "the series
            # is missing" or "ingest is slow".
            #   no series at all -> nothing was ever exported; check the profiler plugin,
            #                       the OTLP endpoint and the collector
            #   total unchanged  -> the exporter is alive and being scraped, but this
            #                       workload drove no NCCL ops through it
            print("  Diagnostics:")
            for metric_name in missing_metrics:
                current = metric_total(prometheus_url, metric_name)
                if current is None:
                    print(f"    {metric_name}: no series in Prometheus")
                    continue
                print(
                    f"    {metric_name}: baseline={_format_value(baseline[metric_name])} "
                    f"current={_format_value(current)} (unchanged after "
                    f"{timeouts.metrics_available}s)"
                )

        assert not missing_metrics, (
            f"Expected {len(nccl_profiler_metrics)} metrics to increase, "
            f"{len(increased_metrics)} did within {timeouts.metrics_available}s. "
            f"Did not increase: {missing_metrics}"
        )

    def test_every_node_reports_nccl_metrics(
        self,
        inferencex_workload,
        workload_profile,
        prometheus_url: str,
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
        _settle_and_snapshot(prometheus_url, [probe], timeouts.quiesce)

        workload_result = _run_workload(inferencex_workload, timeouts.workload)

        def active_participants():
            """GPUs, hosts and communicators whose totals rose during this workload."""
            by_gpu = metric_totals_by_gpu(prometheus_url, probe)
            gpus = {
                key
                for key, total in by_gpu.items()
                if metric_increased(baseline_by_gpu.get(key), total)
            }
            by_comm = metric_totals_by(prometheus_url, probe, label="communicator")
            comms = {
                name
                for name, total in by_comm.items()
                if metric_increased(baseline_by_comm.get(name), total)
            }
            return gpus, {host for host, _ in gpus}, comms

        # Give the last GPUs' samples time to land before judging who is missing; export is
        # chunky, and a whole workload's traffic can appear in a single scrape.
        deadline = time.monotonic() + timeouts.metrics_available
        while True:
            gpus, hosts, comms = active_participants()
            enough = (
                len(hosts) >= coverage.hosts
                and len(gpus) >= coverage.ranks
                and len(comms) >= coverage.communicators
            )
            if enough or time.monotonic() >= deadline:
                break
            time.sleep(METRICS_POLL_INTERVAL)

        gpus_by_host: dict[str, list[str]] = {}
        for host, gpu in sorted(gpus):
            gpus_by_host.setdefault(host, []).append(gpu)

        print(
            f"\n  Did work within {time.time() - workload_result.end_time:.1f}s of the "
            f"workload ending: {len(hosts)}/{coverage.hosts} hosts, "
            f"{len(gpus)}/{coverage.ranks} GPUs, "
            f"{len(comms)}/{coverage.communicators} communicators"
        )
        print(f"    GPUs per host : {gpus_by_host}")
        print(f"    communicators : {sorted(comms)}")

        problems: list[str] = []
        if len(hosts) != coverage.hosts:
            problems.append(
                f"expected {coverage.hosts} host(s) doing work, saw {len(hosts)}: {sorted(hosts)}"
            )
        if len(gpus) != coverage.ranks:
            problems.append(
                f"expected {coverage.ranks} GPU(s) doing work, saw {len(gpus)}, "
                f"per host: {gpus_by_host}"
            )
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
                print(f"  Reporting a series but flat across this workload: {idle}")

        assert not problems, (
            f"NCCL profiler coverage does not match profile '{workload_profile.name}' "
            f"({probe}):\n  - " + "\n  - ".join(problems)
        )

    def test_metrics_do_not_increase_without_a_workload(
        self, workload_profile, prometheus_url: str
    ):
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

        baseline, settled = wait_for_metrics_quiesced(
            prometheus_url, metrics, timeout=quiesce_timeout
        )
        if not settled:
            pytest.fail(
                f"NCCL metric totals still moving after {quiesce_timeout}s; cannot "
                "establish an idle baseline"
            )
        baseline_by_host = {m: metric_totals_by(prometheus_url, m) for m in metrics}

        time.sleep(observation_period)

        increased: dict[str, tuple[float | None, float | None]] = {}
        leaky_hosts: dict[str, list[str]] = {}
        for name in metrics:
            current = metric_total(prometheus_url, name)
            if metric_increased(baseline[name], current):
                increased[name] = (baseline[name], current)
            for host, total in metric_totals_by(prometheus_url, name).items():
                if metric_increased(baseline_by_host[name].get(host), total):
                    leaky_hosts.setdefault(host, []).append(name)

        print(
            f"\n  Idled {observation_period}s after the totals settled; "
            f"{len(increased)}/{len(metrics)} metrics moved"
        )
        assert not increased and not leaky_hosts, (
            "NCCL metric totals rose with no workload running: "
            + ", ".join(
                f"{name} {_format_value(before)} -> {_format_value(after)}"
                for name, (before, after) in increased.items()
            )
            + (f"; per-host movement: {leaky_hosts}" if leaky_hosts else "")
            + ". Either something else is driving NCCL ops, or the quiesce period is too "
            "short and the assertions in this file cannot tell activity from a live exporter."
        )
