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
from production_test_framework.workload.workload import WorkloadStatus

from profiler_otel.conftest import (
    NCCL_PROFILER_METRICS_EXPECTED_PROMPT_WORKLOAD,
    QUIESCE_TIMEOUT,
    expected_nccl_profiler_metrics,
    metric_increased,
    metric_total,
    wait_for_metrics_quiesced,
)

# =============================================================================
# NCCL Profiler Telemetry Tests
# =============================================================================
WORKLOAD_TIMEOUT = 600  # 10 minutes

# How long to wait for NCCL metrics to reach Prometheus after a workload
# completes, and how often to re-check while waiting. Worst case between
# the last NCCL op and a sample being queryable is roughly 16s. This
# timeout is ~4x that budget.
METRICS_AVAILABLE_TIMEOUT = 60  # seconds
METRICS_POLL_INTERVAL = 2  # seconds

# How long the idle guard watches for movement. Matches the budget a real workload gets
# for its metrics to land, so a leak has the same opportunity to show up there as here.
IDLE_OBSERVATION_PERIOD = METRICS_AVAILABLE_TIMEOUT  # seconds


def _format_value(value: float | None) -> str:
    """Render a metric total for logs, distinguishing an absent series from a zero one."""
    return "absent" if value is None else f"{value:.6g}"


def _format_totals(totals: dict[str, float | None]) -> str:
    return ", ".join(f"{name}={_format_value(value)}" for name, value in totals.items())


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
        prometheus_url: str,
    ):
        """
        :title: Telemetry - NCCL metrics exported after inference
        :suite: profiler_otel
        :description: Verify NCCL profiler metrics are exported to Prometheus. 
            Waits for the metric totals to settle, snapshots them.
            Runs the workload, then requires each expected metric from conftest
            (NCCL_PROFILER_METRICS_EXPECTED_*) to increase above that baseline.
        """
        nccl_profiler_metrics = expected_nccl_profiler_metrics(workload)

        # Settle first, then snapshot. The collector republishes every series it has seen
        # on each scrape, so "a sample exists" is true forever after the first NCCL op and
        # proves nothing about this workload. Only a rise above a settled baseline does.
        print("\n  Waiting for NCCL metric totals to settle before taking a baseline...")
        baseline, settled = wait_for_metrics_quiesced(prometheus_url, nccl_profiler_metrics)
        if not settled:
            print(
                f"  WARNING: totals still moving after {QUIESCE_TIMEOUT}s; baseline may "
                "include samples from a previous workload"
            )
        print(f"  Baseline ({len(baseline)} metrics): {_format_totals(baseline)}")

        workload.start()

        # wait for workload to complete
        is_done = workload.wait_for_completion(timeout=WORKLOAD_TIMEOUT)
        assert is_done is True, (
            "Workload did not complete within timeout of {WORKLOAD_TIMEOUT} seconds"
        )

        # get inference result
        workload_result = workload.get_result()

        print("Workload result:")
        match workload_result.result:
            case str():
                print(workload_result.result)
            case InferenceResult():
                print(f"  Generated {len(workload_result.result.text)} characters")
                print(f"  Usage: {workload_result.result.usage}")
                print(f"  Text: {workload_result.result.text}")

        print(f"  Workload runtime: {workload_result.runtime:.1f}s")

        assert (
            workload_result is not None
            and workload_result.status == WorkloadStatus.COMPLETED
        ), "Inference must succeed before checking metrics"

        # Metrics take time to reach Prometheus after the workload finishes (empirically
        # ~10s). Rather than sleeping a fixed amount, poll each expected metric and record
        # it as soon as its total rises above the baseline, returning as soon as they all
        # have or the timeout elapses.
        print(
            f"  Workload window: start={workload_result.start_time:.3f} "
            f"end={workload_result.end_time:.3f}"
        )

        increased_metrics: dict[str, float] = {}
        deadline = time.monotonic() + METRICS_AVAILABLE_TIMEOUT
        while True:
            # Only re-query metrics that have not risen yet; record each as it does.
            for metric_name in nccl_profiler_metrics:
                if metric_name in increased_metrics:
                    continue
                current = metric_total(prometheus_url, metric_name)
                if metric_increased(baseline[metric_name], current):
                    increased_metrics[metric_name] = current
                    print(
                        f"    Increased: {metric_name} "
                        f"{_format_value(baseline[metric_name])} -> {_format_value(current)} "
                        f"(+{time.time() - workload_result.end_time:.1f}s after workload end)"
                    )

            if (
                len(increased_metrics) == len(nccl_profiler_metrics)
                or time.monotonic() >= deadline
            ):
                break
            time.sleep(METRICS_POLL_INTERVAL)

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
                    f"{METRICS_AVAILABLE_TIMEOUT}s)"
                )

        assert not missing_metrics, (
            f"Expected {len(nccl_profiler_metrics)} metrics to increase, "
            f"{len(increased_metrics)} did within {METRICS_AVAILABLE_TIMEOUT}s. "
            f"Did not increase: {missing_metrics}"
        )

    def test_metrics_do_not_increase_without_a_workload(self, prometheus_url: str):
        """
        :title: Telemetry - NCCL metrics do not increase while idle
        :suite: profiler_otel
        :description: Falsifiability guard for the metric assertions. The collector
            republishes every series it has seen on each scrape, so any check based on a
            sample being present passes even when nothing ran. This test runs no workload
            and requires the totals to stay flat -- if it fails, the assertions in this
            file have stopped distinguishing real NCCL activity from a live exporter.
        """
        metrics = NCCL_PROFILER_METRICS_EXPECTED_PROMPT_WORKLOAD

        baseline, settled = wait_for_metrics_quiesced(prometheus_url, metrics)
        if not settled:
            pytest.skip(
                f"NCCL metric totals still moving after {QUIESCE_TIMEOUT}s; cannot "
                "establish an idle baseline"
            )

        time.sleep(IDLE_OBSERVATION_PERIOD)

        increased: dict[str, tuple[float | None, float | None]] = {}
        for name in metrics:
            current = metric_total(prometheus_url, name)
            if metric_increased(baseline[name], current):
                increased[name] = (baseline[name], current)

        print(
            f"\n  Idled {IDLE_OBSERVATION_PERIOD}s after the totals settled; "
            f"{len(increased)}/{len(metrics)} metrics moved"
        )
        assert not increased, (
            "NCCL metric totals rose with no workload running: "
            + ", ".join(
                f"{name} {_format_value(before)} -> {_format_value(after)}"
                for name, (before, after) in increased.items()
            )
            + ". Either something else is driving NCCL ops, or the quiesce period is too "
            "short and the assertions in this file cannot tell activity from a live exporter."
        )
