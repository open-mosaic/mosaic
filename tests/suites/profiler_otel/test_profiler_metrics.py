# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
Tests for vLLM inference with NCCL Profiler OTEL.

These tests validate that the NCCL profiler is exporting telemetry correctly.
"""

import math
import time

import pytest
import requests
from production_test_framework.vllm import InferenceResult
from production_test_framework.workload.workload import WorkloadStatus

from profiler_otel.conftest import expected_nccl_profiler_metrics

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
        ["prompt_workload", "inferencex_workload"],
        indirect=True,
        ids=["prompt_workload", "inferencex_workload"],
    )
    def test_nccl_metrics_exported_after_inference(
        self,
        workload,
        prometheus_url: str,
    ):
        """
        :title: Telemetry - NCCL metrics exported after inference
        :suite: profiler_otel
        :description: Verify NCCL profiler metrics are exported to Prometheus after
            running vLLM inference. Triggers NCCL operations via inference, then
            queries Prometheus for the workload-specific expected metric set from
            conftest (NCCL_PROFILER_METRICS_EXPECTED_*).
        """

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

        # Metrics take time to be scraped into Prometheus after the workload
        # finishes (empirically ~10s). Rather than sleeping a fixed amount, poll
        # the expected metric set and record each metric as soon as it appears,
        # returning immediately once all are present or once the timeout elapses.
        # We'll move the end time window for each iteration of the poll to make sure
        # we don't accidentally exclude any late arriving metrics.
        nccl_profiler_metrics = expected_nccl_profiler_metrics(workload)

        def query_prometheus(promql: str) -> list:
            """Run an instant query and return its result series (empty on any failure)."""
            try:
                response = requests.get(
                    f"{prometheus_url}/api/v1/query",
                    params={"query": promql},
                    timeout=10,
                )
            except requests.exceptions.RequestException:
                return []
            if response.status_code != 200:
                return []
            data = response.json()
            if data.get("status") != "success":
                return []
            return data.get("data", {}).get("result", [])

        def metric_has_data(metric_name: str) -> bool:
            """True when Prometheus holds a sample for *metric_name* timestamped after the workload started."""
            window_s = max(1, math.ceil(time.time() - workload_result.start_time) + 1)
            return bool(
                query_prometheus(f"count_over_time({metric_name}[{window_s}s])")
            )

        print(
            f"  Workload window: start={workload_result.start_time:.3f} "
            f"end={workload_result.end_time:.3f}"
        )

        found_metrics: list[str] = []
        deadline = time.monotonic() + METRICS_AVAILABLE_TIMEOUT
        while True:
            # Only re-query metrics we have not seen yet; record each as it arrives.
            for metric_name in nccl_profiler_metrics:
                if metric_name not in found_metrics and metric_has_data(metric_name):
                    found_metrics.append(metric_name)
                    print(
                        f"    Found: {metric_name} "
                        f"(+{time.time() - workload_result.end_time:.1f}s after workload end)"
                    )

            if (
                len(found_metrics) == len(nccl_profiler_metrics)
                or time.monotonic() >= deadline
            ):
                break
            time.sleep(METRICS_POLL_INTERVAL)

        missing_metrics = [m for m in nccl_profiler_metrics if m not in found_metrics]

        print(
            f"\n  Found {len(found_metrics)}/{len(nccl_profiler_metrics)} NCCL profiler metrics"
        )
        if missing_metrics:
            print(f"  Missing metrics: {missing_metrics}")
            # Re-query each missing metric with no time bound. Whether the series
            # exists, and where its newest sample sits relative to the workload,
            # separates the failure modes for whoever reads the CI log:
            #   no series at all    -> nothing was exported; check vLLM, the OTLP
            #                          endpoint and the collector
            #   newest > end_time   -> ingest lag; raise METRICS_AVAILABLE_TIMEOUT
            #   newest < start_time -> the series is stale; the profiler produced
            #                          nothing for this workload
            print("  Missing metrics diagnostics:")
            for metric_name in missing_metrics:
                series = query_prometheus(metric_name)
                if not series:
                    print(f"    {metric_name}: no series in Prometheus")
                    continue
                newest = max(float(s["value"][0]) for s in series)
                print(
                    f"    {metric_name}: {len(series)} series, newest sample "
                    f"{newest - workload_result.start_time:+.1f}s vs workload start timestamp of {workload_result.start_time:+.1f}, "
                    f"{newest - workload_result.end_time:+.1f}s vs workload end timestamp of {workload_result.end_time:+.1f}"
                )

        assert not missing_metrics, (
            f"Expected {len(nccl_profiler_metrics)} metrics, found {len(found_metrics)} "
            f"within {METRICS_AVAILABLE_TIMEOUT}s. Missing: {missing_metrics}"
        )
