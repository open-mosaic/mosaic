# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
Tests for vLLM inference with NCCL Profiler OTEL.

These tests validate that the NCCL profiler is exporting telemetry correctly.
"""

import time

import pytest
import requests

from framework.vllm import InferenceResult
from framework.workload.workload import WorkloadStatus

# =============================================================================
# NCCL Profiler Telemetry Tests
# =============================================================================


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
        nccl_profiler_metrics: list[str],
    ):
        """
        :title: Telemetry - NCCL metrics exported after inference
        :suite: profiler_otel
        :description: Verify NCCL profiler metrics are exported to Prometheus after
            running vLLM inference. Triggers NCCL operations via inference, then
            queries Prometheus for all metrics defined in telemetry.cc.
        """

        workload.start()

        # wait for workload to complete
        workload.wait_for_completion(timeout=600)

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

        assert workload_result is not None and workload_result.status == WorkloadStatus.COMPLETED, "Inference must succeed before checking metrics"
    

        # The metrics results take about 10 seconds to become available in Prometheus.
        time.sleep(10)

        found_metrics = []
        missing_metrics = []

        for metric_name in nccl_profiler_metrics:
            try:
                response = requests.get(
                    f"{prometheus_url}/api/v1/query_range",
                    params={"query": metric_name, "start": workload_result.start_time, "end": workload_result.end_time, "step": "1.0"},
                    timeout=10,
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        results = data.get("data", {}).get("result", [])
                        if results:
                            found_metrics.append(metric_name)
                            print(f"    Found: {metric_name} ({len(results)} series)")
                        else:
                            missing_metrics.append(metric_name)
                    else:
                        missing_metrics.append(metric_name)
                else:
                    missing_metrics.append(metric_name)
            except requests.exceptions.RequestException:
                missing_metrics.append(metric_name)

        print(f"\n  Found {len(found_metrics)}/{len(nccl_profiler_metrics)} NCCL profiler metrics")

        if missing_metrics:
            print(f"  Missing metrics: {missing_metrics}")

        assert len(found_metrics) == len(nccl_profiler_metrics), (
            f"Expected {len(nccl_profiler_metrics)} metrics, found {len(found_metrics)}. Missing: {missing_metrics}"
        )
