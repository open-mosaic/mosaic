# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

import os

import pytest


def pytest_configure(config):
    """Register custom markers to avoid warnings."""
    config.addinivalue_line("markers", "profiler_otel: marks tests as NCCL profiler OTEL tests")
    config.addinivalue_line("markers", "dashboards: marks tests as Grafana dashboards integration tests")


@pytest.fixture(scope="session")
def grafana_url() -> str:
    """
    Provide the Grafana URL. Used by profiler_otel and dashboards suites.
    """
    host = os.getenv("GRAFANA_HOST", "localhost")
    port = os.getenv("GRAFANA_PORT", "3000")
    return f"http://{host}:{port}"