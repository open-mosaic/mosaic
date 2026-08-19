# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

import os

import pytest


DEFAULT_PROFILE = "default"


def pytest_addoption(parser):
    group = parser.getgroup("mosaic")
    group.addoption(
        "--workload-profile",
        action="store",
        default=DEFAULT_PROFILE,
        metavar="NAME",
        help=(
            "Hardware profile to run against, by filename stem in a --profile-dir "
            f"(default: {DEFAULT_PROFILE})."
        ),
    )
    group.addoption(
        "--profile-dir",
        action="append",
        default=[],
        metavar="PATH",
        dest="profile_dirs",
        help=(
            "Directory of profile YAML files; repeatable. Defaults to the in-repo "
            "profiler_otel/profiles/. Use this option so that a "
            "private repo can supply its own profiles without forking this suite."
        ),
    )


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
