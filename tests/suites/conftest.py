# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest

from profiler_otel.reporting import ReportFormat, Reporter, ReportPlugin


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
    group.addoption(
        "--report-file",
        "--report-html",
        action="store",
        default=None,
        metavar="PATH",
        dest="report_file",
        help=(
            "Write one report to PATH: the run summary followed by the result tables the "
            "tests emit. Without this the same tables print to stdout, which is what CI and "
            "a bare pytest run want. --report-html is kept as an alias."
        ),
    )
    group.addoption(
        "--report-format",
        action="store",
        default=None,
        choices=[fmt.value for fmt in ReportFormat],
        help=(
            "Format for --report-file: html (default) or md. Left off, a .md or .markdown "
            "path infers md and anything else html."
        ),
    )


def pytest_configure(config):
    """Register custom markers to avoid warnings, and open the report."""
    config.addinivalue_line("markers", "profiler_otel: marks tests as NCCL profiler OTEL tests")
    config.addinivalue_line("markers", "dashboards: marks tests as Grafana dashboards integration tests")

    raw_path = config.getoption("report_file")
    path = Path(raw_path).expanduser() if raw_path else None
    chosen = config.getoption("report_format")
    fmt = ReportFormat(chosen) if chosen else (ReportFormat.for_path(path) if path else ReportFormat.HTML)

    config.mosaic_reporter = Reporter(path, fmt=fmt)
    config.pluginmanager.register(ReportPlugin(config.mosaic_reporter), "mosaic-report")


@pytest.fixture
def reporter(request):
    """
    The session's :class:`report.Reporter`, with a detail section open for this test.
    """
    instance = request.config.mosaic_reporter
    instance.start_test(request.node.nodeid)
    return instance


@pytest.fixture(scope="session")
def grafana_url() -> str:
    """
    Provide the Grafana URL. Used by profiler_otel and dashboards suites.
    """
    host = os.getenv("GRAFANA_HOST", "localhost")
    port = os.getenv("GRAFANA_PORT", "3000")
    return f"http://{host}:{port}"
