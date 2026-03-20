# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
Grafana dashboards test fixtures.

This conftest provides fixtures for validating dashboards in deployments/dashboards
against Grafana. Uses the shared grafana_url from tests/suites/conftest.py.
"""

import os
import re
import time
from pathlib import Path

import pytest
import requests


def _parse_dashboards_yml_paths(dashboards_dir: str) -> set[str]:
    """Extract dashboard path basenames from dashboards.yml (stdlib-only)."""
    yml_path = os.path.join(dashboards_dir, "dashboards.yml")
    if not os.path.isfile(yml_path):
        return set()
    path_basenames = set()
    # Match lines like "      path: /var/lib/grafana/dashboards/foo.json"
    path_re = re.compile(r"^\s*path:\s*[\S]+/([^\s/]+\.json)\s*$")
    with open(yml_path, encoding="utf-8") as f:
        for line in f:
            m = path_re.match(line)
            if m:
                path_basenames.add(m.group(1))
    return path_basenames


@pytest.fixture(scope="session")
def dashboards_dir() -> str:
    """
    Path to the dashboards directory (e.g. /mnt/dashboards when run in container).
    """
    return os.getenv("DASHBOARDS_DIR", "/mnt/dashboards")


@pytest.fixture(scope="session")
def expected_dashboard_filenames(dashboards_dir: str) -> set[str]:
    """
    Set of dashboard JSON filenames declared in dashboards.yml.
    """
    return _parse_dashboards_yml_paths(dashboards_dir)


@pytest.fixture(scope="session")
def dashboard_json_files(dashboards_dir: str) -> list[str]:
    """
    List of *.json filenames in the dashboards directory.
    """
    if not os.path.isdir(dashboards_dir):
        return []
    return sorted(f.name for f in Path(dashboards_dir).glob("*.json"))
