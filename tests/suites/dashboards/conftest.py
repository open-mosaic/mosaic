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


def _dashboard_provisioning_paths_from_yml(dashboards_dir: str) -> list[str]:
    """
    Full path string for each provider options.path entry in dashboards.yml.
    Stdlib-only parsing (line regex).
    """
    yml_path = os.path.join(dashboards_dir, "dashboards.yml")
    if not os.path.isfile(yml_path):
        return []
    paths: list[str] = []
    path_re = re.compile(r"^\s*path:\s*(\S+)\s*$")
    with open(yml_path, encoding="utf-8") as f:
        for line in f:
            m = path_re.match(line)
            if m:
                paths.append(m.group(1))
    return paths


def _parse_dashboards_yml_paths(dashboards_dir: str) -> set[str]:
    """Basenames of dashboard JSON files declared in dashboards.yml."""
    return {Path(p).name for p in _dashboard_provisioning_paths_from_yml(dashboards_dir)}


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
def dashboards_yml_provisioning_paths(dashboards_dir: str) -> list[str]:
    """
    Full provisioning path strings from dashboards.yml (container paths).
    """
    return _dashboard_provisioning_paths_from_yml(dashboards_dir)
