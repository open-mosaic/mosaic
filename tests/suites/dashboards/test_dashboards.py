# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
Tests for Grafana dashboards provisioning and availability.

Validates that dashboards.yml references real repo files and that dashboards
in Grafana match provisioning.
"""

from pathlib import Path
from typing import Any

import pytest
import requests

# Container parent directory for options.path in dashboards.yml (compose mount target).
_PROVISIONING_DIR = Path("/var/lib/grafana/dashboards")


# =============================================================================
# Grafana Dashboards Tests
# =============================================================================


@pytest.mark.dashboards
class TestGrafanaDashboards:
    """Tests for Grafana dashboard provisioning and availability."""

    def test_dashboard_json_files_listed_in_yml(
        self,
        dashboards_dir: str,
        dashboards_yml_provisioning_paths: list[str],
    ):
        """
        :title: Provisioning - dashboards.yml paths exist in the repo
        :suite: dashboards
        :description: Every path listed in dashboards.yml must exist as a JSON file
            in the dashboards directory, and must use the intended container path
            (/var/lib/grafana/dashboards/<file>.json) so it resolves to that file
            when the stack mounts deployments/dashboards there.
        """
        if not dashboards_yml_provisioning_paths:
            pytest.fail(
                f"No dashboard paths found in dashboards.yml under {dashboards_dir} "
                "(is DASHBOARDS_DIR set and the directory mounted?)"
            )
        if not Path(dashboards_dir).is_dir():
            pytest.fail(f"Dashboards directory does not exist: {dashboards_dir}")

        errors: list[str] = []
        seen_basenames: set[str] = set()
        for path_str in dashboards_yml_provisioning_paths:
            path_obj = Path(path_str)
            basename = path_obj.name

            if path_obj.suffix != ".json":
                errors.append(f"{path_str!r}: expected a .json file")
                continue

            # Path in YAML must be the Grafana container mount point + basename
            # (no subdirectories), matching deployments/docker-compose.yml.
            try:
                if path_obj.parent != _PROVISIONING_DIR:
                    errors.append(
                        f"{path_str!r}: parent must be {_PROVISIONING_DIR} "
                        f"(got {path_obj.parent})"
                    )
                    continue
            except ValueError:
                errors.append(f"{path_str!r}: invalid path")
                continue

            if basename in seen_basenames:
                errors.append(f"{path_str!r}: duplicate basename {basename!r}")
                continue
            seen_basenames.add(basename)

            repo_file = Path(dashboards_dir) / basename
            if not repo_file.is_file():
                errors.append(
                    f"{path_str!r}: repo file missing at {repo_file} "
                    "(path must resolve to this file)"
                )

        assert not errors, "dashboards.yml provisioning errors:\n" + "\n".join(errors)

    def test_grafana_dashboard_availability_matches_expected(
        self,
        grafana_url: str,
        expected_dashboard_filenames: set[str],
    ):
        """
        :title: Availability - Available Grafana dashboards matches dashboards.yml
        :suite: dashboards
        :description: The number of dashboards returned by Grafana equals the
            number of dashboards declared in dashboards.yml.
        """
        if not expected_dashboard_filenames:
            pytest.fail(
                "No dashboard paths in dashboards.yml "
                "(is DASHBOARDS_DIR set and the directory mounted?)"
            )

        dashboard_list_url = f"{grafana_url}/apis/dashboard.grafana.app/v1beta1/namespaces/default/dashboards"
        dashboard_list: list[str] = []
        params: dict[str, str] = {}

        while True:
            try:
                response = requests.get(dashboard_list_url, params=params, timeout=10)
                if response.status_code == 200:
                    dashboards_json = response.json()
                    dashboard_list.extend(self._dashboard_list_from_json(dashboards_json))

                    if more_dashboards := self._more_dashboards_available(dashboards_json):
                        params.update(more_dashboards)
                    else:
                        break
                else:
                    pytest.fail(f"Grafana dashboard list API failed at {grafana_url}")

            except requests.exceptions.RequestException:
                pytest.fail(f"Grafana not accessible at {grafana_url}")

        # Verify that all expected dashboards are in the list
        for dashboard in expected_dashboard_filenames:
            assert dashboard in dashboard_list, f"Grafana is missing expected dashboard: {dashboard}"

    def test_each_dashboard_loads(
        self,
        grafana_url: str,
    ):
        """
        :title: Connectivity - Each dashboard can be loaded
        :suite: dashboards
        :description: For each dashboard returned by Grafana search, the
            dashboard API returns 200 and valid dashboard JSON (page loads).
        """
        search_url = f"{grafana_url}/api/search"
        params = {"type": "dash-db"}

        try:
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code != 200:
                pytest.fail(f"Grafana search failed at {grafana_url}")
            dashboards = response.json()

        except requests.exceptions.RequestException:
            pytest.fail(f"Grafana not accessible at {grafana_url}")

        if not dashboards:
            pytest.fail(f"No dashboards found in Grafana at {grafana_url}")

        failed = []
        for dashboard in dashboards:
            uid = dashboard.get("uid")
            title = dashboard.get("title", uid or "?")
            if not uid:
                failed.append((None, title, "missing uid"))
                continue
            if "url" in dashboard:
                dashboard_url = f"{grafana_url}{dashboard['url']}"
            else:
                continue

            try:
                r = requests.get(dashboard_url, timeout=10)
                if r.status_code != 200:
                    failed.append((uid, title, f"HTTP {r.status_code}"))
                    continue

            except requests.exceptions.RequestException as e:
                failed.append((uid, title, str(e)))

        assert not failed, (
            "One or more dashboards failed to load: "
            + "; ".join(f"{t} (uid={u}): {msg}" for u, t, msg in failed)
        )

    def _dashboard_list_from_json(self, dashboard_json: dict[str, Any]) -> list[str]:
        dashboards = []

        for item in dashboard_json["items"]:
            annotations = item["metadata"]["annotations"]
            if "grafana.app/sourcePath" in annotations:
                dashboard_path = Path(annotations["grafana.app/sourcePath"]).name
                dashboards.append(dashboard_path)
        return dashboards
    
    def _more_dashboards_available(self, dashboard_json: dict[str, Any]) -> dict[str, str] | None:
        if "continue" in dashboard_json["metadata"]:
            return {"continue": dashboard_json["metadata"]["continue"]}
        return None
