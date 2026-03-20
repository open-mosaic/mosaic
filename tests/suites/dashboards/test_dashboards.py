# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
Tests for Grafana dashboards provisioning and availability.

Validates that dashboards in deployments/dashboards are listed in dashboards.yml
and are available and loadable in Grafana.
"""

from pathlib import Path
from typing import Any

import pytest
import requests


# =============================================================================
# Grafana Dashboards Tests
# =============================================================================


@pytest.mark.dashboards
class TestGrafanaDashboards:
    """Tests for Grafana dashboard provisioning and availability."""

    def test_dashboard_json_files_listed_in_yml(
        self,
        dashboards_dir: str,
        dashboard_json_files: list[str],
        expected_dashboard_filenames: set[str],
    ):
        """
        :title: Provisioning - All dashboard JSON files listed in dashboards.yml
        :suite: dashboards
        :description: Every *.json file in the dashboards directory is declared
            in dashboards.yml so that no dashboard is left out of provisioning.
        """
        if not dashboard_json_files:
            pytest.fail(
                f"No dashboard JSON files found in {dashboards_dir} "
                "(is DASHBOARDS_DIR set and the directory mounted?)"
            )
        missing = [
            f for f in dashboard_json_files if f not in expected_dashboard_filenames
        ]
        assert not missing, (
            f"Dashboard JSON file(s) not listed in dashboards.yml: {missing}. "
            "Add a provider with options.path pointing to each file."
        )

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
        dashboard_list = []
        params = {}

        # Get the list of dashboards from Grafana until all expected dashboards are found
        while True:
            try:
                response = requests.get(dashboard_list_url, params=params, timeout=10)
                if response.status_code == 200:
                    dashboards_json = response.json()
                    dashboards = self._dashboard_list_from_json(dashboards_json)

                    if more_dashboards := self._more_dashboards_available(dashboards_json):
                        params.update(more_dashboards)
                    else:
                        break
                    dashboards += dashboards

                else:
                    pytest.fail(f"Grafana dashboard list API failed at {grafana_url}")

            except requests.exceptions.RequestException:
                pytest.fail(f"Grafana not accessible at {grafana_url}")
        
        # Verify that all expected dashboards are in the list
        for dashboard in expected_dashboard_filenames:
            assert dashboard in dashboards, f"Grafana is missing expected dashboard: {dashboard}"

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
            return

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
