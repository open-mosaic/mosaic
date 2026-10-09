# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
Description of the deployment a run targeted, for the report's header tables.

The rows themselves come from :mod:`production_test_framework.reporting.environment`, shared
with other suites so their reports read the same way. Only what this suite adds on top --
the endpoint override and how the deployment was stood up -- is decided here.
"""

import os

from production_test_framework.reporting.environment import (
    display_path,
    endpoint_address,
    gpu_rows,
    profile_host,
    profile_rows,
    run_rows,
)

from profiler_otel import profiles

__all__ = ["environment_rows"]


def environment_rows(config, profile: profiles.Profile, prometheus_url: str, grafana_url: str) -> list[list[str]]:
    """
    What this run was pointed at, as (property, value) rows for the report's top table.
    """
    spec = profile.model_dump(mode="json")

    override = [f"{key}={os.environ[key]}" for key in ("VLLM_HOST", "VLLM_PORT") if key in os.environ]
    endpoint = endpoint_address(spec) + (f"  (overridden: {', '.join(override)})" if override else "")
    deployment = "external" if profile.is_external else f"compose: {display_path(profile.compose_file)}"

    rows = profile_rows(spec, prometheus_url, grafana_url, endpoint=endpoint, deployment=deployment)
    # The rootdir's checkout is this suite's, which is the code under test even when another
    # repository's Makefile cloned it and drove the run.
    return rows + run_rows(config) + gpu_rows(profile_host(spec))
