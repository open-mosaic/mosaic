# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
Tests for hardware profile discovery and validation.

These run without any deployment, so a malformed profile fails in seconds rather than after a
cluster has been brought up. They validate every profile on the search path, not just the one
selected -- pointing --profile-dir at a private directory therefore validates those profiles
too, in a repository whose CI may not otherwise run this suite.
"""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from profiler_otel import profiles
from profiler_otel.conftest import METRIC_SETS_BY_NAME

# Path a profile file is pretended to live at when a test parses a dict directly. Only its
# parent is used, to resolve the compose path in `_profile_file`.
FAKE_PROFILE_PATH = profiles.BUILTIN_PROFILE_DIR / "unit-test.yaml"


def _profile_file() -> dict[str, Any]:
    """A minimal valid single-machine, self-deployed profile."""
    return {
        "description": "unit test profile",
        "hardware": {"machines": 1, "gpus_per_machine": 2, "sku": "test-sku"},
        "serving": {"mode": "aggregated", "model": "test/model", "tensor_parallel": 2},
        "deployment": {"compose_file": "../../../../deployments/docker-compose-vllm.yml"},
        "endpoint": {"host": "localhost", "port": 8080},
        "coverage": {"hosts": 1, "gpus": 2, "communicators": 1},
        "timeouts": {"workload": 900, "metrics_available": 60, "quiesce": 90},
        "expected_metrics": "aggregated_dense",
        "benchmark_options": {"num_prompts": 4, "ignore_eos": True},
    }


def _disaggregated_profile_file() -> dict[str, Any]:
    """A minimal valid externally deployed, disaggregated profile."""
    raw = _profile_file()
    raw["hardware"] = {"machines": 2, "gpus_per_machine": 8, "sku": "test-sku"}
    raw["serving"] = {
        "mode": "disaggregated",
        "model": "test/model",
        "prefill": {"nodes": 2, "tensor_parallel": 8, "spans_nodes": True},
        "decode": {"nodes": 2, "tensor_parallel": 8, "spans_nodes": True},
        "kv_transfer": "nixl",
    }
    raw["deployment"] = {"external": True}
    raw["endpoint"] = {"base_url": "http://frontend:8000"}
    raw["coverage"] = {"hosts": 2, "gpus": 16, "communicators": 2}
    raw["expected_metrics"] = "disagg_moe"
    return raw


def _parse(raw: dict[str, Any]) -> profiles.Profile:
    return profiles.parse_profile("unit-test", raw, FAKE_PROFILE_PATH)

# Synthetic profiles exercising shapes this repository ships support for but no longer ships
# a profile for: externally deployed clusters, a frontend base_url, multi-host coverage and
# more than one NCCL communicator. Those live in the environments that run them, so without
# these fixtures the code would be shipped from here and only tested elsewhere.
FIXTURE_PROFILE_DIR = Path(__file__).parent / "testdata" / "profiles"


@pytest.mark.profiler_otel
class TestProfiles:
    """Tests for the profile loader."""

    def test_builtin_profiles_are_discovered(self):
        """
        :title: Profiles - built-in profiles are discovered
        :suite: profiler_otel
        :description: The profiles shipped in this repository are found by a default search.
            Only 'default' ships here; every other tier is maintained alongside the
            environment that runs it and supplied via --profile-dir.
        """
        found = profiles.discover(profiles.profile_dirs())
        assert "default" in found, f"the CI profile must exist; found {sorted(found)}"

    def test_test_fixtures_are_not_on_the_default_search_path(self):
        """
        :title: Profiles - test fixtures are not discoverable by default
        :suite: profiler_otel
        :description: The synthetic fixtures must never be selectable by a real run, or a
            typo'd --workload-profile could pick one up and point a test at a nonexistent
            cluster.
        """
        found = profiles.discover(profiles.profile_dirs())
        assert not [n for n in found if n.startswith("fixture_")], (
            f"test fixtures leaked onto the default search path: {sorted(found)}"
        )

    def test_extra_profile_dir_is_searched(self):
        """
        :title: Profiles - --profile-dir extends the search path
        :suite: profiler_otel
        :description: A directory outside this repository can supply profiles without
            forking the suite. This is what lets the internal tiers live elsewhere.
        """
        dirs = profiles.profile_dirs([str(FIXTURE_PROFILE_DIR)])
        found = profiles.discover(dirs)
        assert "default" in found and "fixture_external_disagg" in found, (
            f"expected built-in and supplied profiles together, got {sorted(found)}"
        )

    def test_later_profile_dir_wins_on_duplicate_name(self, tmp_path):
        """
        :title: Profiles - a supplied directory can override a shipped profile
        :suite: profiler_otel
        :description: An environment with its own 'default' should be able to override the
            one shipped here rather than having to rename it.
        """
        shipped = profiles.load("default", profiles.profile_dirs())
        override = tmp_path / "default.yaml"
        # Absolute compose path: the copy lives outside the repo, so the shipped profile's
        # relative one would (correctly) fail validation from here.
        override.write_text(
            shipped.path.read_text()
            .replace("description: ", "description: OVERRIDDEN ", 1)
            .replace(
                "compose_file: ../../../../deployments/docker-compose-vllm.yml",
                f"compose_file: {shipped.compose_file}",
            )
        )
        loaded = profiles.load("default", profiles.profile_dirs([str(tmp_path)]))
        assert loaded.path == override
        assert loaded.description.startswith("OVERRIDDEN")

    def test_external_profile_has_no_compose_file(self):
        """
        :title: Profiles - an externally deployed profile exposes no compose file
        :suite: profiler_otel
        :description: The scale tiers are stood up out of band, so `make setup` must have
            nothing to bring up. Exercised through a fixture because no such profile ships
            in this repository.
        """
        profile = profiles.load(
            "fixture_external_disagg", profiles.profile_dirs([str(FIXTURE_PROFILE_DIR)])
        )
        assert profile.is_external is True
        assert profile.compose_file is None
        assert profile.endpoint.base_url is not None
        assert profile.coverage.communicators >= 2

    def test_compose_file_resolves_relative_to_the_profile(self):
        """
        :title: Profiles - compose paths resolve against the profile's own directory
        :suite: profiler_otel
        :description: This is what allows a profile and its compose file to be moved into
            another repository together. The fixture sits at a different depth from the
            shipped profile, so a path resolved against the repo root would not find it.
        """
        profile = profiles.load(
            "fixture_compose_relative", profiles.profile_dirs([str(FIXTURE_PROFILE_DIR)])
        )
        assert profile.compose_file is not None
        assert profile.compose_file.is_file(), (
            f"{profile.compose_file} does not exist; compose paths are resolved against "
            "the profile's directory"
        )

    def test_all_discovered_profiles_are_valid(self, request):
        """
        :title: Profiles - every profile on the search path validates
        :suite: profiler_otel
        :description: Load and validate every profile, including any supplied via
            --profile-dir. Catches a typo'd key or a compose path with the wrong number of
            ".." segments without needing the hardware the profile describes.
        """
        dirs = profiles.profile_dirs(
            [*request.config.getoption("profile_dirs"), str(FIXTURE_PROFILE_DIR)]
        )
        found = profiles.discover(dirs)
        assert found, f"no profiles found on the search path: {dirs}"

        errors = []
        for name in sorted(found):
            try:
                profiles.load(name, dirs)
            except profiles.ProfileError as exc:
                errors.append(str(exc))
        assert not errors, "invalid profile(s):\n  - " + "\n  - ".join(errors)

    def test_expected_metrics_names_resolve_to_a_metric_list(self, request):
        """
        :title: Profiles - expected_metrics names a real metric set
        :suite: profiler_otel
        :description: Every profile's expected_metrics must map to a list in conftest, or the
            run would fail only once it reached the assertion.
        """
        dirs = profiles.profile_dirs(request.config.getoption("profile_dirs"))
        for name, path in sorted(profiles.discover(dirs).items()):
            profile = profiles.load(name, dirs)
            key = profile.expected_metrics
            assert key in METRIC_SETS_BY_NAME, (
                f"profile {name!r} ({path}) names metric set {key!r}, which conftest does "
                f"not define; known: {sorted(METRIC_SETS_BY_NAME)}"
            )

@pytest.mark.profiler_otel
class TestProfilesCli:
    """
    Tests for the ``python -m profiler_otel.profiles`` query interface.

    This repository's Makefile does not use it -- only 'default' ships here, and its compose
    file is hardcoded. It exists for orchestration in the environments that own the other
    tiers, so it needs covering here or it is untested code in a public repository.
    """

    def test_list_prints_discovered_names(self, capsys):
        """
        :title: Profiles CLI - list prints the discovered profile names
        :suite: profiler_otel
        :description: One name per line, so a caller can iterate them in a shell.
        """
        assert profiles._main(["list", "--profile-dir", str(FIXTURE_PROFILE_DIR)]) == 0
        names = capsys.readouterr().out.split()
        assert "default" in names and "fixture_external_disagg" in names

    def test_compose_file_prints_the_resolved_path(self, capsys):
        """
        :title: Profiles CLI - compose-file prints an absolute, resolved path
        :suite: profiler_otel
        :description: Callers should not have to re-implement resolution relative to the
            profile's own directory.
        """
        assert profiles._main(["compose-file", "default"]) == 0
        printed = Path(capsys.readouterr().out.strip())
        assert printed.is_absolute() and printed.is_file()

    def test_compose_file_prints_nothing_for_an_external_profile(self, capsys):
        """
        :title: Profiles CLI - compose-file is empty for an external deployment
        :suite: profiler_otel
        :description: Empty output and a zero exit mean "nothing to bring up", so a caller
            can branch on it without special-casing.
        """
        code = profiles._main(
            ["compose-file", "fixture_external_disagg", "--profile-dir", str(FIXTURE_PROFILE_DIR)]
        )
        assert code == 0
        assert capsys.readouterr().out.strip() == ""

    def test_deployment_env_prints_the_profile_facts(self, capsys):
        """
        :title: Profiles CLI - deployment-env exposes the values a deployment needs
        :suite: profiler_otel
        :description: A compose file or deployment script should take the model, parallelism,
            port and GPU count from the profile rather than restating them, which is what
            makes the numbers drift apart.
        """
        assert profiles._main(["deployment-env", "default"]) == 0
        env = dict(line.split("=", 1) for line in capsys.readouterr().out.split())
        assert env["PROFILE_NAME"] == "default"
        assert env["PROFILE_MODEL"] == "Qwen/Qwen3-8B"
        assert env["PROFILE_GPUS_PER_MACHINE"] == "2"
        assert env["PROFILE_TENSOR_PARALLEL"] == "2"
        assert env["PROFILE_PORT"] == "8080"

    def test_deployment_env_matches_the_loaded_profile(self, capsys):
        """
        :title: Profiles CLI - deployment-env agrees with the profile it read
        :suite: profiler_otel
        :description: The point of emitting these is that they cannot disagree with the
            profile; assert that directly rather than against literals.
        """
        profile = profiles.load("default", profiles.profile_dirs())
        profiles._main(["deployment-env", "default"])
        env = dict(line.split("=", 1) for line in capsys.readouterr().out.split())
        assert env["PROFILE_MODEL"] == profile.serving.model
        assert env["PROFILE_TENSOR_PARALLEL"] == str(profile.serving.tensor_parallel)
        assert env["PROFILE_GPUS_PER_MACHINE"] == str(profile.hardware.gpus_per_machine)
        assert env["PROFILE_PORT"] == str(profile.endpoint.port)

    def test_deployment_env_covers_externally_deployed_profiles(self, capsys):
        """
        :title: Profiles CLI - deployment-env works for an externally deployed profile
        :suite: profiler_otel
        :description: "external" means this suite does not bring the stack up, not that
            nothing does. A harness in another repository owns those deployments and still
            needs the machine's shape from the profile, so the facts must be emitted.
        """
        code = profiles._main(
            ["deployment-env", "fixture_external_disagg", "--profile-dir", str(FIXTURE_PROFILE_DIR)]
        )
        assert code == 0
        env = dict(line.split("=", 1) for line in capsys.readouterr().out.split())
        assert env["PROFILE_MODEL"] == "fixture/model"
        assert env["PROFILE_GPUS_PER_MACHINE"] == "8"

    def test_deployment_env_omits_fields_a_profile_does_not_define(self, capsys):
        """
        :title: Profiles CLI - deployment-env omits what the profile leaves unset
        :suite: profiler_otel
        :description: A disaggregated profile has no single tensor-parallel size, and one
            reached through a frontend has no port. Emitting an empty value would push a
            blank into a command line; omitting lets the consumer keep its own default.
        """
        profiles._main(
            ["deployment-env", "fixture_external_disagg", "--profile-dir", str(FIXTURE_PROFILE_DIR)]
        )
        env = dict(line.split("=", 1) for line in capsys.readouterr().out.split())
        assert "PROFILE_TENSOR_PARALLEL" not in env, "disaggregated profiles have no single TP"
        assert "PROFILE_PORT" not in env, "a base_url profile has no port"

    def test_unknown_profile_exits_nonzero(self, capsys):
        """
        :title: Profiles CLI - an unknown profile is an error
        :suite: profiler_otel
        :description: Must not exit 0 with empty output, which a caller would read as
            "externally deployed" and silently skip a deployment it needed.
        """
        assert profiles._main(["compose-file", "nope"]) == 1
        captured = capsys.readouterr()
        assert captured.out.strip() == ""
        assert "unknown profile" in captured.err


@pytest.mark.profiler_otel
class TestProfileValidation:
    """
    Tests for the profile models.

    Pure tests: profiles are built as dicts and parsed directly. No hardware and no
    network access.
    """

    def test_a_minimal_profile_round_trips(self) -> None:
        """
        :title: Profiles - a minimal profile parses into typed fields
        :suite: profiler_otel
        :description:
            Parse a single-machine, self-deployed profile and check each section is
            exposed as a typed attribute rather than raw YAML.
        """
        profile = _parse(_profile_file())

        assert profile.name == "unit-test", "name did not round-trip"
        assert profile.hardware.gpus_per_machine == 2, "gpus_per_machine did not round-trip"
        assert profile.hardware.total_gpus == 2, "total_gpus should be machines * gpus"
        assert profile.serving.mode is profiles.ServingMode.AGGREGATED, "mode did not round-trip"
        assert profile.timeouts.workload == 900, "workload timeout did not round-trip"
        assert profile.coverage.gpus == 2, "coverage.gpus did not round-trip"
        assert profile.expected_metrics is profiles.MetricSet.AGGREGATED_DENSE, (
            "expected_metrics did not round-trip"
        )
        assert profile.benchmark_options == {"num_prompts": 4, "ignore_eos": True}, (
            "benchmark_options should pass through untouched"
        )

    def test_a_disaggregated_profile_round_trips(self) -> None:
        """
        :title: Profiles - a disaggregated profile parses its worker pools
        :suite: profiler_otel
        :description:
            The prefill and decode blocks, the base_url endpoint and the external
            deployment marker all survive parsing.
        """
        profile = _parse(_disaggregated_profile_file())

        assert profile.is_external is True, "external deployment not reported"
        assert profile.compose_file is None, "an external profile must expose no compose file"
        assert profile.endpoint.base_url == "http://frontend:8000", "base_url did not round-trip"
        assert profile.serving.prefill is not None, "prefill pool missing"
        assert profile.serving.prefill.spans_nodes is True, "spans_nodes did not round-trip"
        assert profile.serving.kv_transfer == "nixl", "kv_transfer did not round-trip"

    def test_profile_is_immutable(self) -> None:
        """
        :title: Profiles - a parsed profile cannot be modified
        :suite: profiler_otel
        :description:
            Profiles are session-scoped and shared by every test, so one test must not
            be able to change what a later one sees.
        """
        profile = _parse(_profile_file())

        with pytest.raises(ValidationError):
            profile.coverage.gpus = 99

    def test_unknown_key_is_rejected(self) -> None:
        """
        :title: Profiles - an unknown key is an error
        :suite: profiler_otel
        :description:
            A misspelled key must fail rather than be silently ignored, which would
            leave the intended setting at its default.
        """
        raw = _profile_file()
        raw["timeouts"]["quiese"] = 30

        with pytest.raises(profiles.ProfileError, match="quiese"):
            _parse(raw)

    @pytest.mark.parametrize(
        "section,key,value",
        [
            ("coverage", "gpus", 0),
            ("coverage", "hosts", -1),
            ("timeouts", "workload", 0),
            ("hardware", "gpus_per_machine", 0),
        ],
        ids=["zero-coverage-gpus", "negative-hosts", "zero-timeout", "zero-gpus-per-machine"],
    )
    def test_counts_and_timeouts_must_be_positive(self, section, key, value) -> None:
        """
        :title: Profiles - counts and timeouts must be positive
        :suite: profiler_otel
        :description:
            A zero or negative count describes hardware that cannot exist and a
            timeout that expires immediately, so both are rejected at parse time.
        """
        raw = _profile_file()
        raw[section][key] = value

        with pytest.raises(profiles.ProfileError, match=f"{section}.{key}"):
            _parse(raw)

    def test_coverage_may_not_exceed_the_hardware(self) -> None:
        """
        :title: Profiles - coverage must be satisfiable by the hardware
        :suite: profiler_otel
        :description:
            Coverage is what the assertions require, so a value larger than the
            machines or GPUs available is a test that can never pass.
        """
        raw = _profile_file()
        raw["coverage"]["gpus"] = 99

        with pytest.raises(profiles.ProfileError, match="coverage.gpus"):
            _parse(raw)

    def test_coverage_is_not_derived_from_the_hardware(self) -> None:
        """
        :title: Profiles - coverage below the hardware total is allowed
        :suite: profiler_otel
        :description:
            Machines times GPUs is the wrong answer for a disaggregated cluster, where
            a machine may host only a frontend with no GPUs doing work. Coverage is
            declared, so fewer GPUs than the hardware provides must parse.
        """
        raw = _disaggregated_profile_file()
        raw["coverage"] = {"hosts": 1, "gpus": 8, "communicators": 2}

        profile = _parse(raw)

        assert profile.coverage.gpus == 8, "under-declared coverage should be accepted"
        assert profile.hardware.total_gpus == 16, "hardware total should be unaffected"

    @pytest.mark.parametrize(
        "deployment",
        [
            {},
            {"compose_file": "x.yml", "external": True},
        ],
        ids=["neither", "both"],
    )
    def test_deployment_needs_exactly_one_source(self, deployment) -> None:
        """
        :title: Profiles - deployment names exactly one source
        :suite: profiler_otel
        :description:
            Naming neither leaves nothing to bring up and no marker saying so; naming
            both is ambiguous.
        """
        raw = _profile_file()
        raw["deployment"] = deployment

        with pytest.raises(profiles.ProfileError, match="deployment"):
            _parse(raw)

    @pytest.mark.parametrize(
        "endpoint",
        [
            {},
            {"host": "localhost", "port": 8080, "base_url": "http://frontend:8000"},
            {"host": "localhost"},
        ],
        ids=["neither", "both", "host-without-port"],
    )
    def test_endpoint_needs_exactly_one_form(self, endpoint) -> None:
        """
        :title: Profiles - endpoint is a host and port or a base_url
        :suite: profiler_otel
        :description:
            The two forms are mutually exclusive, and a host without a port is
            incomplete. Which form is used decides whether the workload is given
            --host/--port or --base-url.
        """
        raw = _profile_file()
        raw["endpoint"] = endpoint

        with pytest.raises(profiles.ProfileError, match="endpoint"):
            _parse(raw)

    def test_disaggregated_mode_requires_both_worker_pools(self) -> None:
        """
        :title: Profiles - a disaggregated profile declares prefill and decode
        :suite: profiler_otel
        :description:
            Coverage for a disaggregated tier expects a communicator per pool, so a
            missing pool would produce an assertion that cannot be satisfied.
        """
        raw = _disaggregated_profile_file()
        del raw["serving"]["decode"]

        with pytest.raises(profiles.ProfileError, match="decode"):
            _parse(raw)

    def test_aggregated_mode_rejects_worker_pools(self) -> None:
        """
        :title: Profiles - an aggregated profile declares no worker pools
        :suite: profiler_otel
        :description:
            Prefill and decode blocks in an aggregated profile would be silently
            ignored, hiding a profile that does not do what it appears to.
        """
        raw = _profile_file()
        raw["serving"]["prefill"] = {"tensor_parallel": 2}

        with pytest.raises(profiles.ProfileError, match="prefill"):
            _parse(raw)

    def test_unknown_metric_set_is_rejected(self) -> None:
        """
        :title: Profiles - expected_metrics must name a known list
        :suite: profiler_otel
        :description:
            Otherwise the run fails only once it reaches the assertion, long after the
            deployment has been brought up.
        """
        raw = _profile_file()
        raw["expected_metrics"] = "typo_set"

        with pytest.raises(profiles.ProfileError, match="expected_metrics"):
            _parse(raw)

    def test_every_metric_set_name_has_a_list_in_conftest(self) -> None:
        """
        :title: Profiles - the MetricSet enum matches the lists conftest defines
        :suite: profiler_otel
        :description:
            The enum and the dictionary are declared in different modules, so a name
            added to one and not the other would only fail at run time.
        """
        assert {m.value for m in profiles.MetricSet} == set(METRIC_SETS_BY_NAME), (
            "MetricSet and METRIC_SETS_BY_NAME have drifted apart"
        )

    def test_compose_path_that_does_not_resolve_is_rejected(self) -> None:
        """
        :title: Profiles - a compose path must exist relative to the profile
        :suite: profiler_otel
        :description:
            Paths resolve against the profile's own directory, so a wrong number of
            parent segments points at a plausible-looking directory. Caught here rather
            than as a confusing compose error later.
        """
        raw = _profile_file()
        raw["deployment"]["compose_file"] = "../../../deployments/docker-compose-vllm.yml"

        with pytest.raises(profiles.ProfileError, match="does not exist"):
            _parse(raw)

    def test_error_message_carries_the_profile_and_field_path(self) -> None:
        """
        :title: Profiles - errors name the profile and the offending field
        :suite: profiler_otel
        :description:
            A profile may be one of several on a search path, so the message has to
            say which file and which field, not just what was wrong.
        """
        raw = _profile_file()
        del raw["coverage"]["gpus"]

        with pytest.raises(profiles.ProfileError) as exc_info:
            _parse(raw)

        message = str(exc_info.value)
        assert "unit-test" in message, f"message does not name the profile: {message}"
        assert "coverage.gpus" in message, f"message does not name the field: {message}"
