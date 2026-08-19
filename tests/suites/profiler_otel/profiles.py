# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""Load and validate the hardware profiles the profiler OTEL suite runs against.

A profile describes one hardware configuration: the machines and GPUs it runs
on, the serving topology, the load to drive, and the profiler coverage to
expect back. It is a single YAML file whose stem is the profile name.

Only `default` ships in this repository. Other tiers are maintained alongside
the environment that runs them, together with that environment's own deployment
configuration, and are supplied at run time with `--profile-dir`. Paths inside a
profile therefore resolve relative to that profile's own directory, so a profile
and the compose file it names relocate together.

Per-field rules are declared on the models below. Rules that span sections -
coverage against the hardware that has to satisfy it, worker pools against the
serving mode that requires them - live in the `_ProfileSpec` validators, which
raise ProfileError with an explicit field path.
"""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    ValidationError,
    model_validator,
)

PROFILE_SUFFIX = "*.yaml"

#: Profiles shipped with this repository.
BUILTIN_PROFILE_DIR = Path(__file__).parent / "profiles"


class ServingMode(StrEnum):
    """How the deployment under test serves requests."""

    AGGREGATED = "aggregated"
    DISAGGREGATED = "disaggregated"


class MetricSet(StrEnum):
    """Names of the expected-metric lists defined in the suite's conftest."""

    AGGREGATED_DENSE = "aggregated_dense"
    AGGREGATED_MOE = "aggregated_moe"
    DISAGG_MOE = "disagg_moe"


class ProfileError(Exception):
    """Raised when a profile is missing or fails validation. Message carries the field path."""


# A required, non-blank string.
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
#: A count of physical things, so at least one.
Count = Annotated[int, Field(gt=0)]
#: A timeout in seconds.
Seconds = Annotated[int, Field(gt=0)]


class _Model(BaseModel):
    """Shared model behavior: snake_case YAML keys, no unknown keys, immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Hardware(_Model):
    """The machines a profile runs on.

    Documentation, and the bound `Coverage` is checked against. It is not the
    source of the coverage numbers themselves - see `Coverage`.
    """

    machines: Count
    gpus_per_machine: Count
    sku: Text

    @property
    def total_gpus(self) -> int:
        return self.machines * self.gpus_per_machine


class WorkerPool(_Model):
    """One side of a disaggregated deployment: the prefill or the decode workers.

    `spans_nodes` records that the pool's parallel group deliberately straddles
    machines. A pool pinned to a single machine emits no cross-node NCCL: every
    collective stays inside the node, and the only traffic between machines is
    the key-value cache transfer, which the transfer backend handles outside
    NCCL and the profiler never observes.
    """

    nodes: Count | None = None
    workers: Count | None = None
    tensor_parallel: Count
    expert_parallel: Count | None = None
    spans_nodes: StrictBool = False


class Serving(_Model):
    """The model being served and how it is parallelised.

    `prefill` and `decode` apply to a disaggregated deployment only; the
    aggregated form uses the top-level parallelism instead. `_ProfileSpec`
    rejects a profile that mixes the two.
    """

    mode: ServingMode
    model: Text
    tensor_parallel: Count | None = None
    expert_parallel: Count | None = None
    prefill: WorkerPool | None = None
    decode: WorkerPool | None = None
    kv_transfer: Text | None = None

    # `model` collides with pydantic's own namespace, which would otherwise warn.
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class Deployment(_Model):
    """Who brings the stack up.

    Exactly one of `compose_file` (this suite does) or `external` (it is
    already running). `compose_file` is as written in YAML; `Profile` exposes
    the resolved absolute path.
    """

    compose_file: Text | None = None
    external: StrictBool = False

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Deployment:
        if bool(self.compose_file) == self.external:
            raise ValueError(
                "needs exactly one of 'compose_file' or 'external: true'; "
                f"got compose_file={self.compose_file!r}, external={self.external}"
            )
        return self


class Endpoint(_Model):
    """Where the workload sends requests.

    Either a host and port, or a single `base_url` - a disaggregated cluster is
    reached through one frontend, which a host and port cannot express.
    """

    host: Text | None = None
    port: Annotated[int, Field(gt=0, lt=65536)] | None = None
    base_url: Text | None = None

    @model_validator(mode="after")
    def _exactly_one_form(self) -> Endpoint:
        has_host_port = self.host is not None or self.port is not None
        if bool(self.base_url) == has_host_port:
            raise ValueError(
                "needs either 'base_url' or 'host' and 'port', not both and not neither"
            )
        if has_host_port and (self.host is None or self.port is None):
            raise ValueError("'host' and 'port' must be given together")
        return self


class Coverage(_Model):
    """What the profiler must actually report during a run.

    Declared rather than derived from `Hardware`: on a disaggregated cluster the
    prefill and decode pools can differ in size, and a machine may host only a
    frontend or router with no GPU ranks, so machines times GPUs is the wrong
    answer.

    `communicators` matters because prefill and decode form separate NCCL
    communicators. A total summed across every series rises when only one pool
    is alive, so requiring more than one is what catches a dead decode pool.
    """

    hosts: Count
    ranks: Count
    communicators: Count


class Timeouts(_Model):
    """Per-run time budgets, in seconds.

    These differ by an order of magnitude between a two-GPU machine and a
    96-GPU cluster, which is why they are per profile rather than constants.
    """

    workload: Seconds
    metrics_available: Seconds
    quiesce: Seconds


class Profile(_Model):
    """A fully validated profile, with paths resolved against the file it came from."""

    name: Text
    path: Path
    description: Text
    hardware: Hardware
    serving: Serving
    deployment: Deployment
    endpoint: Endpoint
    coverage: Coverage
    timeouts: Timeouts
    expected_metrics: MetricSet
    benchmark_options: dict[str, Any]
    otel_endpoint: Text | None = None
    compose_file: Path | None = None

    @property
    def is_external(self) -> bool:
        """True when the deployment is stood up out of band, so setup does not apply."""
        return self.deployment.external


class _ProfileSpec(_Model):
    """The profile as written in YAML, before paths are resolved against its location."""

    description: Text
    hardware: Hardware
    serving: Serving
    deployment: Deployment
    endpoint: Endpoint
    coverage: Coverage
    timeouts: Timeouts
    expected_metrics: MetricSet
    benchmark_options: dict[str, Any]
    otel_endpoint: Text | None = None

    @model_validator(mode="after")
    def _check_serving_matches_mode(self) -> _ProfileSpec:
        pools = {"prefill": self.serving.prefill, "decode": self.serving.decode}
        if self.serving.mode is ServingMode.DISAGGREGATED:
            if missing := [name for name, pool in pools.items() if pool is None]:
                raise ProfileError(
                    f"serving: mode 'disaggregated' requires {' and '.join(missing)}"
                )
        elif present := [name for name, pool in pools.items() if pool is not None]:
            raise ProfileError(
                f"serving: mode 'aggregated' must not declare {' or '.join(present)}"
            )
        return self

    @model_validator(mode="after")
    def _check_coverage_fits_hardware(self) -> _ProfileSpec:
        # Coverage is what the assertions require, so a value the hardware cannot
        # satisfy is a test that can never pass. Caught here rather than after a
        # cluster has been brought up.
        if self.coverage.hosts > self.hardware.machines:
            raise ProfileError(
                f"coverage.hosts: {self.coverage.hosts} exceeds hardware.machines "
                f"({self.hardware.machines})"
            )
        if self.coverage.ranks > self.hardware.total_gpus:
            raise ProfileError(
                f"coverage.ranks: {self.coverage.ranks} exceeds the "
                f"{self.hardware.total_gpus} GPUs the hardware provides"
            )
        return self

    def resolve_compose_file(self, path: Path) -> Path | None:
        """Absolute compose path for a self-deployed profile, or None for an external one.

        Resolved against the profile's own directory, not the repository root, so
        a profile and its compose file can be moved into another repository
        together. A wrong number of parent segments otherwise points at a
        plausible-looking directory and surfaces much later as a compose error.
        """
        if self.deployment.compose_file is None:
            return None
        candidate = Path(self.deployment.compose_file)
        if not candidate.is_absolute():
            candidate = (path.parent / candidate).resolve()
        if not candidate.is_file():
            raise ProfileError(
                f"deployment.compose_file: {self.deployment.compose_file!r} resolves to "
                f"{candidate}, which does not exist (paths are relative to the profile's "
                "own directory)"
            )
        return candidate


def _format_location(loc: tuple[Any, ...]) -> str:
    """Render a pydantic error location as a field path, e.g. serving.prefill.nodes."""
    if not loc:
        return "profile"
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else str(part)
    return out


def profile_dirs(extra: list[str] | None = None) -> list[Path]:
    """Search path: the built-in directory first, then any caller-supplied ones."""
    return [BUILTIN_PROFILE_DIR, *(Path(p) for p in (extra or []))]


def discover(dirs: list[Path]) -> dict[str, Path]:
    """Map profile name to file across the search path.

    Later directories win on a duplicate name, so a supplied directory can
    legitimately override a profile shipped here.
    """
    found: dict[str, Path] = {}
    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(PROFILE_SUFFIX)):
            found[path.stem] = path
    return found


def parse_profile(name: str, data: Any, path: Path) -> Profile:
    """Validate the already-parsed contents of a profile file.

    Raises ProfileError.
    """
    where = f"profile {name!r} ({path})"
    try:
        spec = _ProfileSpec.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(f"{_format_location(e['loc'])}: {e['msg']}" for e in exc.errors())
        raise ProfileError(f"{where}: {details}") from exc
    except ProfileError as exc:
        # Raised by the cross-section validators, which report a field path of their own.
        raise ProfileError(f"{where}: {exc}") from exc

    try:
        compose_file = spec.resolve_compose_file(path)
    except ProfileError as exc:
        raise ProfileError(f"{where}: {exc}") from exc

    return Profile(
        name=name,
        path=path,
        compose_file=compose_file,
        **spec.model_dump(),
    )


def load(name: str, dirs: list[Path]) -> Profile:
    """Read and validate one profile by name.

    Raises ProfileError for an unknown name, malformed YAML, or any validation
    failure. Callers should surface that as a test failure, not a skip, so that
    a misconfigured run cannot report green:

        try:
            profile = load(name, profile_dirs(extra))
        except ProfileError as exc:
            pytest.fail(str(exc))
    """
    available = discover(dirs)
    if name not in available:
        searched = ", ".join(str(d) for d in dirs)
        known = ", ".join(sorted(available)) or "(none found)"
        raise ProfileError(f"unknown profile {name!r}. Available: {known}. Searched: {searched}")

    path = available[name]
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ProfileError(f"profile {name!r} ({path}) is not valid YAML: {exc}") from exc
    if not data:
        raise ProfileError(f"profile {name!r} ({path}) is empty")
    return parse_profile(name, data, path)


def _main(argv: list[str] | None = None) -> int:
    """Query profiles from a shell, for orchestration that is not this repository's Makefile.

        python -m profiler_otel.profiles compose-file <name> [--profile-dir DIR ...]
        python -m profiler_otel.profiles list [--profile-dir DIR ...]

    `compose-file` prints nothing and exits 0 for an externally deployed profile,
    so a caller can treat empty output as "nothing to bring up".
    """
    import argparse

    parser = argparse.ArgumentParser(prog="profiler_otel.profiles")
    parser.add_argument("command", choices=("compose-file", "list"))
    parser.add_argument("name", nargs="?")
    parser.add_argument("--profile-dir", action="append", dest="profile_dirs", default=[])
    args = parser.parse_args(argv)

    dirs = profile_dirs(args.profile_dirs)

    if args.command == "list":
        for name in sorted(discover(dirs)):
            print(name)
        return 0

    if not args.name:
        parser.error("compose-file requires a profile name")
    try:
        profile = load(args.name, dirs)
    except ProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if profile.compose_file is not None:
        print(profile.compose_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
