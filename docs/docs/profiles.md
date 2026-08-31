---
icon: fontawesome/solid/sliders
title: Hardware Profiles
---

<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# Hardware Profiles

A profile is a single YAML file describing one hardware configuration: the machines and GPUs it
runs on, the serving topology, the load to drive, and the profiler coverage to expect back. The
filename stem is the profile name.

Profiles exist because the [profiler OTEL suite](testsuite.md) has to run unchanged against
everything from a two-GPU CI machine to a ninety-six-GPU disaggregated cluster. The load, the
timeouts and the number of reporting ranks differ by orders of magnitude between them, and none
of that belongs in the test code.

They live in `tests/suites/profiler_otel/profiles/`.

## Which profiles ship here

Only `default` — the configuration continuous integration runs, matching
`deployments/docker-compose-vllm.yml`. It is selected when no profile is named, so nothing has
to pass arguments for the common case.

Other tiers are maintained alongside the environment that runs them, together with that
environment's own deployment configuration, and are supplied at run time with `--profile-dir`.
Keeping them out of this repository avoids publishing internal cluster topology, and lets each
environment change its own configuration without a change here.

## Selecting a profile

```bash title="Against the stack this repository brings up"
make test
```

```bash title="With profiles supplied from elsewhere"
uv run pytest --workload-profile internal --profile-dir /path/to/your/profiles
```

`--profile-dir` is repeatable and searched after the built-in directory, so a supplied profile
may also override one shipped here. An unknown name fails immediately and lists the names it
did find.

Paths inside a profile resolve relative to **that profile's own directory**, so a profile and
the deployment configuration it refers to can be moved into another repository together.

!!! note
    Profile selection is a command-line option rather than an environment variable, so a value
    left set in a shell cannot silently change a later run.

## Schema

| Key | Meaning |
|-----|---------|
| `description` | One line, printed at the start of a run |
| `hardware` | `machines`, `gpus_per_machine`, `sku`. Documentation, and the upper bound `coverage` is validated against |
| `serving` | `mode` (`aggregated` or `disaggregated`), `model`, and the parallelism. A disaggregated profile adds `prefill` and `decode` blocks |
| `deployment` | Exactly one of `compose_file` (this suite brings the stack up) or `external: true` (stood up out of band) |
| `endpoint` | Either `host` and `port`, or `base_url` for a single frontend |
| `otel_endpoint` | Where profiler telemetry is sent |
| `coverage` | The `hosts`, `ranks` and `communicators` the profiler must report from |
| `timeouts` | `workload`, `metrics_available` and `quiesce`, in seconds |
| `expected_metrics` | Names a metric list defined in the suite's `conftest.py` |
| `benchmark_options` | Passed verbatim to the benchmark workload |

### Benchmark options

Keys in `benchmark_options` become `benchmark_serving.py` flags by the usual conventions:
`num_prompts: 64` becomes `--num-prompts 64`, `ignore_eos: true` becomes `--ignore-eos`, and a
value of `None` omits the flag entirely. Nothing enumerates the arguments that script accepts,
so any flag it supports can be set from a profile without a code change.

Omitting a flag is how a profile drops a default it does not want. A cluster reached through a
single frontend sets `base_url` and clears the host and port:

```yaml title="Targeting one frontend instead of a host and port"
endpoint:
  base_url: "http://frontend:8000"
```

## Coverage

`coverage` is what the assertions check: how many hosts, ranks and communicators must actually
report telemetry during a run.

It is declared rather than calculated from `hardware`. On a disaggregated cluster the prefill
and decode pools can differ in size, and a machine may host only a frontend or a router with no
GPU ranks at all, so `machines` multiplied by `gpus_per_machine` is the wrong answer. The
`hardware` block stays as documentation and as a sanity bound.

`communicators` matters because prefill and decode form **separate NCCL communicators**. A total
summed across every series rises when only one pool is alive, so requiring more than one is what
catches a dead decode pool.

### Reading coverage from a running cluster

Rather than predicting the numbers, run once and read them back. The profiler labels every
series with `hostname`, `rank`, `local_rank` and `communicator`:

```text title="Prometheus queries"
sum by (hostname) (nccl_profiler_collective_bytes_total)
sum by (hostname, rank) (nccl_profiler_collective_bytes_total)
sum by (communicator) (nccl_profiler_collective_bytes_total)
```

### Cross-node traffic needs a pool that spans machines

A worker pool pinned to one machine produces no cross-node NCCL. Every collective stays inside
the node, and the only traffic between machines is the key-value cache transfer, which the
transfer backend handles outside NCCL and the profiler therefore never observes.

A profile spanning several machines whose pools each sit on one machine will report more hosts
and no new NCCL behaviour. Use wide expert parallelism, or a tensor-parallel group that straddles
machines, when cross-node traffic is the point of the tier.

## Externally deployed clusters

A profile with `external: true` names no deployment configuration; the serving cluster and the
observability stack are expected to be running already.

!!! warning
    Every machine in a multi-machine tier must reach the same collector. The address used by the
    single-machine configuration is the Docker bridge gateway, which is reachable only from the
    host it runs on. A profile that leaves it unchanged reports no telemetry at all, which looks
    exactly like a broken profiler. Confirm `otel_endpoint` is reachable from every machine
    before reading anything into a failing run.

## Querying profiles from other tooling

If your environment drives its own deployment, these avoid re-implementing profile parsing and
path resolution:

```bash title="Query the profiles on a search path"
python -m profiler_otel.profiles list [--profile-dir DIR ...]
python -m profiler_otel.profiles compose-file NAME [--profile-dir DIR ...]
```

`compose-file` prints an absolute path, or nothing at all with a zero exit status when the
profile is externally deployed — so empty output means there is nothing to bring up. An unknown
profile exits non-zero and writes to standard error, so it cannot be mistaken for that case.

## Validating profiles

Profiles are parsed into pydantic models, so a malformed one fails when it loads rather than
part-way through a run. Validation needs no hardware:

```bash title="Validate a directory of profiles"
uv run pytest profiler_otel/test_profiles.py --profile-dir /path/to/your/profiles
```

Errors name the profile and the field path, for example
`coverage.ranks: 99 exceeds the 16 GPUs the hardware provides`. What is checked:

- **Unknown keys are rejected.** A misspelled key would otherwise be ignored and leave the
  setting it was meant to change at its default.
- **Types and ranges.** Counts and timeouts must be positive; a port must be a valid port.
- **`coverage` must be satisfiable** by the machines and GPUs in `hardware`.
- **`deployment` names exactly one source**, `compose_file` or `external`.
- **`endpoint` is one form or the other**, a host and port or a `base_url`.
- **Worker pools match the serving mode.** A disaggregated profile declares both `prefill` and
  `decode`; an aggregated one declares neither.
- **`expected_metrics` names a list that exists.**
- **The compose path resolves** to a file that is really there, relative to the profile.

Running this against a directory of profiles kept elsewhere gives that directory the same
checks, in a repository whose own tests may never run this suite.

`tests/suites/profiler_otel/testdata/profiles/` holds synthetic profiles used only by those
tests. They cover shapes this repository supports but no longer ships a profile for, such as
external deployments and multi-machine coverage. They are deliberately not on the default search
path and cannot be selected by an ordinary run.
