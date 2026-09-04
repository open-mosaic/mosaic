# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
Description of the deployment a run targeted, for the report's header tables.
"""

import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from profiler_otel import profiles

__all__ = ["benchmark_option_rows", "environment_rows"]


def _display_path(path: Path) -> str:
    """
    A path safe to put in a shared report: repo-relative, with the private prefix dropped.

    Relative to the enclosing git repository -- prefixed with the repository's own directory 
    name, so the result reads the way anyone with a checkout would recognise it.
    """
    resolved = path.resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / ".git").exists():
            return str(Path(parent.name) / resolved.relative_to(parent))
    return f".../{resolved.parent.name}/{resolved.name}"


def _pool_summary(pool: profiles.WorkerPool) -> str:
    """One line describing a prefill or decode pool, omitting what the profile left unset."""
    parts = []
    if pool.nodes is not None:
        parts.append(f"{pool.nodes} node(s)")
    if pool.workers is not None:
        parts.append(f"{pool.workers} worker(s)")
    parts.append(f"TP={pool.tensor_parallel}")
    if pool.expert_parallel is not None:
        parts.append(f"EP={pool.expert_parallel}")
    parts.append("spans nodes" if pool.spans_nodes else "in-node")
    return ", ".join(parts)


def _command_output(argv: list[str], timeout: float = 5.0) -> tuple[str | None, str]:
    """
    ``(stdout, reason)`` for *argv* -- stdout is None on any failure, and *reason* says why.

    Used only to describe the environment, so no failure mode raises: a missing driver must
    never turn a telemetry result into an error.
    """
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return None, f"{argv[0]} not found on PATH"
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout:g}s"
    except OSError as exc:
        return None, str(exc)

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        return None, detail[0] if detail else f"exit status {proc.returncode}"
    output = proc.stdout.strip()
    return (output, "") if output else (None, "command produced no output")


def _parse_driver_versions(version_text: str) -> dict[str, str]:
    """
    Pull the version fields out of ``nvidia-smi --version`` or a plain ``nvidia-smi`` header.

    Three generations of output are handled, because which one a node prints depends on its
    driver and all three are still in the field: 610+, 5xx, and older builds with no --version.
    """
    raw: dict[str, str] = {}
    for key, pattern in (
        ("nvidia-smi", r"NVIDIA-SMI version\s*:\s*(\S+)"),
        ("NVML", r"NVML version\s*:\s*(\S+)"),
        ("KMD", r"KMD version\s*:\s*(\S+)"),
        ("cuda_umd", r"CUDA UMD version\s*:\s*(\S+)"),
        ("driver_field", r"DRIVER version\s*:\s*(\S+)"),
        ("cuda_field", r"^\s*CUDA version\s*:\s*(\S+)"),
        ("driver_header", r"Driver Version:\s*(\S+)"),
        ("cuda_header", r"CUDA Version:\s*(\S+)"),
    ):
        if match := re.search(pattern, version_text, re.IGNORECASE | re.MULTILINE):
            value = match.group(1)

            if not value.lower().startswith("deprecated"):
                raw[key] = value

    found = {key: raw[key] for key in ("nvidia-smi", "NVML", "KMD") if key in raw}

    if driver := raw.get("driver_field") or raw.get("driver_header") or raw.get("KMD"):
        found["driver"] = driver
    if cuda := raw.get("cuda_umd") or raw.get("cuda_field") or raw.get("cuda_header"):
        found["CUDA"] = cuda
    return found


_GPU_INFO_SCRIPT = (
    "nvidia-smi --version 2>/dev/null || nvidia-smi 2>/dev/null; "
    "echo '===KMD==='; cat /proc/driver/nvidia/version 2>/dev/null; "
    "echo '===GPUS==='; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null"
)


def _head_node_host(profile: profiles.Profile) -> str:
    """
    The host to collect GPU versions from.
    """
    if override := os.getenv("GPU_INFO_HOST"):
        return override
    if profile.endpoint.base_url is not None:
        return urlparse(profile.endpoint.base_url).hostname or ""
    return profile.endpoint.host or ""


def _ssh_target(host: str) -> str:
    """
    ``user@host`` for the probe, or a bare host so ``~/.ssh/config`` decides.
    """
    user = os.getenv("GPU_INFO_SSH_USER") or os.getenv("SSH_USER")
    return f"{user}@{host}" if user else host


def _remote_output(host: str, script: str, timeout: float = 25.0) -> tuple[str | None, str]:
    """
    Run *script* on *host* over SSH, returning ``(stdout, reason)``.
    """
    return _command_output(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "StrictHostKeyChecking=accept-new",
            _ssh_target(host),
            script,
        ],
        timeout=timeout,
    )


def _gpu_environment_rows(profile: profiles.Profile) -> list[list[str]]:
    """
    GPU, CUDA and kernel-module versions from the head node.
    """
    host = _head_node_host(profile)
    if not host:
        return [["GPU driver", "not collected -- no head node host to query (set GPU_INFO_HOST)"]]

    label = f"head node {host}"
    output, reason = _remote_output(host, _GPU_INFO_SCRIPT)
    if output is None:
        detail = reason.rstrip(". ")
        auth_failure = any(word in reason.lower() for word in ("denied", "authentication"))
        hint = " Set GPU_INFO_SSH_USER if the cluster login differs." if auth_failure and not os.getenv(
            "GPU_INFO_SSH_USER"
        ) else ""
        return [["GPU driver", f"not collected -- ssh {_ssh_target(host)}: {detail}.{hint}"]]

    version_text, _, rest = output.partition("===KMD===")
    kmd_text, _, inventory = rest.partition("===GPUS===")

    versions = _parse_driver_versions(version_text)

    if "KMD" not in versions and (match := re.search(r"NVRM version:.*?(\d+\.\d+(?:\.\d+)?)", kmd_text)):
        versions["KMD"] = match.group(1)

    rows: list[list[str]] = []
    for name in ("driver", "CUDA", "KMD", "nvidia-smi", "NVML"):
        if name in versions:
            rows.append([f"{name} version ({label})", versions[name]])
    if "open kernel module" in kmd_text.lower():
        rows.append([f"KMD variant ({label})", "open kernel modules"])

    gpu_lines = [line for line in inventory.splitlines() if "," in line]
    if gpu_lines:
        names = sorted({line.split(",", 2)[1].strip() for line in gpu_lines})
        rows.append([f"GPUs ({label})", f"{len(gpu_lines)}x {', '.join(names)}"])

    if not rows:
        rows.append(["GPU driver", f"not collected -- {host} answered but reported no NVIDIA driver"])
    return rows


def environment_rows(profile: profiles.Profile, prometheus_url: str, grafana_url: str) -> list[list[str]]:
    """
    What this run was pointed at, as (property, value) rows for the report's top table.
    """
    hardware, serving, coverage, timeouts = profile.hardware, profile.serving, profile.coverage, profile.timeouts

    if profile.endpoint.base_url is not None:
        endpoint = profile.endpoint.base_url
    else:
        endpoint = f"{profile.endpoint.host}:{profile.endpoint.port}"
    override = [f"{key}={os.environ[key]}" for key in ("VLLM_HOST", "VLLM_PORT") if key in os.environ]

    rows = [
        ["profile", f"{profile.name}  ({_display_path(profile.path)})"],
        ["description", profile.description],
        ["machines", str(hardware.machines)],
        ["GPUs per machine", str(hardware.gpus_per_machine)],
        ["GPUs total", str(hardware.total_gpus)],
        ["GPU SKU", hardware.sku],
        ["serving mode", str(serving.mode)],
        ["model", serving.model],
    ]
    if serving.tensor_parallel is not None:
        rows.append(["tensor parallel", str(serving.tensor_parallel)])
    if serving.expert_parallel is not None:
        rows.append(["expert parallel", str(serving.expert_parallel)])

    if serving.prefill is not None:
        rows.append(["prefill pool", _pool_summary(serving.prefill)])
    if serving.decode is not None:
        rows.append(["decode pool", _pool_summary(serving.decode)])
    if serving.kv_transfer is not None:
        rows.append(["KV transfer", serving.kv_transfer])
    rows += [
        ["coverage expected", f"{coverage.hosts} host(s), {coverage.gpus} GPU(s), {coverage.communicators} comm(s)"],
        ["expected metrics", str(profile.expected_metrics)],
        ["endpoint", endpoint + (f"  (overridden: {', '.join(override)})" if override else "")],
        [
            "deployment",
            "external" if profile.is_external else f"compose: {_display_path(profile.compose_file)}",
        ],
        ["Prometheus", prometheus_url],
        ["Grafana", grafana_url],
        ["timeouts", f"workload {timeouts.workload}s, metrics {timeouts.metrics_available}s, quiesce {timeouts.quiesce}s"],
        ["test runner", f"{platform.node()}  ({platform.system()} {platform.release()})"],
    ]
    return rows + _gpu_environment_rows(profile)


_BENCHMARK_OPTION_ORDER = (
    "num_prompts",
    "max_concurrency",
    "random_input_len",
    "random_output_len",
    "num_warmups",
    "dataset_name",
    "model",
    "backend",
    "ignore_eos",
)


def benchmark_option_rows(options: dict[str, Any]) -> list[list[str]]:
    """
    Benchmark options as (option, value, flag) rows, load-shaping options first.

    The flag column is not decoration: these keys are converted to ``benchmark_serving.py``
    arguments generically, so showing the conversion is what lets a reader reproduce the run
    by hand.
    """

    def rank(item: tuple[int, str]) -> tuple[int, int]:
        position, key = item
        listed = _BENCHMARK_OPTION_ORDER.index(key) if key in _BENCHMARK_OPTION_ORDER else len(
            _BENCHMARK_OPTION_ORDER
        )
        return listed, position

    rows: list[list[str]] = []
    for _, key in sorted(enumerate(options), key=rank):
        value = options[key]
        flag = "--" + key.replace("_", "-")
        if value is None:
            rows.append([key, "unset", "(omitted)"])
        elif isinstance(value, bool):
            rows.append([key, "true" if value else "false", flag if value else "(omitted)"])
        else:
            rows.append([key, str(value), f"{flag} {value}"])
    return rows

