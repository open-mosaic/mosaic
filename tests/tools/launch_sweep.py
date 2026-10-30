#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
Run the profiler OTEL suite once per row of a sweep table, then chart the results.

Reads ``sweep.config``, rewrites the named keys in the profile before each run, and writes that
row's report to its own file. Aborts on the first row that fails, and restores the profile on
any exit so an interrupted sweep never leaves a modified file behind.

This is a tool in ``tests/tools``; the suite it drives, the profiles it rewrites and the sweep
table it reads are in ``tests/suites``, and pytest is run from there. Paths are resolved from
this file's own location, so it can be run from any working directory.

Usage::

    ./launch_sweep.py --list                 # the table and each row's report filename
    ./launch_sweep.py --dry-run              # print each row's plan, change nothing
    ./launch_sweep.py                        # run every row
    ./launch_sweep.py --start 6              # resume at row 6 (1-based, as --list numbers)
    ./launch_sweep.py --only 6               # one row
    ./launch_sweep.py --keep-going           # skip past a failed row
    ./launch_sweep.py --graph                # run, then chart
    ./launch_sweep.py --graph --out-dir DIR  # chart an existing sweep, run nothing

Graphing needs matplotlib, which the suite does not depend on. Either::

    uv run --with matplotlib ./launch_sweep.py --graph --out-dir /tmp/cai_sweep_...

or install it once into the environment. The script says so if the import fails.

Results are always written to ``sweep_results.csv`` in the output directory, whether or not
graphs are produced -- that file is the table view of every chart, and the thing to hand to a
spreadsheet or a notebook.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent

#: The suite this script drives. It is not the script's own directory: this is a tool in
#: `tests/tools`, beside its own sweep table, while the profiles it rewrites and the pytest
#: project it runs live in `tests/suites`. Resolved from this file's location so the pair can
#: be checked out anywhere, and verified below rather than left to surface as a puzzling
#: pytest error three steps later.
SUITE_DIR = SCRIPT_DIR.parent / "suites"

#: Consecutive failures that end a --keep-going sweep.
CONSECUTIVE_FAILURE_LIMIT = 2

ABBREV = {
    "max_concurrency": "c",
    "num_prompts": "p",
    "num_warmups": "w",
    "random_input_len": "in",
    "random_output_len": "out",
    "timeouts.workload": "t",
    "tensor_parallel": "tp",
}

RESULT_MEASURES = {
    "requests completed": "requests_completed",
    "benchmark duration (timed requests)": "benchmark_duration_s",
    "tokens in (prompt)": "tokens_in",
    "tokens out (generated)": "tokens_out",
    "throughput (requests)": "req_per_s",
    "throughput (prompt+generated)": "total_tok_per_s",
    "throughput (generated only)": "output_tok_per_s",
    "TTFT mean": "ttft_mean_ms",
    "TTFT p99": "ttft_p99_ms",
    "TPOT mean": "tpot_mean_ms",
    "TPOT p99": "tpot_p99_ms",
    "container wall time (incl. startup/teardown)": "wall_time_s",
}


# =============================================================================
# The sweep table
# =============================================================================


@dataclass(frozen=True)
class SweepRow:
    """One row of the table: the profile keys to set, in the order the header names them."""

    index: int
    values: dict[str, str]

    def report_stem(self, label: str) -> str:
        """Filename stem for this row's report, built from its own values."""
        if "name" in self.values:
            return f"{label}_{self.values['name']}"
        parts = "".join(f"_{ABBREV.get(key, key)}{value}" for key, value in self.values.items())
        return f"{label}{parts}"


def parse_config(path: Path) -> tuple[list[str], list[SweepRow]]:
    """
    Read the sweep table: a header naming profile keys, then one row of values per run.

    Raises SystemExit with a specific message on any malformed input.
    """
    header: list[str] = []
    rows: list[SweepRow] = []
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if not header:
            header = fields
            continue
        if len(fields) != len(header):
            sys.exit(f"{path}:{number}: {len(fields)} value(s) for {len(header)} column(s): {line}")
        rows.append(SweepRow(index=len(rows) + 1, values=dict(zip(header, fields))))

    if not header:
        sys.exit(f"{path}: no header line")
    if not rows:
        sys.exit(f"{path}: header but no rows")
    return header, rows


# =============================================================================
# Profile editing
# =============================================================================


def read_profile_value(profile: Path, key: str) -> str | None:
    """The value at *key* (dotted paths allowed), or None when the profile has no such key."""
    lines = profile.read_text().splitlines()
    try:
        index = find_key_line(lines, key)
    except SystemExit:
        return None
    return lines[index].split(":", 1)[1].strip().strip("\"'") or None


def safe_label(text: str) -> str:
    """
    Reduce *text* to something usable as one filename component.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or "sweep"


def model_label(model: str) -> str:
    """
    Filename-safe tag for a model id: ``openai/gpt-oss-120b`` -> ``gpt_oss_120b``.
    """
    return safe_label(re.sub(r"[^a-z0-9]+", "_", model.rsplit("/", 1)[-1].lower()))


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def find_key_line(lines: list[str], key: str) -> int:
    """
    Index of the line holding *key*, which may be a dotted path like ``timeouts.workload``.
    """
    parts = key.split(".")
    start, stop, depth = 0, len(lines), -1

    for position, part in enumerate(parts):
        leaf = position == len(parts) - 1
        # A parent is "name:" with nothing after it; a leaf is "name: value".
        pattern = re.compile(rf"^[ \t]*{re.escape(part)}:[ \t]*{'\\S' if leaf else '$'}")
        hits = [
            index for index in range(start, stop) if pattern.match(lines[index]) and _indent_of(lines[index]) > depth
        ]
        if len(hits) != 1:
            where = ".".join(parts[: position + 1])
            sys.exit(f"expected exactly one '{where}' in the profile, found {len(hits)}")

        index = hits[0]
        if leaf:
            return index
        # Narrow to this parent's block: everything more indented, up to the next dedent.
        depth = _indent_of(lines[index])
        start = index + 1
        stop = next(
            (
                i
                for i in range(start, len(lines))
                if lines[i].strip() and not lines[i].lstrip().startswith("#") and _indent_of(lines[i]) <= depth
            ),
            len(lines),
        )
    raise AssertionError("unreachable")


def patch_profile(profile: Path, key: str, value: str) -> None:
    """
    Rewrite one ``key: value`` line in place. *key* may be dotted (``timeouts.workload``).
    """
    lines = profile.read_text().splitlines(keepends=True)
    index = find_key_line(lines, key)
    leaf = key.rsplit(".", 1)[-1]
    ending = "\n" if lines[index].endswith("\n") else ""
    lines[index] = f"{' ' * _indent_of(lines[index])}{leaf}: {value}{ending}"
    profile.write_text("".join(lines))


# =============================================================================
# The run environment
# =============================================================================

#: Grafana's own default, for a profile that does not name a port. Mirrors the default on
#: `profiler_otel.profiles.Endpoint.grafana_port`, which is where the value belongs: this
#: script reads profile keys as text so that it runs under a bare interpreter, and so cannot
#: import the schema to ask. Keep the two in step.
DEFAULT_GRAFANA_PORT = "3000"


def resolve_run_env(profile: Path, args: argparse.Namespace) -> dict[str, str]:
    """
    The addresses a run needs, from the command line or the profile that describes the cluster.

    Nothing here is defaulted to a particular cluster. A hard-coded host is right exactly once,
    on the machine it was written for, and wrong silently everywhere else: a sweep pointed at a
    second cluster would go on reading the first one's metrics and report its numbers as this
    one's. So the command line wins, the profile's ``endpoint`` block answers otherwise, and a
    run that can name neither stops before it starts.

    ``GPU_INFO_SSH_USER`` is the exception. Without it the report states why the head node's
    driver and CUDA rows are missing and is otherwise complete, which is not worth ending a
    sweep over.
    """
    endpoint_host = read_profile_value(profile, "endpoint.host")
    resolved = {
        "PROMETHEUS_HOST": args.prometheus_host
        or read_profile_value(profile, "endpoint.prometheus_host")
        or endpoint_host,
        "GRAFANA_HOST": args.grafana_host
        or read_profile_value(profile, "endpoint.grafana_host")
        or endpoint_host,
        "GRAFANA_PORT": args.grafana_port
        or read_profile_value(profile, "endpoint.grafana_port")
        or DEFAULT_GRAFANA_PORT,
    }
    for name, value in resolved.items():
        if not value:
            option = "--" + name.lower().replace("_", "-")
            sys.exit(
                f"{name} is not set: pass {option}, export {name}, or give profile "
                f"{profile.stem} an endpoint.{name.lower()} -- or an endpoint.host for the "
                "Prometheus and Grafana addresses to fall back to"
            )

    ssh_user = args.gpu_info_ssh_user or read_profile_value(profile, "endpoint.gpu_info_ssh_user")
    if ssh_user:
        resolved["GPU_INFO_SSH_USER"] = ssh_user
    return resolved


# =============================================================================
# Reading results back out of a report
# =============================================================================

_DURATION_UNITS = {"hr": 3600, "hrs": 3600, "min": 60, "mins": 60, "sec": 1, "secs": 1}


def to_number(text: str) -> float | None:
    """
    Coerce a report cell back to a number, or None when it carries no measurement.

    The report formats for people -- thousands separators, and durations as "19 mins 23 secs" --
    so reading it back means undoing that.
    """
    cell = text.strip()
    if not cell or cell in {"-", "absent"}:
        return None

    if match := re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+)", cell):
        if all(unit in _DURATION_UNITS for _, unit in match):
            return sum(float(amount) * _DURATION_UNITS[unit] for amount, unit in match)

    try:
        return float(cell.replace(",", ""))
    except ValueError:
        return None


def _table_rows(report: Path) -> list[list[str]]:
    """Every table row in *report*, as lists of cell strings, for either output format."""
    text = report.read_text()
    if report.suffix.lower() in {".html", ".htm"}:
        rows = []
        for block in re.findall(r"<tr>(.*?)</tr>", text, re.DOTALL):
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", block, re.DOTALL)
            rows.append([re.sub(r"<[^>]+>", "", cell).strip() for cell in cells])
        return rows

    return [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if line.lstrip().startswith("|")
    ]


def parse_report(report: Path) -> dict[str, float]:
    """
    Pull the InferenceX measurements out of a report.
    """
    found: dict[str, float] = {}
    for cells in _table_rows(report):
        if len(cells) < 2:
            continue
        column = RESULT_MEASURES.get(cells[0])
        if column is None or column in found:
            continue
        value = to_number(cells[1])
        if value is not None:
            found[column] = value
    return found


# =============================================================================
# Graphs
# =============================================================================

# Hue colors used in the charts
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"


def _style_axes(ax, title: str, xlabel: str, ylabel: str, log_x: bool, xticks=None, log_y: bool = False) -> None:
    """Recessive grid and axes, explicit surface, no top/right spines."""
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    if log_x:
        ax.set_xscale("log", base=2)
        if xticks:
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(int(t)) for t in xticks])
    if log_y:
        ax.set_yscale("log")


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    return path


def make_graphs(rows: list[dict[str, str]], out_dir: Path) -> list[Path]:
    """
    Chart the sweep. Returns the files written.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit(
            "matplotlib is required for --graph and is not installed. Re-run as:\n"
            "  uv run --with matplotlib ./launch_sweep.py --graph "
            f"--out-dir {out_dir}"
        )

    def series(*columns: str) -> list[list[float]]:
        """Rows having every requested column, as parallel float lists."""
        usable = [r for r in rows if all(r.get(c) not in (None, "") for c in columns)]
        usable.sort(key=lambda r: float(r[columns[0]]))
        return [[float(r[c]) for r in usable] for c in columns]

    written: list[Path] = []
    concurrencies = sorted({float(r["max_concurrency"]) for r in rows if r.get("max_concurrency")})

    def line_chart(name, title, xcol, xlabel, ylabel, plots, log_x=True, log_y=False):
        """*plots* is a list of (column, label, color); one or two entries."""
        columns = [xcol] + [column for column, _, _ in plots]
        data = series(*columns)
        if not data or not data[0]:
            return
        fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
        for values, (_, label, color) in zip(data[1:], plots):
            ax.plot(
                data[0],
                values,
                color=color,
                linewidth=2,
                marker="o",
                markersize=7,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=label,
            )
        _style_axes(ax, title, xlabel, ylabel, log_x, concurrencies if log_x else None, log_y)
        # A legend is present whenever there are two series; a single series is named by the
        # title, so a one-entry legend box would be noise.
        if len(plots) > 1:
            legend = ax.legend(frameon=False, labelcolor=INK, fontsize=9)
            legend.set_title(None)
        written.append(_save(fig, out_dir / name))
        plt.close(fig)

    line_chart(
        "throughput_generated_vs_concurrency.png",
        "Generated-token throughput vs client concurrency",
        "max_concurrency",
        "max_concurrency (requests in flight, log2)",
        "generated tokens / second",
        [("output_tok_per_s", "generated only", SERIES_1)],
    )
    line_chart(
        "throughput_total_vs_concurrency.png",
        "Total-token throughput (prompt + generated) vs client concurrency",
        "max_concurrency",
        "max_concurrency (log2)",
        "total tokens / second",
        [("total_tok_per_s", "prompt + generated", SERIES_2)],
    )
    line_chart(
        "throughput_generated_vs_num_prompts.png",
        "Generated-token throughput vs prompts per run",
        "num_prompts",
        "num_prompts (log2)",
        "generated tokens / second",
        [("output_tok_per_s", "generated only", SERIES_1)],
    )
    line_chart(
        "throughput_total_vs_num_prompts.png",
        "Total-token throughput (prompt + generated) vs prompts per run",
        "num_prompts",
        "num_prompts (log2)",
        "total tokens / second",
        [("total_tok_per_s", "prompt + generated", SERIES_2)],
    )
    line_chart(
        "ttft_vs_concurrency.png",
        "Time to first token vs client concurrency",
        "max_concurrency",
        "max_concurrency (log2)",
        "TTFT (ms, log)",
        [("ttft_mean_ms", "mean", SERIES_1), ("ttft_p99_ms", "p99", SERIES_2)],
        # TTFT runs from tens of milliseconds to tens of seconds across the sweep; on a linear
        # axis every row below saturation sits flat on the baseline.
        log_y=True,
    )
    line_chart(
        "tpot_vs_concurrency.png",
        "Time per output token vs client concurrency",
        "max_concurrency",
        "max_concurrency (log2)",
        "TPOT (ms)",
        [("tpot_mean_ms", "mean", SERIES_1), ("tpot_p99_ms", "p99", SERIES_2)],
    )
    line_chart(
        "request_throughput_vs_concurrency.png",
        "Request throughput vs client concurrency",
        "max_concurrency",
        "max_concurrency (log2)",
        "requests / second",
        [("req_per_s", "requests/s", SERIES_1)],
    )

    per_request = []
    for row in rows:
        concurrency, throughput = row.get("max_concurrency"), row.get("output_tok_per_s")
        if concurrency and throughput:
            per_request.append((float(concurrency), float(throughput) / float(concurrency)))
    if per_request:
        per_request.sort()
        fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
        ax.plot(
            [c for c, _ in per_request],
            [v for _, v in per_request],
            color=SERIES_1,
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        _style_axes(
            ax,
            "Scaling efficiency: generated tokens/s per request in flight",
            "max_concurrency (log2)",
            "tokens / second / request",
            True,
            concurrencies,
        )
        written.append(_save(fig, out_dir / "scaling_efficiency.png"))
        plt.close(fig)

    pareto = [
        (float(r["ttft_p99_ms"]), float(r["output_tok_per_s"]), r["max_concurrency"])
        for r in rows
        if r.get("ttft_p99_ms") and r.get("output_tok_per_s") and r.get("max_concurrency")
    ]
    if pareto:
        pareto.sort(key=lambda item: float(item[2]))
        fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
        ax.plot(
            [p[0] for p in pareto],
            [p[1] for p in pareto],
            color=SERIES_1,
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            alpha=0.95,
            zorder=3,
        )
        for ttft, throughput, concurrency in pareto:
            ax.annotate(
                f"c={concurrency}",
                (ttft, throughput),
                textcoords="offset points",
                xytext=(7, -3),
                color=INK_MUTED,
                fontsize=8,
            )
        # Log on both axes: throughput and tail latency each span two orders of magnitude
        # across the sweep, and on linear axes the low-concurrency rows collapse into one
        # blob with their labels overprinting each other.
        _style_axes(
            ax,
            "Throughput vs tail latency (each point is one sweep row)",
            "TTFT p99 (ms, log)",
            "generated tokens / second (log)",
            False,
            log_y=True,
        )
        ax.set_xscale("log")
        written.append(_save(fig, out_dir / "throughput_vs_ttft_p99.png"))
        plt.close(fig)

    return written


# =============================================================================
# Running
# =============================================================================


def load_results_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_results_csv(path: Path, header: list[str], records: list[dict[str, str]]) -> None:
    """One row per run: its configuration, then its measurements."""
    columns = header + ["report", "status"] + list(RESULT_MEASURES.values())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(
        description="Run the profiler OTEL suite once per row of a sweep table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "sweep.config")
    parser.add_argument(
        "--profile", default=os.environ.get("PROFILE", "cai_4n"), help="profile name for --workload-profile"
    )
    parser.add_argument(
        "--label",
        default=os.environ.get("LABEL"),
        help="tag at the front of each report filename (default: derived from the model being served)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="reports, logs and CSV go here (default /tmp/cai_sweep_<stamp>)"
    )
    parser.add_argument("--model", default=os.environ.get("MODEL"), help="written to serving.model before every run")
    parser.add_argument("--pytest-k", default=os.environ.get("PYTEST_K", "not nccl_workload"))
    parser.add_argument("--report-ext", default=os.environ.get("REPORT_EXT", "html"), choices=["html", "md"])
    parser.add_argument(
        "--prometheus-host",
        default=os.environ.get("PROMETHEUS_HOST"),
        help="where NCCL metrics are read from (default: the profile's endpoint.prometheus_host, else endpoint.host)",
    )
    parser.add_argument(
        "--grafana-host",
        default=os.environ.get("GRAFANA_HOST"),
        help="Grafana for the dashboards checks (default: the profile's endpoint.grafana_host, else endpoint.host)",
    )
    parser.add_argument(
        "--grafana-port",
        default=os.environ.get("GRAFANA_PORT"),
        help=f"Grafana port (default: the profile's endpoint.grafana_port, else {DEFAULT_GRAFANA_PORT})",
    )
    parser.add_argument(
        "--gpu-info-ssh-user",
        default=os.environ.get("GPU_INFO_SSH_USER"),
        help="SSH login for the head-node GPU probe (default: the profile's endpoint.gpu_info_ssh_user). "
        "Without one, reports omit the driver and CUDA versions",
    )
    parser.add_argument("--start", type=int, default=1, help="resume at this row (1-based)")
    parser.add_argument("--only", type=int, default=None, help="run just this row")
    parser.add_argument("--list", action="store_true", help="show the table and exit")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help=f"carry on after a failed row, stopping only after {CONSECUTIVE_FAILURE_LIMIT} failures in a row",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="chart the results; with --out-dir and no rows to run, charts an existing sweep without running anything",
    )
    args = parser.parse_args()

    if not (SUITE_DIR / "pyproject.toml").is_file():
        sys.exit(
            f"the suite is not where this script expects it: {SUITE_DIR} has no "
            "pyproject.toml. This script belongs in tests/tools, beside tests/suites."
        )

    profile_path = SUITE_DIR / "profiler_otel" / "profiles" / f"{args.profile}.yaml"
    if not profile_path.is_file():
        sys.exit(f"profile not found: {profile_path}")
    if not args.config.is_file():
        sys.exit(f"sweep config not found: {args.config}")

    # Default the label to the model actually being served, so report names can never
    # describe a previous model. --model wins when set, since that is what the run will use.
    if args.label:
        cleaned = safe_label(args.label)
        if cleaned != args.label:
            _log.warning(f"  note: label {args.label!r} is not filename-safe; using {cleaned!r}")
        args.label = cleaned
    else:
        model = args.model or read_profile_value(profile_path, "model") or ""
        args.label = model_label(model) if model else "sweep"

    header, rows = parse_config(args.config)

    if args.list:
        widths = [max(len(key), 12) for key in header]
        _log.info("#    " + "".join(key.ljust(w + 2) for key, w in zip(header, widths)) + "report")
        for row in rows:
            values = "".join(row.values[key].ljust(w + 2) for key, w in zip(header, widths))
            _log.info(f"{row.index:<5}{values}{row.report_stem(args.label)}.{args.report_ext}")
        return 0

    # Charting an existing sweep: --graph with an --out-dir that already has a CSV, and no
    # intention of running anything.
    out_dir = args.out_dir or Path(f"/tmp/cai_sweep_{time.strftime('%Y%m%d-%H%M%S')}")
    csv_path = out_dir / "sweep_results.csv"
    if args.graph and args.out_dir and csv_path.is_file() and args.only is None and args.start == 1:
        written = make_graphs(load_results_csv(csv_path), out_dir)
        _log.info(f"Wrote {len(written)} chart(s) to {out_dir}")
        for path in written:
            _log.info(f"  {path.name}")
        return 0

    selected = [r for r in rows if (r.index == args.only if args.only else r.index >= args.start)]
    if not selected:
        sys.exit("no rows selected")

    # Before the banner, so a run that cannot name its Prometheus says so instead of printing
    # a plan it will not carry out. --dry-run gets the same check for the same reason.
    run_env = resolve_run_env(profile_path, args)

    out_dir.mkdir(parents=True, exist_ok=True)
    _log.info(f"Sweep of {len(selected)} of {len(rows)} row(s) from {args.config}")
    _log.info(f"  profile : {profile_path}")
    _log.info(f"  reports : {out_dir}")
    _log.info(f"  model   : {args.model or '<unchanged in profile>'}")
    _log.info(f"  columns : {' '.join(header)}")
    grafana = f"{run_env['GRAFANA_HOST']}:{run_env['GRAFANA_PORT']}"
    _log.info(f"  metrics : Prometheus {run_env['PROMETHEUS_HOST']}, Grafana {grafana}")
    if "GPU_INFO_SSH_USER" in run_env:
        _log.info(f"  gpu ssh : {run_env['GPU_INFO_SSH_USER']}")
    else:
        _log.info("  gpu ssh : none -- reports will omit the head node's driver and CUDA versions")
    if args.dry_run:
        _log.info("  DRY RUN -- no profile edits, no test runs")
    _log.info("")

    # Exported so both pytest and the report's head-node GPU probe see them.
    env = dict(os.environ)
    env.update(run_env)

    records: list[dict[str, str]] = []
    handle_fd, backup_name = tempfile.mkstemp(prefix=f"{args.profile}.orig.")
    os.close(handle_fd)
    backup = Path(backup_name)
    shutil.copy(profile_path, backup)
    sweep_started = time.time()
    failed_row: SweepRow | None = None
    failures: list[SweepRow] = []
    consecutive = 0
    exit_code = 0

    try:
        for row in selected:
            stem = row.report_stem(args.label)
            report = out_dir / f"{stem}.{args.report_ext}"
            log = out_dir / f"{stem}.log"
            described = " ".join(f"{k}={v}" for k, v in row.values.items())
            _log.info(f"[{row.index}/{len(rows)}] {described}")
            _log.info(f"         report: {report}")

            if args.dry_run:
                _log.info("         (dry run) would rewrite those keys and run pytest\n")
                continue

            for key, value in row.values.items():
                if key != "name":
                    patch_profile(profile_path, key, value)
            if args.model:
                patch_profile(profile_path, "model", args.model)

            started = time.time()
            command = [
                "uv",
                "run",
                "pytest",
                "-v",
                "--workload-profile",
                args.profile,
                "-k",
                args.pytest_k,
                f"--report-file={report}",
            ]
            with log.open("w") as handle:
                process = subprocess.run(
                    command,
                    cwd=SUITE_DIR,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                handle.write(process.stdout)
            _log.info(process.stdout.rstrip("\n"))

            elapsed = time.time() - started
            record = dict(row.values)
            record["report"] = report.name
            record["status"] = "passed" if process.returncode == 0 else "failed"
            if report.is_file():
                record.update({k: str(v) for k, v in parse_report(report).items()})
            records.append(record)

            if process.returncode != 0:
                consecutive += 1
                failures.append(row)
                exit_code = process.returncode
                _log.error(
                    f"         FAILED (pytest exit {process.returncode}) after {elapsed / 60:.0f}m {elapsed % 60:.0f}s"
                )
                if not args.keep_going:
                    _log.info("")
                    failed_row = row
                    break
                # Two in a row means the deployment is down rather than one row being too
                # heavy for it, and every remaining row would fail the same way -- on a
                # 12-hour sweep that is hours of wasted cluster time.
                if consecutive >= CONSECUTIVE_FAILURE_LIMIT:
                    _log.error(f"         {consecutive} failures in a row -- stopping\n")
                    failed_row = row
                    break
                _log.info("         continuing (--keep-going)\n")
                continue

            consecutive = 0
            _log.info(f"         PASSED in {elapsed / 60:.0f}m {elapsed % 60:.0f}s\n")
    finally:
        shutil.copy(backup, profile_path)
        backup.unlink(missing_ok=True)
        _log.info(f"  Restored {args.profile}.yaml to its original contents.")

    if records:
        write_results_csv(csv_path, header, records)
        _log.info(f"  Results: {csv_path}")

    if failures:
        _log.error(f"\n{len(failures)} row(s) failed:")
        for row in failures:
            _log.error(f"  row {row.index}: {out_dir / (row.report_stem(args.label) + '.log')}")

    if failed_row is not None:
        _log.error(f"\nStopped at row {failed_row.index}/{len(rows)}.")
        _log.info(f"  resume with : {sys.argv[0]} --start {failed_row.index} --out-dir {out_dir}")
        return exit_code

    if not args.dry_run:
        total = time.time() - sweep_started
        passed = sum(1 for record in records if record.get("status") == "passed")
        _log.info(f"Completed {passed} of {len(records)} row(s) in {total // 3600:.0f}h {(total % 3600) // 60:.0f}m.")
        if args.graph and records:
            written = make_graphs(load_results_csv(csv_path), out_dir)
            _log.info(f"Wrote {len(written)} chart(s):")
            for path in written:
                _log.info(f"  {path.name}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
