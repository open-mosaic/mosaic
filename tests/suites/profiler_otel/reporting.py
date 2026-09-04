# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
Combined result report for the mosaic test suites.
"""

import datetime as dt
import math
import html
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

__all__ = [
    "COMPACT_DECIMALS",
    "SIGNIFICANT_DECIMALS",
    "CoverageStatus",
    "MetricStatus",
    "ReportFormat",
    "ReportPlugin",
    "Reporter",
    "Table",
    "format_delta",
    "format_duration",
    "format_number",
    "format_value",
    "render_table",
    "render_table_markdown",
]


class ReportFormat(StrEnum):
    """
    Output format for the report file.

    ``StrEnum`` so the pytest option's string value converts with no lookup table, and
    :meth:`for_path` can infer from a filename when the option is left off.
    """

    HTML = "html"
    MD = "md"

    @classmethod
    def for_path(cls, path: Path) -> "ReportFormat":
        """Infer a format from *path*'s suffix, defaulting to HTML for anything unfamiliar."""
        return cls.MD if path.suffix.lower() in (".md", ".markdown") else cls.HTML


class MetricStatus(StrEnum):
    """
    What happened to one expected metric across a workload.

    ``StrEnum`` so a member drops straight into a table cell and an f-string. The three cases
    need different fixes, which is the whole reason the suite distinguishes them:

    * :attr:`ROSE` -- the total increased; the profiler and the pipeline both work.
    * :attr:`FLAT` -- the series exists and is being scraped, but this workload drove no NCCL
      ops through it. The exporter is alive; either the instrumentation for this metric is not
      recording, or the workload genuinely does not exercise it.
    * :attr:`NO_SERIES` -- Prometheus has no series under this name at all, so nothing was ever
      exported. Check the profiler plugin, the OTLP endpoint and the collector.
    """

    ROSE = "rose"
    FLAT = "flat"
    NO_SERIES = "no series"

    @property
    def is_failure(self) -> bool:
        return self is not MetricStatus.ROSE

    @property
    def remedy(self) -> str:
        """One line on what to look at, shown beside a metric that did not increase."""
        match self:
            case MetricStatus.ROSE:
                return ""
            case MetricStatus.FLAT:
                return "scraped but did not move -- workload drove no ops through it"
            case MetricStatus.NO_SERIES:
                return "never exported -- check the profiler plugin, OTLP endpoint and collector"

    @classmethod
    def for_metric(cls, rose: bool, current: float | None) -> "MetricStatus":
        """Classify one metric from whether it rose and whether it has a series at all."""
        if rose:
            return cls.ROSE
        return cls.NO_SERIES if current is None else cls.FLAT


class CoverageStatus(StrEnum):
    """Whether a participant count met the profile's declared coverage."""

    OK = "ok"
    SHORT = "short"

    @property
    def is_failure(self) -> bool:
        return self is CoverageStatus.SHORT

    @classmethod
    def for_counts(cls, seen: int, expected: int) -> "CoverageStatus":
        return cls.OK if seen >= expected else cls.SHORT


_STATUS_CLASS = {
    MetricStatus.ROSE.value: "ok",
    MetricStatus.FLAT.value: "warn",
    MetricStatus.NO_SERIES.value: "bad",
    CoverageStatus.OK.value: "ok",
    CoverageStatus.SHORT.value: "bad",
    "passed": "ok",
    "failed": "bad",
    "error": "bad",
    "skipped": "warn",
    "xfailed": "warn",
    "xpassed": "warn",
}


SIGNIFICANT_DECIMALS = 4

COMPACT_DECIMALS = 2


def _decimal_places(value: float, significant: int | None = None) -> int:
    """
    How many decimals to show for *value*: as many as it has, capped at *significant* ones.
    """
    if significant is None:
        significant = SIGNIFICANT_DECIMALS

    text = repr(float(value))
    if "e" in text or "E" in text:
        # repr went exponential, which happens below ~1e-5. Derive the leading-zero count from
        # the exponent instead of parsing a mantissa.
        exponent = math.floor(math.log10(abs(value)))
        return max(0, -exponent - 1) + significant

    fraction = text.partition(".")[2].rstrip("0")
    if len(fraction) <= COMPACT_DECIMALS:
        return len(fraction)
    leading_zeros = len(fraction) - len(fraction.lstrip("0"))
    return min(len(fraction), leading_zeros + significant)


def _format_decimal(value: float, significant: int | None = None) -> str:
    """
    Grouped decimal notation, never scientific.
    """
    if not math.isfinite(value):
        return str(value)
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.{_decimal_places(value, significant)}f}"


def format_value(value: float | None, significant: int | None = None) -> str:
    """Render a metric total, distinguishing an absent series from a zero one."""
    return "absent" if value is None else _format_decimal(value, significant)


def format_number(value: float | None, spec: str | None = None, significant: int | None = None) -> str:
    """
    Render a benchmark number, or "-" when the run did not report it.
    """
    if value is None:
        return "-"
    return f"{value:{spec}}" if spec else _format_decimal(value, significant)


def format_duration(seconds: float | None) -> str:
    """
    Converts seconds as ``1 hr 12 mins 3 secs`` format, or "-" when the run did not report it.
    """
    if seconds is None:
        return "-"
    if not math.isfinite(seconds):
        return str(seconds)
    if abs(seconds) < 60:
        return f"{_format_decimal(seconds)} {'sec' if seconds == 1 else 'secs'}"

    whole = round(seconds)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    for amount, unit in ((hours, "hr"), (minutes, "min"), (secs, "sec")):
        if amount:
            parts.append(f"{amount} {unit}" + ("" if amount == 1 else "s"))
    return " ".join(parts)


def format_delta(baseline: float | None, current: float | None) -> str:
    """
    Percentage change from *baseline* to *current*.
    """
    if baseline is None or current is None:
        return "-"
    if baseline == 0:
        return "new" if current > 0 else "+0.00%"
    return f"{(current - baseline) / baseline * 100:+.2f}%"


@dataclass
class Table:
    """
    One table, sink-agnostic.

    *left* names the columns to left-align on top of column 0, which always is: right-aligned
    prose reads as ragged, while right-aligned numbers line up so a value orders of magnitude
    from its neighbours is obvious. *status_column* marks the column the HTML sink colours.
    """

    headers: list[str]
    rows: list[list[str]]
    title: str = ""
    left: set[int] = field(default_factory=set)
    status_column: int | None = None


def render_table(table: Table, indent: str = "  ") -> str:
    """Render *table* as fixed-width text, returned as one string."""
    if not table.rows:
        return f"{indent}(no rows)"

    left_aligned = {0} | table.left
    widths = [max(len(row[i]) for row in (table.headers, *table.rows)) for i in range(len(table.headers))]

    def line(cells: list[str]) -> str:
        rendered = (
            cell.ljust(widths[i]) if i in left_aligned else cell.rjust(widths[i]) for i, cell in enumerate(cells)
        )
        return (indent + "  ".join(rendered)).rstrip()

    separator = ["-" * width for width in widths]
    return "\n".join([line(table.headers), line(separator), *(line(row) for row in table.rows)])


def _render_table_html(table: Table) -> str:
    """Render *table* as an HTML table, escaping every cell."""
    if not table.rows:
        return "<p class='empty'>(no rows)</p>"

    left_aligned = {0} | table.left

    def cell(tag: str, index: int, text: str) -> str:
        classes = [] if index in left_aligned else ["num"]
        if tag == "td" and index == table.status_column:
            classes.append(_STATUS_CLASS.get(text, ""))
        present = [c for c in classes if c]
        attr = f" class='{' '.join(present)}'" if present else ""
        return f"<{tag}{attr}>{html.escape(text)}</{tag}>"

    head = "".join(cell("th", i, h) for i, h in enumerate(table.headers))
    body = "".join(
        "<tr>" + "".join(cell("td", i, value) for i, value in enumerate(row)) + "</tr>" for row in table.rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


_CSS = """
:root { color-scheme: light dark;
  --fg:#1c1c1e; --bg:#fbfbfd; --muted:#6b6b70; --rule:#d8d8dd; --panel:#fff;
  --ok:#1a7f47; --ok-bg:#e7f6ec; --warn:#8a5a00; --warn-bg:#fdf3e0; --bad:#b3261e; --bad-bg:#fdeceb; }
@media (prefers-color-scheme: dark) { :root {
  --fg:#e8e8ea; --bg:#16161a; --muted:#9a9aa2; --rule:#33333a; --panel:#1e1e24;
  --ok:#5cd68f; --ok-bg:#122a1c; --warn:#e0b25c; --warn-bg:#2b2213; --bad:#ff8a80; --bad-bg:#2e1614; } }
body { margin:0; padding:2rem 1.5rem; background:var(--bg); color:var(--fg);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
main { max-width:1150px; margin:0 auto; }
h1 { font-size:1.45rem; margin:0 0 .25rem; }
h2 { font-size:1.15rem; margin:2.5rem 0 .75rem; padding-bottom:.35rem; border-bottom:2px solid var(--rule); }
h3 { font-size:1rem; margin:1.75rem 0 .5rem; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
h4 { font-size:.8rem; font-weight:600; color:var(--muted); margin:1.25rem 0 .4rem;
  text-transform:uppercase; letter-spacing:.05em; }
.meta { color:var(--muted); margin:0 0 .5rem; }
.tally { margin:0 0 1.5rem; display:flex; gap:.5rem; flex-wrap:wrap; }
.tally span { padding:.15rem .55rem; border-radius:999px; font-weight:600; font-size:.82rem;
  border:1px solid var(--rule); }
.tally .ok { color:var(--ok); background:var(--ok-bg); }
.tally .bad { color:var(--bad); background:var(--bad-bg); }
.tally .warn { color:var(--warn); background:var(--warn-bg); }
.note { margin:.5rem 0; }
.empty { color:var(--muted); font-style:italic; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; background:var(--panel); }
th, td { padding:.4rem .6rem; border-bottom:1px solid var(--rule); text-align:left; white-space:nowrap; }
th { font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); font-weight:600; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
td.ok { color:var(--ok); background:var(--ok-bg); font-weight:600; }
td.warn { color:var(--warn); background:var(--warn-bg); font-weight:600; }
td.bad { color:var(--bad); background:var(--bad-bg); font-weight:600; }
tbody tr:last-child td { border-bottom:none; }
pre.failure { background:var(--bad-bg); color:var(--fg); border-left:3px solid var(--bad);
  padding:.75rem 1rem; overflow-x:auto; font-size:12px; line-height:1.45; margin:.5rem 0 0; }
a.top { color:var(--muted); font-size:.8rem; text-decoration:none; }
"""




def _md_escape(text: str) -> str:
    """Escape the one character that would break a Markdown table cell."""
    return text.replace("|", r"\|")


def render_table_markdown(table: Table) -> str:
    """
    Render *table* as a Markdown table with every column left-aligned and padded.
    """
    if not table.rows:
        return "_(no rows)_"

    cells = [[_md_escape(c) for c in table.headers], *[[_md_escape(c) for c in row] for row in table.rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(table.headers))]

    def line(row: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"

    # ":---" marks left alignment; the dashes fill the rest of the column so the rule is as
    # wide as the data above and below it.
    rule = [":" + "-" * (width - 1) if width > 1 else ":" for width in widths]
    return "\n".join([line(cells[0]), "| " + " | ".join(rule) + " |", *(line(row) for row in cells[1:])])


@dataclass
class _Outcome:
    """One row of the summary half: what pytest reported for a test."""

    nodeid: str
    outcome: str
    duration: float


@dataclass
class _Note:
    """A line of prose in a detail section."""

    text: str


@dataclass
class _Failure:
    """A test's failure text, shown at the end of its own detail section."""

    text: str


class Reporter:
    """
    Collect run outcomes and result tables, then render both into one file.
    """

    def __init__(
        self,
        path: Path | None = None,
        fmt: ReportFormat = ReportFormat.HTML,
        title: str = "Mosaic test results",
    ):
        self._path = path
        self._format = ReportFormat(fmt)
        self._title = title
        self._started = dt.datetime.now().astimezone()
        #: Detail sections as (nodeid, items), in the order the tests ran.
        self._sections: list[tuple[str, list[Table | _Note | _Failure]]] = []
        #: Summary rows, in the order pytest finished the tests.
        self._outcomes: list[_Outcome] = []
        #: Tables rendered above the summary -- what the run was pointed at and how it was
        #: configured. Ordered as added, deduplicated by title.
        self._header_tables: list[Table] = []

    @property
    def writes_file(self) -> bool:
        return self._path is not None

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def format(self) -> ReportFormat:
        return self._format

    # -- header tables -------------------------------------------------------------------

    def add_header_table(
        self,
        rows: list[list[str]],
        *,
        title: str,
        headers: list[str] | None = None,
        left: set[int] | None = None,
    ) -> None:
        """
        Add a table above the run summary, describing the run as a whole.
        """
        if any(table.title == title for table in self._header_tables):
            return
        table = Table(
            headers=headers or ["property", "value"],
            rows=[[str(cell) for cell in row] for row in rows],
            title=title,
            left=left if left is not None else {1},
        )
        self._header_tables.append(table)
        if not self.writes_file:
            print(f"\n  {title}")
            print(render_table(table))
            return
        self.flush()

    def set_environment(self, rows: list[list[str]], title: str = "Environment") -> None:
        """What this run was pointed at -- profile, hardware shape, endpoints, driver."""
        self.add_header_table(rows, title=title)

    # -- detail half ---------------------------------------------------------------------

    def start_test(self, nodeid: str) -> None:
        """Open a detail section for *nodeid*, reusing it if it already exists."""
        if self.writes_file:
            self._section(nodeid)

    def note(self, text: str) -> None:
        """A line of prose -- a heading for the table that follows, or a warning."""
        if not self.writes_file:
            print(f"  {text.strip()}")
            return
        self._append(_Note(text.strip()))

    def table(
        self,
        headers: list[str],
        rows: list[list[str]],
        *,
        title: str = "",
        left: set[int] | None = None,
        status_column: int | None = None,
    ) -> None:
        """Add one table. Cells must already be strings -- use the ``format_*`` helpers."""
        built = Table(
            headers=headers,
            rows=[[str(cell) for cell in row] for row in rows],
            title=title,
            left=left or set(),
            status_column=status_column,
        )
        if not self.writes_file:
            if title:
                print(f"\n  {title}")
            print(render_table(built))
            return
        self._append(built)

    # -- summary half, fed by the conftest hooks -----------------------------------------

    def record_outcome(self, nodeid: str, outcome: str, duration: float, failure_text: str = "") -> None:
        """
        Record one test's result for the summary table.
        """
        self._outcomes.append(_Outcome(nodeid=nodeid, outcome=outcome, duration=duration))
        if self.writes_file and failure_text:
            self._section(nodeid).append(_Failure(failure_text))
        self.flush()

    # -- rendering -----------------------------------------------------------------------

    def _section(self, nodeid: str) -> list[Table | _Note | _Failure]:
        """The item list for *nodeid*, created on first use."""
        for name, items in self._sections:
            if name == nodeid:
                return items
        items: list[Table | _Note | _Failure] = []
        self._sections.append((nodeid, items))
        return items

    def _append(self, item: Table | _Note | _Failure) -> None:
        if not self._sections:
            self._sections.append(("", []))
        self._sections[-1][1].append(item)
        self.flush()

    def flush(self) -> None:
        """Write the report file. A no-op when reporting to stdout."""
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        body = self._render_markdown() if self._format is ReportFormat.MD else self._render_html()
        self._path.write_text(body, encoding="utf-8")

    def _summary_table(self) -> Table:
        return Table(
            headers=["test", "outcome", "duration"],
            rows=[[o.nodeid, o.outcome, f"{o.duration:.2f}s"] for o in self._outcomes],
            status_column=1,
        )

    def _tally(self) -> list[tuple[str, str]]:
        """(label, outcome-or-empty) pairs for the counts line, in a stable order."""
        counts: dict[str, int] = {}
        for outcome in self._outcomes:
            counts[outcome.outcome] = counts.get(outcome.outcome, 0) + 1
        total = sum(outcome.duration for outcome in self._outcomes)
        pairs = [(f"{counts[name]} {name}", name) for name in sorted(counts)]
        pairs.append((f"{total:.1f}s total", ""))
        return pairs

    def _render_html(self) -> str:
        detail: list[str] = []
        for nodeid, items in self._sections:
            if not items:
                continue
            if nodeid:
                detail.append(f"<h3>{html.escape(nodeid)}</h3>")
            for item in items:
                match item:
                    case _Note(text):
                        detail.append(f"<p class='note'>{html.escape(text)}</p>")
                    case _Failure(text):
                        detail.append(f"<pre class='failure'>{html.escape(text)}</pre>")
                    case Table() as table:
                        heading = f"<h4>{html.escape(table.title)}</h4>" if table.title else ""
                        detail.append(f"{heading}<div class='scroll'>{_render_table_html(table)}</div>")

        chips = "".join(
            f"<span class='{_STATUS_CLASS.get(outcome, '')}'>{html.escape(label)}</span>"
            for label, outcome in self._tally()
        )
        details_section = "<h2>Details</h2>" + "".join(detail) if detail else ""
        environment = "".join(
            f"<h2>{html.escape(table.title)}</h2><div class='scroll'>{_render_table_html(table)}</div>"
            for table in self._header_tables
        )
        return (
            "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(self._title)}</title><style>{_CSS}</style></head><body><main>"
            f"<h1>{html.escape(self._title)}</h1>"
            f"<p class='meta'>{self._started:%Y-%m-%d %H:%M:%S %Z}</p>"
            f"<p class='tally'>{chips}</p>"
            f"{environment}"
            "<h2>Test results</h2>"
            f"<div class='scroll'>{_render_table_html(self._summary_table())}</div>"
            f"{details_section}"
            "</main></body></html>\n"
        )

    def _render_markdown(self) -> str:
        out: list[str] = [
            f"# {self._title}",
            "",
            f"{self._started:%Y-%m-%d %H:%M:%S %Z} — " + ", ".join(label for label, _ in self._tally()),
            "",
        ]
        for table in self._header_tables:
            out += [f"## {table.title}", "", render_table_markdown(table), ""]
        out += [
            "## Test results",
            "",
            render_table_markdown(self._summary_table()),
            "",
        ]

        detail: list[str] = []
        for nodeid, items in self._sections:
            if not items:
                continue
            if nodeid:
                detail += [f"### {nodeid}", ""]
            for item in items:
                match item:
                    case _Note(text):
                        detail += [text, ""]
                    case _Failure(text):
                        # Fenced, so a traceback's own indentation and pipes survive intact.
                        detail += ["```text", text, "```", ""]
                    case Table() as table:
                        if table.title:
                            detail += [f"#### {table.title}", ""]
                        detail += [render_table_markdown(table), ""]

        if detail:
            out += ["## Details", "", *detail]
        return "\n".join(out).rstrip() + "\n"


class ReportPlugin:
    """
    Feeds pytest's own results into a :class:`Reporter`'s summary half.
    """

    def __init__(self, reporter: Reporter):
        self.reporter = reporter

    def pytest_runtest_logreport(self, report):
        """
        Record one finished phase per test.
        """
        if report.when == "call":
            outcome = report.outcome
        elif report.failed:
            outcome = "error"
        elif report.when == "setup" and report.skipped:
            outcome = "skipped"
        else:
            return

        self.reporter.record_outcome(
            nodeid=report.nodeid,
            outcome=outcome,
            duration=report.duration,
            failure_text=report.longreprtext if report.failed else "",
        )

    def pytest_sessionfinish(self, session):
        """Write the file one last time and say where it went."""
        if not self.reporter.writes_file:
            return
        self.reporter.flush()
        writer = session.config.get_terminal_writer()
        writer.line(f"\n{self.reporter.format.upper()} report: {self.reporter.path}")
