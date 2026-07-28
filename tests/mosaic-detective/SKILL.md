---
name: mosaic-detective
description: Diagnose faults in an OpenMosaic GPU cluster from Prometheus metrics. Use when asked what happened to a cluster, why collectives slowed down, why GPU metrics stopped, what caused a throughput drop, to investigate a suspected NCCL, GPU, network, or telemetry fault at a given time, or when the user says "Kowalski, analysis!".
---

# Mosaic Detective

Diagnose faults in a multi-GPU OpenMosaic cluster by querying its Prometheus
metrics.

## Setup

`PROMETHEUS_HOST` and `PROMETHEUS_PORT` are already exported. Do not check
them. Run all scripts from this directory.

## Available scripts

- `scripts/check_collector_health.py` — which scrape targets are up or down
- `scripts/metric_timeline.py <metric> [--minutes N] [--transitions] [--find-anomaly]` — how a metric moved, and when its rate shifted
- `scripts/compare_ranks.py <metric> [--minutes N] [--stat delta|mean] [--by rank|link]` — compare a metric across ranks, or across source→dest links
- `scripts/list_metrics.py [--filter SUBSTR]` — what metrics exist

Do not write raw PromQL or inline Python. Every question is covered by these
scripts. If something seems unanswerable with them, say so rather than
improvising — a query with a wrong label returns empty, which reads as
"flatlined" and produces a wrong diagnosis.

## Diagnostic procedure

**1. Is the telemetry pipeline alive?**

    python scripts/check_collector_health.py

`up` is authoritative. If `otel-collector` shows DOWN, the collector is down —
report it. Do NOT talk yourself out of this because NCCL metrics still look
recent: after the collector dies, its last samples linger in Prometheus and
appear "fresh" for a minute or two even though nothing new is arriving. Stale-
but-present data is the EXPECTED look of a just-killed collector, not evidence
against it.

Confirm which it is by checking whether NCCL counters are still *advancing*:
re-query collective bytes over the last 1–2 minutes and compare the newest
value to one from 30s earlier. If it has not moved, the collector is dead even
if the last value is recent. `vllm-*` targets being DOWN is normal — ignore.

**2. What is the overall picture?**

    python scripts/metric_timeline.py nccl_profiler_collective_bytes_total --minutes 1
    python scripts/metric_timeline.py nccl_profiler_collective_bytes_total --minutes 5

Is throughput normal, reduced, or completely stopped (first == last)?

**Always compare the two.** Convert each to MB/s per rank: (last - first) /
seconds / 1e6. The 5-minute figure is the cluster's own recent baseline; the
1-minute figure is now.

If the 1-minute rate is below ~0.8x the 5-minute rate, throughput is
degrading **right now** and you must not report healthy. A fault that started
seconds ago barely moves the 5-minute average but halves the 1-minute one —
this ratio is the only thing that catches a degradation in its first minute.

Both figures being equally low is also degradation, once the fault has run
long enough to drag the wide window down. Compare that value against the
known healthy rate in fault-signatures.md before concluding health.

State both numbers in your report.

**3. Is it one rank, or all of them?**

    python scripts/compare_ranks.py nccl_profiler_collective_bytes_total --minutes 2 --stat delta
    python scripts/compare_ranks.py DCGM_FI_DEV_SM_CLOCK --minutes 2 --stat mean

The first shows whether a rank has stopped. The second shows whether a GPU is
clock-limited — the ONLY way to localise a clamp, because all-reduce
synchronises ranks and makes NCCL metrics uniform even when one GPU is slow.

**4. When did it change?**

    python scripts/metric_timeline.py <metric> --minutes 15 --find-anomaly

Reports where each series' rate of change shifted — catches a gauge dropping
(clamped clock) or a counter freezing (dead rank or collector). Works cleanly
on `nccl_profiler_collective_bytes_total` and DCGM gauges. Avoid it on
`nccl_profiler_transfer_time_microseconds_sum` — that metric is too noisy at
15s resolution and produces spurious results.

## Be decisive

Reach a diagnosis in as few steps as possible; the four checks above are
usually enough. Once the evidence is unambiguous, stop and report — do not
re-run a comparison at multiple window sizes to refine a number that is
already conclusive, and do not chase corroboration for a clear finding. A
clock ratio of 0.5 against uniform peers is already diagnostic; you need not
narrow until it reads 0.26. If a check is ambiguous or unhelpful, move on
rather than retrying it a different way.

Decisiveness applies to fault *type*, which the four checks resolve reliably.
It does not apply to naming a specific rank as the cause of a job stop — that
has its own procedure in fault-signatures.md and "unresolved" is a valid
verdict there.

## Window strategy

Faults are usually recent. Start narrow (`--minutes 2`), widen only if you
find nothing. A ratio over a window spanning both healthy and faulty periods
is diluted — a GPU clamped to 26% of peers reads 0.86 over a mostly-pre-fault
10-minute window but 0.26 over a 3-minute post-fault window. If a ratio looks
mildly off rather than clearly wrong, narrow once and re-check; one narrowing
is enough.

Prometheus retains history, so a wide `--find-anomaly` scan may surface a
fault that is already resolved. Before reporting, confirm the fault is active
**now** by re-checking the most recent 1–2 minutes. If values have returned to
normal, say so ("a clock clamp occurred at 09:38 but has since cleared")
rather than reporting it as current. Signs of a resolved fault: a mean sitting
between healthy and faulty values, a return to normal near the window's end,
or an anomaly timestamp many minutes old.

## Reading the results

See `fault-signatures.md` for each fault's signature and how to tell them
apart, and `metrics-reference.md` for which metrics to trust.

## Reporting

State the fault type, the evidence, and what you ruled out. Report in plain
operational terms ("a rank process died on golf", "the interconnect is
degraded") — no internal ticket numbers or section references. If evidence is
ambiguous, say so rather than guessing.