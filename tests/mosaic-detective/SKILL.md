---
name: mosaic-detective
description: Diagnose faults in an OpenMosaic GPU cluster from Prometheus metrics. Use when asked what happened to a cluster, why collectives slowed down, why GPU metrics stopped, what caused a throughput drop, to investigate a suspected NCCL, GPU, network, or telemetry fault at a given time, or when the user says "Kowalski, analysis!".
---

# Mosaic Detective

Diagnose faults in a multi-GPU OpenMosaic cluster by querying its Prometheus
metrics.

## Setup

`PROMETHEUS_HOST` and `PROMETHEUS_PORT` are already exported in the
environment. Do not check them. Run all scripts from this directory.

## Available scripts

- `scripts/check_collector_health.py` — which scrape targets are up or down
- `scripts/metric_timeline.py <metric> [--minutes N] [--transitions]` — how a
  metric moved, and when it changed
- `scripts/compare_ranks.py <metric> [--minutes N] [--stat delta|mean]` —
  compare a metric across ranks
- `scripts/list_metrics.py [--filter SUBSTR]` — what metrics exist

## Diagnostic procedure

**1. Is the telemetry pipeline alive?**

    python scripts/check_collector_health.py

If `otel-collector` is DOWN, metrics have stopped arriving. Ignore `vllm-*`
targets being DOWN — that is the normal idle state.

**2. What is the overall picture?**

    python scripts/metric_timeline.py nccl_profiler_collective_bytes_total --minutes 5

Is throughput normal, reduced, or completely stopped (first == last)?

**3. Is it one rank, or all of them?**

    python scripts/compare_ranks.py nccl_profiler_collective_bytes_total --minutes 5 --stat delta
    python scripts/compare_ranks.py DCGM_FI_DEV_SM_CLOCK --minutes 5 --stat mean

The first shows whether a rank has stopped. The second shows whether a GPU is
clock-limited — the ONLY way to localise a clamp, because all-reduce
synchronises ranks and makes NCCL metrics uniform even when one GPU is slow.

**4. When did it change?**

    python scripts/metric_timeline.py <metric> --minutes 60 --transitions

Prints timestamps where a value changed sharply. Add `--threshold 0.5` if noisy.

**5. If a metric name returns no data**

    python scripts/list_metrics.py --filter nccl

An unrecognised name returns "no data", which is NOT the same as flatlined.

## Do not write raw PromQL or inline Python

Every question you need is covered by these scripts. If something seems
unanswerable with them, say so rather than improvising — an improvised query
with a wrong label returns empty, which reads as "flatlined" and produces a
wrong diagnosis.

## Window strategy

Faults are usually recent. Start narrow, widen only if you find nothing:
`--minutes 5`, then 30, then 120.

A ratio computed over a window spanning both healthy and faulty periods is
diluted. A GPU clamped to 26% of peers reads 0.86 over a mostly-pre-fault
10-minute window, but 0.26 over a 3-minute post-fault window. If a ratio looks
mildly off rather than clearly wrong, narrow the window and re-check.

Note that Prometheus retains history: widening the window may surface an OLD
fault that has already been resolved. Check whether the fault is still active
before reporting it as current.

## Reading the results

See `fault-signatures.md` for each fault's signature and the rules for
distinguishing them. See `metrics-reference.md` for which metrics to trust.

## Reporting

State the fault type, the evidence, and what you ruled out. If evidence is
ambiguous, say so rather than guessing.
