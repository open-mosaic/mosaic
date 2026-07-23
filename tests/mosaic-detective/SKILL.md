---
name: mosaic-detective
description: Diagnose faults in an OpenMosaic GPU cluster from Prometheus metrics. Use when asked what happened to a cluster, why collectives slowed down, why GPU metrics stopped, what caused a throughput drop, to investigate a suspected NCCL, GPU, network, or telemetry fault at a given time, or when the user says "Kowalski, analysis!".
---

# Mosaic Detective

Diagnose faults in a multi-GPU OpenMosaic cluster by querying its Prometheus
metrics. Never write raw PromQL — use the scripts below.

## Setup

All scripts need the Prometheus host. Set it once:
cat > metrics-reference.md << 'EOF'
# Metrics reference

## Trusted

| metric | type | stat | notes |
|---|---|---|---|
| `nccl_profiler_collective_bytes_total` | counter | delta | bytes moved per rank; has `rank` and `hostname` labels |
| `nccl_profiler_collective_time_microseconds_sum` | counter | delta | cumulative collective time |
| `nccl_profiler_rank_bytes_total` | counter | delta | per-rank bytes |
| `DCGM_FI_DEV_SM_CLOCK` | gauge | mean | per-GPU clock; the only reliable clock-clamp localiser |
| `DCGM_FI_DEV_POWER_USAGE` | gauge | mean | per-GPU power draw |
| `up` | gauge | — | scrape target liveness, via check_collector_health |
| `mosaic_gpu_rank_mapping` | info | — | rank → hostname/gpu_uuid; only exports during a workload |

## Avoid

`nccl_profiler_rank_rate_MB_per_second_*` — reports physically impossible
values; TCP `send()` returns at kernel handoff, not on the wire.

`node_network_*`, `node_netstat_*` — the node-exporter container reports its
own virtual `eth0`, not the physical interconnect. Not usable for diagnosing
interconnect faults.

`nccl_profiler_rank_latency_microseconds_*`, `*_transfer_latency_*` — may not
export under fixed-size load.

## Labels

NCCL metrics carry `rank` and `hostname` directly.

DCGM metrics carry `UUID`, `host`, `gpu` but **no `rank`**. `compare_ranks.py`
joins them to ranks via `mosaic_gpu_rank_mapping`, which requires a running
workload. Use `host` (lowercase), not `Hostname` — the latter is a container ID.


## Window strategy

Faults are usually recent. Start narrow and widen only if you find nothing.

1. Start with `--minutes 5`. This catches an active fault cleanly, without
   averaging it against healthy data from before it started.
2. If nothing looks wrong, widen to `--minutes 30`, then `--minutes 120`.
3. Once you have found something, use `--transitions` to pin down exactly
   when it started:
   `python scripts/metric_timeline.py <metric> --minutes 60 --transitions`

**Why narrow first:** a ratio computed over a window that spans both healthy
and faulty periods is diluted. A GPU clamped to 26% of peers reads as 0.86
over a 10-minute window that is mostly pre-fault, but 0.26 over a 3-minute
window that is entirely post-fault. If a ratio looks mildly off rather than
clearly wrong, narrow the window and re-check before concluding.