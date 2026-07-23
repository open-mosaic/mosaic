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
| `mosaic_gpu_rank_mapping` | info | — | rank to hostname/gpu_uuid; only exports during a workload |

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

DCGM metrics carry `UUID`, `host`, `gpu` but no `rank`. `compare_ranks.py`
joins them to ranks via `mosaic_gpu_rank_mapping`, which requires a running
workload. Use `host` (lowercase), not `Hostname` — the latter is a container ID.
