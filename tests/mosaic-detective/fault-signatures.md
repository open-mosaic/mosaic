# Fault signatures

How each fault appears in Mosaic metrics. Signatures are expressed as
**relationships between ranks**, not absolute values, so they port across
clusters and hardware.

## Diagnostic order

1. `check_collector_health.py` — is the telemetry pipeline alive?
2. `metric_timeline.py nccl_profiler_collective_bytes_total` — is throughput
   normal, degraded, or stopped?
3. `compare_ranks.py <metric> --stat delta|mean` — is it one rank, one node,
   or all of them?

## Key property: all-reduce synchronises ranks

Every rank waits for the slowest. A fault affecting ONE GPU therefore slows
ALL ranks equally. Per-rank comparison of NCCL metrics does **not** localise
degradation faults — all ranks read ~1.00 even when one GPU is crippled.
Localisation must come from hardware metrics (DCGM), which are per-GPU and
unaffected by synchronisation.

## Choosing --stat

Counters (`*_total`, `*_sum`) → `--stat delta` (growth over the window).
Gauges (clocks, temperatures, utilisation) → `--stat mean`.
Using `delta` on a gauge yields ~0 for every rank and a meaningless ratio.

---

## 1. GPU clock clamp (T2.2)

| check | result |
|---|---|
| collector health | all targets UP |
| collective bytes, `--stat delta` | all ranks ~1.00 (uniform) |
| collective time, `--stat delta` | all ranks ~1.00 (uniform) |
| `DCGM_FI_DEV_SM_CLOCK`, `--stat mean` | **one rank far below peers** |
| cluster throughput vs baseline | reduced |

**Diagnosis:** throughput down, ranks uniform, one GPU's clock ratio well
below its peers → that rank's GPU is clock-limited.

Worked example (2-node Blackwell reference cluster): healthy SM clock
2790–2820 MHz, spread ~1%. Clamped rank read 742 MHz, ratio 0.26.

---

## 2. Network degradation — netem delay/loss (T2.3)

| check | result |
|---|---|
| collector health | all targets UP |
| collective bytes, `--stat delta` | all ranks ~1.00 (uniform) |
| `DCGM_FI_DEV_SM_CLOCK`, `--stat mean` | **all ranks normal (~1.00)** |
| cluster throughput vs baseline | reduced |

**Diagnosis by elimination:** throughput down, ranks uniform, GPU clocks all
healthy, telemetry pipeline healthy → the interconnect is degraded.

Worked example: healthy ~4.6 GB/min per rank; under 50 ms added delay,
~2.3 GB/min (roughly half). Severe delay can reduce counter growth enough
that short windows show near-zero movement — widen the window before
concluding the counters have stopped.

**Known limitation:** node-level network counters
(`node_network_*`, `node_netstat_*`) report the node-exporter *container's*
virtual interface (`eth0`), not the physical interconnect. They cannot
localise this fault to a node on this deployment. Diagnose the fault *type*;
do not attempt to name the node from metrics alone.

---

## 3. OTel collector killed (T2.5)

| check | result |
|---|---|
| collector health | **`otel-collector` DOWN** |
| collective bytes, `--stat delta` | **all ranks 0.00 — every series frozen** |
| `metric_timeline` | first == last == min == max for every series |

**Diagnosis:** every NCCL series stops advancing *simultaneously* and the
collector target is DOWN → this is a telemetry gap, **not** a cluster
failure. The job is still running; you simply cannot see it.

---

## 4. Rank killed (T2.4)

| check | result |
|---|---|
| collector health | all targets UP (**collector still UP**) |
| collective bytes, `--stat delta` | **one rank 0.00, peers ~1.00** |

**Diagnosis:** one rank's counters freeze while peers keep advancing, and the
collector is healthy → that rank's process died.

Peers may continue for a period before the job aborts, so do not require all
ranks to stop. The discriminator against fault 3 is: **collector UP + only
some ranks frozen**.

---

## Distinguishing the two "flatline" faults

This is the distinction that matters most:

- **All** ranks frozen + collector **DOWN** → collector killed (§3)
- **One** rank frozen + collector **UP** → rank killed (§4)

## Environment noise — not faults

`vllm-bravo` and `vllm-golf` targets are DOWN whenever no inference workload
is running. This is the normal idle state. **Only `otel-collector` being DOWN
is diagnostic.** Do not report vLLM targets as a fault.

NCCL metrics (`nccl_profiler_*`, `mosaic_gpu_rank_mapping`) only export while
a workload is running. Their absence means "no workload", which is NOT the
same as "the ranks died" — check collector health and whether the metrics
existed earlier in the window before concluding a failure.
