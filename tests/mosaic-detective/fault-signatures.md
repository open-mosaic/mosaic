# Fault signatures

How each fault appears in Mosaic metrics. Signatures are expressed as
**relationships between ranks**, not absolute values, so they port across
clusters and hardware.

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

## 1. GPU clock clamp

| check | result |
|---|---|
| collector health | all targets UP |
| collective bytes, `--stat delta` | all ranks ~1.00 (uniform) |
| `DCGM_FI_DEV_SM_CLOCK`, `--stat mean` | **one rank far below peers** |
| cluster throughput vs baseline | reduced |

**Diagnosis:** throughput down, ranks uniform, one GPU's clock ratio well
below its peers → that rank's GPU is clock-limited.

Worked example (2-node Blackwell reference cluster): healthy SM clock
2790–2820 MHz, spread ~1%. Clamped rank read 742 MHz, ratio 0.26.

A clamp is a **step change to a fixed value**, not a gradual decline. If the
clock appears to be "trending down", that is window dilution — the mean is
averaging pre-clamp and post-clamp samples. Narrow the window to see the true
clamped value.

---

## 2. Network degradation — netem delay/loss

| check | result |
|---|---|
| collector health | all targets UP |
| collective bytes, `--stat delta` | all ranks ~1.00 (uniform) |
| `DCGM_FI_DEV_SM_CLOCK`, `--stat mean` | **all ranks normal (~1.00)** |
| cluster throughput vs baseline | reduced |

**Diagnosis by elimination:** throughput down, ranks uniform, GPU clocks all
healthy, telemetry pipeline healthy → the interconnect is degraded.

**Key signal:** netem produces no per-rank divergence — every rank stays
uniform, which superficially looks healthy. The fault shows only as the
whole cluster's rate dropping relative to its own earlier healthy period.
Uniformity across ranks does NOT mean healthy.

Read the rate transition (`--find-anomaly` on collective bytes):
- rate → **0.00/s** on all ranks = stopped (collector kill or job death)
- rate → **lower nonzero** on all ranks = degraded, still running
- rate drop + all SM clocks normal + collector UP = **network degradation**
- rate drop + one SM clock low = clock clamp, not network

Worked example: healthy ~78.9 MB/s per rank; under 20 ms added delay this fell
to ~26.3 MB/s.

**A pure delay fault is transient.** TCP adapts its congestion window within
1–2 minutes and throughput recovers, after which the cluster looks healthy
because it is. Delay is only detectable in the window just after onset. Packet
loss is different — it causes ongoing retransmission and stays visible.

**Known limitation:** node-level network counters (`node_network_*`,
`node_netstat_*`) report the node-exporter container's virtual interface, not
the physical interconnect. Diagnose the fault *type*; do not name the node
from these metrics.

---

## 3. OTel collector killed

| check | result |
|---|---|
| collector health | **`otel-collector` DOWN** |
| collective bytes, `--stat delta` | **all ranks 0.00 — every series frozen** |

**Diagnosis:** every NCCL series stops advancing simultaneously and the
collector target is DOWN → a telemetry gap, **not** a cluster failure. The job
is still running; you simply cannot see it.

---

## 4. Rank killed

| check | result |
|---|---|
| collector health | all targets UP (**collector still UP**) |
| collective bytes, `--stat delta` | **all ranks 0.00 — every series frozen** |

**Diagnosis:** all counters freeze abruptly while the collector stays healthy
→ the job stopped. Check the stagger to identify a dying rank.

The job aborts within seconds of a rank dying, so by the time you look, all
ranks are usually frozen — the collector state, not which ranks stopped, is
the discriminator.

**The dying rank is identifiable from timing.** Its counter stops roughly
10–20 seconds before its peers', because the survivors keep running until they
block on the next collective. `--find-anomaly` timestamps reveal the order:
the rank with the earliest stop is the one that died. GPU clocks drop to idle
in the same order, node by node.

Worked example: rank 2 on golf stopped at 18:12:24; the other three at
18:12:39; golf's GPUs idled at 18:12:50 and bravo's at 18:13:05.

If the timestamps are all identical, the stagger has been lost to the scrape
interval — report that the job stopped, and note that a killed rank and a
completed run are indistinguishable in that case.

---

## Distinguishing the "flatline" cases

When every NCCL counter stops advancing, the collector state is the
discriminator — NOT which ranks stopped:

- All ranks frozen + collector **DOWN** → collector killed. The job is still
  running; you have lost visibility into it.
- All ranks frozen + collector **UP** → the job stopped. Either a rank died or
  the run completed.

**A killed rank and a completed job look similar.** Both show all ranks
freezing with the collector healthy and GPUs going idle. Do not rule out a
rank kill merely because there is no asymmetry — by the time you look, the
survivors have already blocked and stopped too. The exception is the
stop-order stagger above: if one rank stopped measurably first, a rank died.
If you cannot tell, say so.

## Environment noise — not faults

`vllm-bravo` and `vllm-golf` targets are DOWN whenever no inference workload
is running — the normal idle state. **Only `otel-collector` being DOWN is
diagnostic.** Do not report vLLM targets as a fault.

NCCL metrics only export while a workload is running. Their absence means "no
workload", NOT "the ranks died" — check collector health before concluding a
failure.