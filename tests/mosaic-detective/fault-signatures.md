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

**Localising the dead rank — apply in order, stop at the first that resolves.**

The job aborts within seconds of a rank dying, so by the time you look, all
ranks are usually frozen — the collector state, not which ranks stopped, is
the discriminator for *fault type*. Localising *which* rank died is a separate
question and often unanswerable.

1. `metric_timeline.py namedprocess_namegroup_num_procs --minutes 15 --find-anomaly`
   If one host steps 2→1 while the other is still at 2, the dying rank is on
   that host. Report **confirmed (host)**. If both hosts step 2→0 within the
   same sample, this check is exhausted — continue.

2. `metric_timeline.py nccl_profiler_collective_bytes_total --minutes 15 --find-anomaly`
   Compare stop timestamps **only between ranks on the same host**. Ranks on
   different hosts are scraped by different targets and their timestamps may
   differ by up to one scrape interval from scrape phase alone — a cross-host
   stagger is not evidence and must not be reported as such.

3. A rank showing a **reduced but nonzero final sample** while its peers are
   already at zero is the **last survivor**, not the victim. It kept moving
   bytes after the others blocked. Do not name it as the dead rank. The victim
   typically stops in the same sample as its healthy same-host peer and leaves
   no signature at all.

4. Otherwise, report: "the job stopped; the dying rank is not resolvable at
   this scrape resolution." This is a complete and correct answer. Do NOT name
   a rank on the strength of a single-interval difference or a trailing partial
   — naming the wrong rank is worse than reporting it unresolved.

Always state a confidence: **confirmed** (step 1 resolved it) or **unresolved**
(step 4). Never name a rank without one.

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