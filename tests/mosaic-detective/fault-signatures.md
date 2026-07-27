# Fault signatures

How each fault appears in Mosaic metrics. Signatures are expressed as
**relationships between ranks**, not absolute values, so they port across
clusters and hardware.

Cluster shape varies: rank count, ranks per host, host count and scrape
interval are all deployment-specific. Never assume a particular topology.
Read it from `compare_ranks.py` output and from the rank-to-host mapping.

Worked examples throughout are observations from the reference cluster
(2 nodes, 4 Blackwell GPUs, 1 GbE interconnect). Absolute values differ on
other hardware; the ratios and the shape of each signature do not. Use them
as calibration for what a real fault looks like, not as thresholds.

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

## Never report "healthy" on ambiguous evidence

Reporting a degraded cluster as healthy is the worst available error: it ends
the investigation and leaves an operator trusting a broken system. A false
alarm merely costs someone a look.

Only report a healthy cluster when throughput is at its expected level, ranks
are uniform, clocks are uniform and the collector is up — all four, with no
unexplained drop anywhere in the window. If throughput is below the expected
rate and you cannot establish by how much, say so: "possible degradation,
magnitude unclear" is a better answer than "healthy".

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

**Worked example:** healthy SM clock 2790–2820 MHz, spread ~1% across GPUs.
The clamped rank read 742 MHz, ratio 0.26 against its peers.

A clamp is a **step change to a fixed value**, not a gradual decline. If the
clock appears to be "trending down", that is window dilution — the mean is
averaging pre-clamp and post-clamp samples. Narrow the window to see the true
clamped value.

Judge the clamp by the ratio against peers, not by an absolute clock value.
Healthy clocks differ by hardware generation, power cap and cooling; peers on
the same cluster do not.

**A clamp is only diagnosable while the job is running.** If collective bytes
have stopped entirely, every GPU falls to idle and clock ratios become
meaningless — see section 4.

---

## 2. Network degradation — netem delay/loss

| check | result |
|---|---|
| collector health | all targets UP |
| collective bytes, `--stat delta` | all ranks ~1.00 (uniform) |
| `DCGM_FI_DEV_SM_CLOCK`, `--stat mean` | **all ranks normal (~1.00)** |
| cluster throughput vs baseline | **reduced, sustained** |

**Diagnosis by elimination:** throughput down, ranks uniform, GPU clocks all
healthy, telemetry pipeline healthy → the interconnect is degraded.

**Worked examples (reference cluster):**

- Healthy steady-state throughput: **~78.9 MB/s per rank**.
- Under 20 ms added delay this fell to **~26.3 MB/s** — a dramatic drop.
- Under 0.1% packet loss the fall is **much subtler, around 20–30% below
  baseline** — roughly 55–65 MB/s per rank. This is the easiest fault in the
  set to miss, because the cluster still looks busy and every rank stays
  uniform. A rate in that band is degradation, not health.

**Key signal:** netem produces no per-rank divergence — every rank stays
uniform, which superficially looks healthy. The fault shows only as the
whole cluster's rate dropping relative to its own earlier healthy period.
Uniformity across ranks does NOT mean healthy.

**Any sustained fall in collective throughput across all ranks is a fault
until proven otherwise.** With clocks uniform and the collector up, network
degradation is the diagnosis. Do not hunt for reasons the drop might be
normal — a drop is the thing you are looking for.

Read the rate transition (`--find-anomaly` on collective bytes):
- rate → **0.00/s** on all ranks = stopped (collector kill or job death)
- rate → **lower nonzero** on all ranks = degraded, still running
- rate drop + all SM clocks normal + collector UP = **network degradation**
- rate drop + one SM clock low = clock clamp, not network

**Do not compare the current rate against the window's overall mean.** If the
window spans the fault, that mean is dragged down by the degraded period
itself, and the comparison can show the current rate as "at or above baseline"
while the cluster is plainly degraded. Compare against the healthy plateau
earlier in the window, or against the reference rate above.

**A pure delay fault is transient.** TCP adapts its congestion window within
1–2 minutes and throughput recovers, after which the cluster looks healthy
because it is. Delay is only detectable in the window just after onset.

**Packet loss is different — it stays visible, and it settles.** Retransmission
is continuous, so the rate holds steady at a new level below baseline. That
steadiness reads as "recovered" and is the most common way to miss this fault.
**Steadiness is not recovery.** Only a return to the baseline rate is recovery.

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
→ the job stopped.

The job aborts within seconds of a rank dying, so by the time you look, all
ranks are usually frozen — the collector state, not which ranks stopped, is
the discriminator.

### Confirm the stop, then stop

Two further checks are worth running as evidence, one call each, and no more.

**Process count — did the workload actually exit?**

    python scripts/metric_timeline.py namedprocess_namegroup_num_procs --minutes 15 --find-anomaly

Workload processes going to zero on every host confirms the job exited, rather
than the metrics merely stalling. If this returns nothing at all, the cluster
may not run process-exporter — check with `list_metrics.py --filter
namedprocess` before treating an empty result as a count of zero. An empty
result from a metric that does not exist is not evidence.

**GPU clocks — was any GPU clamped before the stop?**

    python scripts/compare_ranks.py DCGM_FI_DEV_SM_CLOCK --minutes 15 --stat mean

Run this **once**, on a window wide enough to span the pre-stop period, purely
to rule out a clamp that preceded the failure. Uniform idle clocks afterwards
are expected and are a consequence of the stop, not a fault.

Do **not** re-run it at a narrower window to chase a ratio that looks odd.
After a job stop those ratios are dilution artifacts from a window straddling
the stop event, and disproving one costs another query for no diagnostic gain.

A "no data" result here usually means the rank mapping was unavailable, not
that the GPUs stopped reporting — widen the window rather than concluding the
metric is absent.

### Do not attempt to localise the dead rank

**Do not name a rank as the one that died.** The abort propagates to every
rank inside a single scrape interval, so the victim stops in the same sample
as its healthy peers and leaves no signature. There is nothing in the metrics
to find, and any rank named from this data is a guess presented as a finding.

Verified on the reference cluster: rank 1 on bravo was killed, and rank 0 —
its same-host peer, scraped by the same target — stopped in the same sample.
No stagger was observable at 15s scrape resolution.

Report it in these terms: *the job stopped at HH:MM:SS; which rank died first
is not resolvable at this scrape resolution.* That is a complete and correct
answer, not a gap in the investigation.

**Two patterns that look like evidence and are not:**

- A rank showing a **reduced but nonzero final sample** while its peers have
  already reached zero is the **last survivor**, not the victim — it kept
  moving bytes after the others blocked. Do not report it, even as suggestive.
- A **single-sample difference** between stop timestamps is scrape jitter,
  especially across hosts, which are scraped by different targets. Only a gap
  of several samples would mean anything, and that does not occur here.

---

## Distinguishing the "flatline" cases

When every NCCL counter stops advancing, the collector state is the
discriminator — NOT which ranks stopped:

- All ranks frozen + collector **DOWN** → collector killed. The job is still
  running; you have lost visibility into it.
- All ranks frozen + collector **UP** → the job stopped. Either a rank died or
  the run completed.

**A killed rank and a completed job look similar.** Both show all ranks
freezing with the collector healthy and GPUs falling to idle clocks. Do not
rule out a rank kill merely because there is no asymmetry — by the time you
look, the survivors have already blocked and stopped too, in the same sample
as the victim.

Absence of a stagger is therefore **not** evidence of a clean completion, and
presence of a trailing partial is **not** evidence of a rank death. If the two
cannot be separated, say so and report both as possible.

## Environment noise — not faults

`vllm-*` targets are DOWN whenever no inference workload is running — the
normal idle state. **Only `otel-collector` being DOWN is diagnostic.** Do not
report vLLM targets as a fault.

NCCL metrics only export while a workload is running. Their absence means "no
workload", NOT "the ranks died" — check collector health before concluding a
failure.

Idle GPU clocks across *every* GPU are a consequence of no workload running,
not a per-GPU fault. A clamp shows as one GPU low **relative to busy peers**.