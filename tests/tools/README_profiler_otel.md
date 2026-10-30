<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# profiler_otel sweeps

`launch_sweep.py` runs the profiler OTEL suite once per row of a sweep table, writes each run's
report to its own file, collects every run's numbers into one CSV, and charts them. Use it to
find where a deployment saturates: the concurrency at which throughput stops rising and latency
starts climbing.

The tool and its `sweep.config` live in `tests/tools`; the suite it drives and the profiles it
rewrites are in `tests/suites`. It finds them relative to itself, so `./launch_sweep.py` works from
`tests/tools` (or by full path from anywhere), while a plain `pytest` run is made from
`tests/suites` as before. Paths below are written from whichever of the two the command belongs
to.

---

## Before you start

**1. The deployment must already be serving.** The suite drives an existing cluster; it deploys
nothing (`deployment: external: true` in the profile). Bring vLLM up first and confirm it
answers:

```bash
curl -m 5 http://<endpoint.host>:<endpoint.port>/health
```

**2. The profile must describe that deployment.** The profile you point the suite at declares
the hardware and the coverage the tests assert on. See [Generating the
profile](#generating-the-profile) below for how to write one and which fields decide whether the
tests pass.

**3. Passwordless SSH to the head node**, for the driver/CUDA versions in each report's
Environment table. Without it the report still generates and says why the versions are missing.

```bash
ssh -o BatchMode=yes username@endpoint.host nvidia-smi --version
```

**4. Addresses.** The cluster's own addresses come from the profile, which is the one place
they are written down. Each has a command-line flag and an environment variable for pointing a
single run somewhere else, in that order of precedence:

| Flag | Variable | Profile key | Default |
|---|---|---|---|
| `--gpu-info-ssh-user` | `GPU_INFO_SSH_USER` | `endpoint.gpu_info_ssh_user` | none; the report says why the driver rows are missing |
| `--prometheus-host` | `PROMETHEUS_HOST` | `endpoint.prometheus_host` | `endpoint.host` |
| `--grafana-host` | `GRAFANA_HOST` | `endpoint.grafana_host` | `endpoint.host` |
| `--grafana-port` | `GRAFANA_PORT` | `endpoint.grafana_port` | `3000` |

Nothing is defaulted to a particular cluster. A sweep that can name neither a Prometheus nor a
Grafana stops before its first row rather than reading someone else's metrics; a profile reached
through `endpoint.base_url` has no host to fall back to and has to name both.

The rest of the run is described the same way:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL` | unset | written to `serving.model` before every row |
| `LABEL` | derived from the model | tag at the front of report filenames |

Do **not** export `LABEL` to a full model id — a `/` in it is a path separator. The script
sanitises it and says so, but the derived default (`openai/gpt-oss-120b` → `gpt_oss_120b`) is
usually what you want, so leave it unset.

---

## Generating the profile

A profile is a YAML file describing one machine and the load to put on it. Nothing about the
format is specific to this suite, which is why the profiles live in a generically named
directory. The suite selects one by **filename stem** — `--workload-profile <my_cluster>`, never a
path — from `tests/suites/profiler_otel/profiles/`, or from any directory passed with
`--profile-dir`.

**1. Copy a template — do not edit a profile that is already in use.** The profiles sitting in
`tests/suites/profiler_otel/profiles/` were each written for one specific cluster and carry that cluster's
endpoint address, GPU counts and timeouts. Read them as examples, but start your own file from
[the template](#a-template-profile) below, saved under a name of your own:

```bash
$EDITOR ../suites/profiler_otel/profiles/<my_cluster>.yaml   # paste the template, then edit
```

**2. Change the values to match the deployment being tested** — the shape the cluster is
actually running, not what the hardware could in principle provide. The table below says which
fields decide whether the tests pass. Describing the cluster to an assistant and having it fill
the template in gets most of the way there, but check the coverage numbers by hand.

**3. Set `external: true` under `deployment`.** This tells the suite the stack is already
serving and stops it trying to bring up its own compose stack.

**4. Point `endpoint` at the serving node** — the proxy for a disaggregated deployment. The
template's defaults are placeholders, so this must be set when driving a remote cluster.

**5. Leave it in `tests/suites/profiler_otel/profiles/`** and refer to it by stem from then on:
`--workload-profile <my_cluster>`, or `--profile <my_cluster>` for `launch_sweep.py`.

### A template profile

Placeholders in braces are the values to replace. This one describes a four-node disaggregated
deployment running two cohorts of one worker each at TP=4 — the fields, not the numbers, are
what to copy:

```yaml
description: Four-node disaggregated, 4 GPUs per node

hardware:
  machines: 4
  gpus_per_machine: 4
  sku: rtx-pro-6000-blackwell

serving:
  mode: disaggregated
  model: {MODEL}
  prefill: {nodes: 2, workers: 1, tensor_parallel: 4, spans_nodes: false}
  decode:  {nodes: 2, workers: 1, tensor_parallel: 4, spans_nodes: false}
  kv_transfer: nixl

deployment:
  external: true

# {N1_IP} is the address of node 1 — the head node
endpoint:
  host: {N1_IP}
  port: 8192
  gpu_info_ssh_user: {USER}      # SSH login on the head node, for its driver and CUDA versions
  # prometheus_host: {N1_IP}   # default: `host` above
  # grafana_host: {N1_IP}      # default: `host` above
  # grafana_port: 3000           # default: 3000

coverage:
  hosts: 4
  gpus: 16
  communicators: 2

timeouts:
  workload: 1800
  metrics_available: 90
  quiesce: 90

expected_metrics: disagg_moe

benchmark_options:
  num_prompts: 512
  max_concurrency: 16
  random_input_len: 1024
  random_output_len: 128
  num_warmups: 4
  ignore_eos: true
  disable_tqdm: true
```

For an **aggregated** deployment, replace the `prefill`/`decode` pair with a single
`tensor_parallel:` under `serving`, set `coverage.communicators: 1`, and use
`expected_metrics: aggregated_dense`.

### The profile fields that decide whether tests pass

| Field | Meaning |
|---|---|
| `hardware.machines`, `hardware.gpus_per_machine` | the shape the deployment actually uses — match the launcher's `NUM_GPUS_PER_NODE`, not what `nvidia-smi` reports, if the containers are restricted to a subset |
| `coverage.hosts`, `coverage.gpus` | **exact** match, not a floor. A cluster reporting more GPUs than declared fails the same way as one reporting fewer |
| `coverage.communicators` | a floor (at least this many). Prefill and decode form separate NCCL communicators, so a disaggregated deployment wants at least 2 — this is what catches one cohort being dead |
| `serving.mode` | `aggregated` or `disaggregated`; the disaggregated form requires `prefill` and `decode` blocks |
| `serving.prefill` / `serving.decode` | per-cohort shape. `spans_nodes` records whether one worker's parallel group crosses a machine — `false` when `tp × pp` fits inside a node, in which case no NCCL collective leaves the node and the only inter-machine traffic is the KV transfer, which the profiler does not observe |
| `endpoint.host` / `endpoint.port` | where requests go — the proxy for a disaggregated deployment. Defaults are localhost, so this must be set when driving a remote cluster (or overridden with `VLLM_HOST`/`VLLM_PORT`) |
| `endpoint.prometheus_host`, `endpoint.grafana_host`, `endpoint.grafana_port`, `endpoint.gpu_info_ssh_user` | where results are read back from, both hosts defaulting to `endpoint.host`. A run pointed at the wrong Prometheus fails as if the cluster reported nothing — see item 4 of [Before you start](#before-you-start) for the flags and variables that override them |
| `timeouts.workload` | how long a run may take before the suite kills it |
| `benchmark_options` | passed verbatim to `benchmark_serving.py`; any flag that script accepts works here |

`launch_sweep.py` rewrites the swept keys in this profile before each row and restores the file
afterwards — see [The sweep table](#the-sweep-table) below.

---

## The sweep table

`sweep.config` is a plain table. The first non-comment line is the header; each column names a
key the script rewrites in the profile before that row runs.

```
max_concurrency   num_prompts   num_warmups   timeouts.workload
1                 64            2             1800
4                 64            8             1800
...
```

- A **dotted path** (`timeouts.workload`, `serving.model`, `coverage.gpus`) names one exact
  place in the profile. A bare name works too when it is unique in the file.
- Any key in the profile can be a column — `random_input_len`, `tensor_parallel`,
  `timeouts.metrics_available` — so the same machinery sweeps prompt length or parallelism, not
  just concurrency.
- Every row must fill every column; a short row aborts before anything runs, because it would
  otherwise silently inherit the previous row's value.
- An optional `name` column overrides that row's report filename.
- `#` comments and blank lines are ignored.

**The profile is restored on exit** — success, failure, or Ctrl-C — so an interrupted sweep
never leaves a modified profile behind.

---

## Running

From `tests/tools`:

```bash
./launch_sweep.py --list            # the table, and the filename each row will produce
./launch_sweep.py --dry-run         # print each row's plan; touch nothing
./launch_sweep.py                   # run every row
uv run --with matplotlib  ./launch_sweep.py --graph           # run, then chart
```

| Flag | Effect |
|---|---|
| `--list` | show the table and exit |
| `--dry-run` | print the plan; no profile edits, no runs |
| `--start N` | resume at row N (1-based, as `--list` numbers them) |
| `--only N` | run just row N |
| `--keep-going` | carry on past a failed row; stop after **2 failures in a row** |
| `--graph` | chart the results |
| `--config PATH` | a different table |
| `--out-dir DIR` | where reports, logs and the CSV go (default `/tmp/cai_sweep_<timestamp>`) |
| `--model ID` | write `serving.model` before every row |
| `--label TAG` | override the derived filename tag |
| `--profile NAME` | the profile to sweep, by stem (default `cai_4n`, or `$PROFILE`) |
| `--report-ext {html,md}` | report format |
| `--pytest-k EXPR` | pytest's `-k` for every row (default `not nccl_workload`, or `$PYTEST_K`).

**Failures.** By default the sweep stops at the first failed row and prints the `--start N` to
resume from. With `--keep-going` it records the row and continues, stopping only after two
consecutive failures — one failed row usually means that row is too heavy for the cluster, but
two in a row means the deployment is down and every remaining row will fail the same way. The
exit code is non-zero whenever any row failed, even if the sweep ran to the end.

**Charting an existing sweep**, without running anything:

```bash
uv run --with matplotlib ./launch_sweep.py --graph --out-dir /tmp/cai_sweep_20260904-140659
```

matplotlib is not a suite dependency, so graphing needs `uv run --with matplotlib` (or install
it once). The script tells you this if the import fails.

---

## Writing the report to a file

`launch_sweep.py` names and writes a report per row on its own (`--report-ext` picks the
format). A single `pytest` run writes one only when asked; append any of these:

```
--report-file=PATH
--report-format=[html|md]      # default html; a .md path infers md
--report-html=PATH             # alias for --report-file
--report-title=TEXT            # header title; default "Test Report"
--report-logo=PATH             # header image (svg/png/jpeg/gif/webp), embedded in the
                               # file itself; no logo when left off
--report-max-failures=N        # how many tracebacks the Failures section shows (default 5);
                               # every failure is listed in its table regardless
```

The report itself lives in the framework
(`production_test_framework.reporting`), so every suite that uses it publishes the
same document. Results are grouped into sections by marker; set `report_category_markers` in
`[tool.pytest.ini_options]` to change which markers name a section and in what order.

For example, from `tests/suites`:

```bash
uv run pytest -v --workload-profile <my_cluster> -k "not nccl_workload" \
  --report-file=/tmp/<my_cluster>_disagg.html
```

A direct `pytest` run reads Prometheus and Grafana from the profile's `endpoint` block, exactly
as a sweep does, and the same variables override it for one run:

```bash
GPU_INFO_SSH_USER=<username> \
PROMETHEUS_HOST=<host> GRAFANA_HOST=<host> GRAFANA_PORT=3000 \
  uv run pytest -v --workload-profile <my_cluster> -k "not nccl_workload"
```

`GPU_INFO_SSH_USER` lets the report collect driver, CUDA and kernel-module versions from the
head node over SSH. Without it the report still generates and states why those rows are missing.

**Copy your key to the head node before the first run.** Passwordless SSH is usually established
*between cluster nodes*; the machine driving the tests is normally not one of them, so its key is
not yet trusted anywhere. From the test machine:

```bash
ssh-copy-id <username>@<endpoint.host>
```

Then confirm it works non-interactively, which is exactly what the report does:

```bash
ssh -o BatchMode=yes <username>@<endpoint.host> nvidia-smi --version
```

Two details make this fail more often than it should:

- **The cluster login is usually not your local one.** `ssh <endpoint.host>` with no user goes as
  whoever you are locally and is refused; that is what `GPU_INFO_SSH_USER` exists for. A `Host`
  entry in the test machine's `~/.ssh/config` setting `User <username>` for the cluster addresses
  removes the need to set the variable at all.
- **`BatchMode=yes` means no password fallback.** The report never prompts — a missing key fails
  immediately rather than stalling the run — so a working interactive `ssh` is not proof that the
  probe will succeed. Test with the flag.

If it is not collected, the report says so with the exact target and SSH's own error, for example
`not collected -- ssh <endpoint.host>: Permission denied (publickey)`. Everything else in the
report is unaffected; only the driver rows are missing.

---

## What you get

Per row, in the output directory:

| File | Contents |
|---|---|
| `<label>_c<N>_p<N>_w<N>_t<N>.html` | the suite's own report: environment, workload configuration, results, NCCL metric coverage |
| `<label>_c<N>_p<N>_w<N>_t<N>.log` | that row's full pytest console output |

Plus, once per sweep:

| File | Contents |
|---|---|
| `sweep_results.csv` | one row per run: its configuration, `status`, and every measurement |
| `*.png` | the charts, with `--graph` |

The filename tag encodes the row that produced it — concurrency, prompts, warm-ups and workload
timeout — so a report is identifiable without opening it, and two sweeps of the same table
differ only by `<label>`.

### The HTML report

One file per run, four sections:

1. **Environment** — the profile and its resolved path, hardware shape, serving mode, prefill
   and decode cohort shapes, KV transfer, declared coverage, endpoint, the Prometheus and
   Grafana URLs, timeouts, the test runner, and the head node's driver / CUDA / KMD versions.
   This is what makes a result reproducible six months later.
2. **Test results** — every test with outcome and duration, colour-coded.
3. **Details** — per test: the workload configuration actually passed to `benchmark_serving.py`
   (with the equivalent command-line flag for each option), the benchmark results, the NCCL
   metric table showing each metric's baseline, current value, delta and status, and the
   per-host GPU coverage breakdown.
4. **Failure text**, appended to the section of whichever test failed, so a red row and the
   numbers explaining it are in one document.

The workload result table looks like this:

```
  measure                                        value  unit
  --------------------------------------------  ------  ------
  requests completed                             8,192
  benchmark duration (timed requests)      4 mins 54 secs
  tokens in (prompt)                         8,388,608  tokens
  tokens out (generated)                     1,048,576  tokens
  throughput (requests)                          27.78  req/s
  throughput (prompt+generated)              32,007.69  tok/s
  throughput (generated only)                 3,556.41  tok/s
  TTFT mean                                  29,604.82  ms
  TTFT p99                                   46,727.30  ms
  TPOT mean                                      30.22  ms
  TPOT p99                                       42.27  ms
  container wall time (incl. startup/teardown)  5 mins 55 secs
```

The NCCL coverage table states, per metric, whether it rose:

```
  metric                                             baseline             current    delta  status
  ------------------------------------------------  ------------------  ------------------  -------  ---------
  nccl_profiler_collective_bytes_total              28,330,700,000,000  29,691,700,000,000   +4.80%  rose
  nccl_profiler_collective_count_sum                        10,785,700          12,763,800  +18.34%  rose
  nccl_profiler_rank_latency_microseconds_sum                   absent              absent        -  no series
  nccl_profiler_transfer_latency_microseconds_sum              4.4126              4.4126   +0.00%  flat
```

Three statuses, needing three different fixes:

- **rose** — the metric increased; profiler and pipeline both working.
- **flat** — the series is being scraped but did not move. The exporter is alive; either this
  metric's instrumentation is not recording or the workload does not exercise it.
- **no series** — Prometheus has no series under that name. Nothing was ever exported: check
  the profiler plugin, the OTLP endpoint and the collector.

### `sweep_results.csv`

One row per run — the table view behind every chart, and the file to hand to a spreadsheet.
Written even when a sweep aborts, since the rows that ran are still hours of cluster time. Its
measurement columns are:

```
requests_completed  benchmark_duration_s  tokens_in  tokens_out
req_per_s  total_tok_per_s  output_tok_per_s
ttft_mean_ms  ttft_p99_ms  tpot_mean_ms  tpot_p99_ms  wall_time_s
```

```csv
max_concurrency,num_prompts,num_warmups,timeouts.workload,report,status,requests_completed,benchmark_duration_s,tokens_in,tokens_out,req_per_s,total_tok_per_s,output_tok_per_s,ttft_mean_ms,ttft_p99_ms,tpot_mean_ms,tpot_p99_ms,wall_time_s
1,64,2,1800,..._c1_p64_w2_t1800.html,passed,64,250.0,65536,8192,0.256,295.0,32.8,587.5,753.0,26.1,29.8,268.0
16,128,32,1800,..._c16_p128_w32_t1800.html,passed,128,33.7,131072,16384,3.799,4376.6,486.3,659.1,1063.8,27.2,27.9,51.7
256,2048,512,3000,..._c256_p2048_w512_t3000.html,passed,2048,81.0,2097152,262144,25.15,28971.2,3219.0,5637.0,9505.9,29.5,30.9,110.0
1024,8192,2048,5400,..._c1024_p8192_w2048_t5400.html,passed,8165,294.0,8360960,1045120,27.78,32007.7,3556.4,29604.8,46727.3,30.2,42.3,355.0
2048,16384,4096,7200,..._c2048_p16384_w4096_t7200.html,passed,16384,807.0,16777216,2097152,20.31,23393.1,2599.2,90931.9,97889.3,28.9,31.0,946.0
```

*(Values illustrative — the shape is what matters: throughput rising, plateauing, then falling
while TTFT climbs.)*

Two clocks are recorded and they mean different things. `benchmark_duration_s` is
`benchmark_serving.py`'s own timed request phase; `wall_time_s` is the whole containerised run —
image start, model connect, warm-ups, the benchmark, teardown. Throughput is computed against
the former. The gap between them is your per-run overhead.

---

## The charts

Each is a single measure on a single y-axis. Concurrency axes are log2 because the table doubles
per row; a linear axis crushes every early row into the origin.

### `throughput_generated_vs_concurrency.png`
Generated tokens/second against `max_concurrency`. **The primary chart.** The curve rises, bends,
and flattens; the bend is where the backend saturates. If it *falls* after the bend you are past
saturation and into queueing — that is the point to stop, not to push through.

### `throughput_total_vs_concurrency.png`
The same, counting prompt + generated tokens. Separate from the chart above rather than a second
series on it: with fixed input/output lengths, total is exactly `generated × (in+out)/out` — a
constant 9× at 1024/128 — so sharing an axis flattens the generated curve against the baseline
and hides the knee. Read this one when comparing against prefill-heavy figures elsewhere.

### `throughput_generated_vs_num_prompts.png` / `throughput_total_vs_num_prompts.png`
The same two measures against sample size. In a table where `num_prompts` tracks concurrency
these mirror the concurrency charts. They earn their place when concurrency is **held** and only
the sample grows: a flat line means the measurement is stable at that operating point, and a
drift means the run is not in steady state — the sample is too short, or something is degrading
over time.

### `ttft_vs_concurrency.png`
Time to first token, mean and p99, on a log y-axis (it spans milliseconds to minutes across a
full sweep). This is the *cost* of the throughput above. TTFT climbing steeply while throughput
flattens is the definition of saturation: extra concurrency is buying queue depth, not work.

### `tpot_vs_concurrency.png`
Time per output token, mean and p99. Distinguishes two failure modes that look alike in TTFT: if
TPOT stays flat while TTFT explodes, requests are queueing but decode is healthy — the batch is
full and admission is the bottleneck. If TPOT *also* rises, decode itself is degrading, which
points at KV-cache pressure or preemption rather than admission.

### `request_throughput_vs_concurrency.png`
Completed requests/second. Tracks token throughput when output lengths are uniform; diverges
when they are not, and is the number to quote for request-oriented SLOs.

### `scaling_efficiency.png`
Generated tokens/second **per request in flight** — throughput divided by concurrency. Flat means
concurrency is buying throughput linearly. The point where it starts falling is the knee, stated
more sharply than the throughput curve shows it, because a gentle bend there is a clear break
here.

### `throughput_vs_ttft_p99.png`
Throughput against tail latency, log-log, one point per row labelled with its concurrency. **The
operating-point chart.** Pick a tail-latency budget on the x-axis and read off the best
throughput available within it, and which concurrency delivers it. Points crowd vertically at the
left (throughput rising for free) then turn right (latency rising for nothing) — the corner is
where to run.

---


## Troubleshooting

**`FAIL: request failure rate N% exceeds 5% threshold`** — `benchmark_serving.py`'s own gate, not
the suite's. That row offered more load than the deployment could serve and requests timed out
in the queue. It is a legitimate result: that row is past capacity. Use `--keep-going` to record
it and continue.

The signature to check is TPOT: if TPOT is unchanged from lower-concurrency rows while TTFT and
end-to-end latency are in the tens of seconds or minutes, requests were queueing rather than
failing to generate, and the slowest fraction exceeded the client's per-request deadline — the
deployment saturates below that point. Confirm in the row's `.log` that there are no connection
resets, 5xx responses or transport errors; those indicate a proxy or KV-transfer fault rather
than simple queueing, which is a different and more serious problem.

**`unknown profile 'profiler_otel/profiles/<my_cluster>'`** — `--workload-profile` takes a
*name* (`<my_cluster>`), not a path.

**`vLLM server not ready within timeout`** — the readiness probe is pointed at
`endpoint.host`/`endpoint.port` from the profile, which defaults to `localhost`. Set those to the
serving node, or override with `VLLM_HOST`/`VLLM_PORT` passed at the command line.

**`expected N GPU(s) doing work, saw M`** — coverage is an exact match. Either the profile's
`coverage.gpus` is wrong, or the deployment is using a different number of GPUs than declared.

**`GPU driver: not collected -- ssh ...: Permission denied`** — set `GPU_INFO_SSH_USER`, or add
the login to `~/.ssh/config`. Everything else in the report still works.

**Row 12 of a long sweep fails ten hours in** — always start with `--dry-run`, then run rows 1–6
to confirm the shape before committing to the tail. `--start N` resumes; `--keep-going` survives
isolated failures.
