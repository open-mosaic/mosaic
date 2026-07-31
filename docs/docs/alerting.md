---
icon: fontawesome/solid/bell
title: Grafana Alerting
---

<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# Grafana alerting

Three alert rules cover the fault classes the profiler can see: throughput falling off, one
GPU running slower than its peers, and the collector going away. They are useful on their
own for a human-monitored cluster, and they are also what triggers the
[Fault Detective](mosaic-detective.md).

The rules are not shipped as importable files. Datasource UIDs, folder names and org IDs
are specific to one Grafana, and the thresholds depend on your hardware, so an imported copy
of someone else's alerts is either broken or wrong for your cluster. Build them from what
follows instead. It takes a few minutes.

## The idea behind them

Two of the three compare the cluster against itself rather than against a fixed number.

A hardcoded threshold like "alert below 60 MB/s" only works on the cluster it was measured
on. A ratio between a recent window and an older one asks whether things got worse, which is
a question you can ask anywhere. The same principle runs through the detective's diagnostic
procedure, at a different resolution.

## Rule 1: throughput degradation

```promql
sum(rate(nccl_profiler_collective_bytes_total[30s]))
  /
sum(rate(nccl_profiler_collective_bytes_total[20m] offset 3m))
```

Condition: below `0.85`. Pending period: `10s`.

Recent throughput over the last 30 seconds, divided by a 20 minute baseline taken from
before the fault would have started. The `offset 3m` matters. Without it a slow degradation
drags the baseline down with it and the ratio never moves.

This is the rule that catches network faults, and it is the one most worth tuning. A healthy
cluster is not flat: collective batching produces regular oscillation, and on the reference
cluster the healthy per-rank rate swings by roughly a third in normal operation. Set the
threshold too close to 1.0 and that oscillation fires it.

## Rule 2: GPU clock divergence

```promql
max(DCGM_FI_DEV_SM_CLOCK) / min(DCGM_FI_DEV_SM_CLOCK)
```

Condition: above `1.1`.

The fastest GPU divided by the slowest. Under a synchronous collective every rank should
clock similarly, so a persistent spread means one GPU is throttling or clamped.

This comes from DCGM rather than the profiler, and that is the point. On a network-bound
cluster a clock clamp barely moves throughput at all, so the profiler cannot see it. The
rule needs a source outside the collective.

It has one failure mode worth guarding. When no workload is running the clocks idle at
different rates and the ratio spikes, so the rule fires on startup and shutdown. Add a floor:

```promql
(max(DCGM_FI_DEV_SM_CLOCK) / min(DCGM_FI_DEV_SM_CLOCK))
  and on() (min(DCGM_FI_DEV_SM_CLOCK) > 500)
```

A pending period of `30s` is also worth setting, so a single unlucky evaluation during a job
transition does not fire.

## Rule 3: collector down

```promql
up{job="otel-collector"}
```

Condition: below `1`.

The simplest of the three and the one that prevents the worst misdiagnosis. If the collector
dies, every profiler metric stops, which looks exactly like the cluster dying. Without this
rule you cannot tell the difference between a broken cluster and a broken view of a healthy
one.

A pending period of `30s` avoids firing on a scrape blip.

## Choosing your thresholds

Do not copy the numbers above without checking them. They came from a two node, four GPU
cluster on 1 GbE, and they are starting points rather than recommendations.

Run your normal workload with nothing injected for at least half an hour, then look at what
healthy actually does:

- For throughput, watch the ratio expression itself over that window. It should sit near
  1.0. Note how far it dips during normal operation and set the threshold below the
  deepest healthy dip, with room to spare.
- For clocks, watch the max/min ratio. Note the spread under load and set the threshold
  above it.
- For the collector, no calibration needed. It is up or it is not.

The variance is the thing you are looking for, not the average. A single snapshot tells you
nothing about how far healthy wanders.

## Evaluation and delivery

Put all three in one evaluation group. The reference setup uses a group named `mosaic-10s`
evaluating every `10s`, which is fast enough that detection latency is dominated by the
metric scrape interval rather than by Grafana.

Each rule routes straight to its contact point through the rule's own notification settings,
so no notification policy tree is needed:

| Setting | Value | Why |
|---|---|---|
| Group wait | `10s` | How long to wait before sending the first notification. The default is much longer |
| Group interval | `3m` | Minimum gap between notifications for the same group |

Group wait is the one that surprises people. Left at the default, an alert that has already
fired takes minutes to be delivered, which looks like the alert not working.

Group interval is doing the same job as the detective's own cooldown, which also defaults to
three minutes. If you are sending to a human, tune this one. If you are sending to the
detective, either mechanism suppresses the repeats and you only need to think about one of
them.

## Sending to a receiver

Create a webhook contact point pointing at wherever the receiver listens. If the receiver
runs as a container on the same Docker network as Grafana, address it by container name
rather than by IP, so the configuration is not tied to one machine's networking.

Use the **Test** button once it is set up. That verifies the network path without waiting
for a real fault, and it is the fastest way to find out that Grafana cannot reach the
receiver.

## Two things to know

**A workload that stops entirely may not alert.** If the process exits, the throughput
numerator and denominator both fall to zero, and the ratio becomes undefined rather than
low. With `noDataState` set to `NoData` that stays quiet. Test this case on your own setup
and decide whether you want a separate rule for it.

**File-provisioned rules are read-only in the UI.** If you provision from a file rather than
building the rules in Grafana, you cannot then tune the thresholds through the interface,
which makes calibration awkward. Build them in the UI first, tune them, then export if you
want a backup.

## Backing up your configuration

Grafana keeps alert definitions in its own database, so they disappear with the volume.
Once your rules are tuned, export them and store the result somewhere outside the cluster:

```bash
curl -s -u <user>:<pass> \
  "http://<grafana>/api/v1/provisioning/alert-rules/export?format=yaml"

curl -s -u <user>:<pass> \
  "http://<grafana>/api/v1/provisioning/contact-points/export?format=yaml"
```

If you later change a rule through the UI, re-export it. An old export that no longer
matches what is running is worse than none, because you will trust it.
