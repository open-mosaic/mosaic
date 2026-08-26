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
GPU running slower than its peers, and the collector going away. They are useful on their own
for a human-monitored cluster, and they are also what triggers the
[Fault Detective](mosaic-detective.md).

## Treat these as a starting point

Nothing from the alert payload reaches the agent. Kowalski takes the rule title for posting the
alert to the user and discards the rest, then runs the same investigation whatever fired it,
querying Prometheus from scratch. These three are examples, built to give Kowalski something to
react to. Any rule that reliably says something is wrong will trigger it, so change the
expressions, retune the thresholds, or write your own for faults specific to your hardware,
following the [Grafana alerting documentation](https://grafana.com/docs/grafana/latest/alerting/).

What is worth keeping is the shape. Two of the three compare the cluster against itself. A
threshold like "alert below 60 MB/s" only works on the cluster it was measured on; a ratio
between a recent window and an older one asks whether things got worse, which you can ask
anywhere.

The rules and the contact point are in `tests/mosaic-detective/grafana/`, ready to drop into
`/etc/grafana/provisioning/alerting/`. The thresholds in them came from the reference cluster
and are the part you have to set yourself.

## Rule 1: throughput degradation

```promql
sum(rate(nccl_profiler_collective_bytes_total[30s]))
  /
sum(rate(nccl_profiler_collective_bytes_total[20m] offset 3m))
```

Condition: below `0.85`. Pending period: `10s`.

Recent throughput over the last 30 seconds, divided by a 20 minute baseline taken from before
the fault would have started. The `offset 3m` matters. Without it a slow degradation drags the
baseline down with it and the ratio never moves.

This is the rule that catches network faults, and it is the one most worth tuning. A healthy
cluster is not flat: collective batching produces regular oscillation, and on the reference
cluster the healthy per-rank rate swings by roughly a third in normal operation. Set the
threshold too close to 1.0 and that oscillation fires it.

## Rule 2: GPU clock divergence

```promql
max(DCGM_FI_DEV_SM_CLOCK) / min(DCGM_FI_DEV_SM_CLOCK)
```

Condition: above `1.1`.

The fastest GPU divided by the slowest. Under a synchronous collective every rank should clock
similarly, so a persistent spread means one is throttling or clamped.

This comes from DCGM rather than the profiler, and that is the point: on a network-bound
cluster a clock clamp barely moves throughput, so the profiler cannot see it.

One failure mode worth guarding. With no workload running the clocks idle at different rates
and the ratio spikes, so the rule fires on every startup and shutdown. The provisioned rule
guards against this with a floor:

```promql
(max(DCGM_FI_DEV_SM_CLOCK) / min(DCGM_FI_DEV_SM_CLOCK))
  and on() (min(DCGM_FI_DEV_SM_CLOCK) > 500)
```

It also uses a pending period of `30s`, so a single unlucky evaluation during a job transition
does not fire.

## Rule 3: collector down

```promql
up{job="otel-collector"}
```

Condition: below `1`.

The simplest of the three, and it prevents the worst misdiagnosis. If the collector dies every
profiler metric stops, which looks exactly like the cluster dying. Without this rule you cannot
tell a broken cluster from a broken view of a healthy one.

A pending period of `30s` avoids firing on a scrape blip.

## Choosing your thresholds

The numbers above came from a two node, four GPU cluster on 1 GbE. Run your workload with
nothing injected for half an hour and watch what healthy does:

- Throughput: watch the ratio expression itself. It should sit near 1.0. Set the threshold
  below the deepest healthy dip, with room to spare.
- Clocks: watch the max/min spread under load and set the threshold above it.
- Collector: nothing to calibrate. It is up or it is not.

You are looking for the variance, not the average. A single snapshot tells you nothing about
how far healthy wanders.

## Evaluation and delivery

Put all three in one group. The reference setup uses `mosaic-10s` evaluating every `10s`, fast
enough that detection latency comes from the scrape interval rather than from Grafana.

Each rule routes straight to its contact point through its own notification settings, so you do
not need a notification policy tree:

| Setting | Value | Why |
|---|---|---|
| Group wait | `10s` | How long before the first notification is sent. The default is much longer |
| Group interval | `3m` | Minimum gap between notifications for the same group |

Group wait is the one that catches people. Left at the default, an alert that has already fired
takes minutes to arrive, which looks like the alert not working.

Group interval does the same job as the detective's own cooldown, which also defaults to three
minutes. Sending to a human, tune this one. Sending to the detective, either suppresses the
repeats and you only need to think about one.

The contact point is covered in [Fault Detective](mosaic-detective.md). Address the receiver by
container name, not by IP, and use the **Test** button.

## Two things to know

**A workload that stops entirely may not alert.** If the process exits, numerator and
denominator both fall to zero and the ratio is undefined rather than low. With `noDataState` set
to `NoData` that stays quiet. Test it on your own setup and decide whether you want a separate
rule.

**Provisioned rules are read-only in the UI.** Alerting provisioning is also not re-read from
disk on an interval the way dashboard provisioning is, so a changed file needs a Grafana restart
or `POST /api/admin/provisioning/alerting/reload`. That only bites during calibration, when you
want to nudge a number and watch the effect; build the rules in the UI first if you would rather
tune them interactively, then export and provision from the result.

Rules built in the UI live in Grafana's database and go with the volume, so keep an export
outside the cluster and redo it after any change. A stale export is worse than none, because you
will trust it.
