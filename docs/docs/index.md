---
icon: fontawesome/solid/grip
title: Overview
---

<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# Introduction

{==Open Mosaic is an open-source always-on observability tool for GPU collective communication,
providing near real-time visibility into performance and reliability issues in large-scale AI workloads.==}
It treats collective communication as first-class OpenTelemetry data, enabling correlation with GPU, network, and system signals in a single view.

At GPU scale, where failures and inefficiencies are inevitable,
Open Mosaic makes these issues visible early, shifting observability from postmortem analysis to continuous operations.

No offline tracing, no bespoke pipelines, and no invasive instrumentation.

Open Mosaic is hosted on [GitHub](https://github.com/open-mosaic/mosaic), licensed under **Apache 2.0**.

# Background

Modern AI systems don't usually fail all at once. They fail a little, all the time.
As GPU clusters scale into the hundreds or thousands of devices, partial failures, slow ranks, and network hiccups become the norm rather than the exception.
Recent large-scale studies show that {==GPU failures are frequent enough to materially impact training and inference reliability==},
especially for long-running or tightly synchronized workloads.
In practice, large jobs are statistically guaranteed to encounter hardware or communication issues before they complete.

At the same time, the economics of AI are shifting.
The cost per token drops as workloads move from dense models toward Mixture-of-Experts (MoE), but sensitivity to system-level variance increases.
MoE inference and training amplify fan-out, collective communication, and cross-node synchronization.
Research and industry benchmarks consistently show that {==communication and networking, not raw compute, become the dominant bottleneck as systems scale==}.
Even on high-bandwidth interconnects, collective operations prevent linear scale-out, leaving GPUs idle while waiting on stragglers or congested links.

The problem isn't a lack of metrics. It’s that visibility arrives too late.
Most GPU and CCL tools focus on postmortem analysis: they help explain why performance was bad after a job has already failed or burned significant compute budget.
But at modern GPU scale, observability must be continuous.
Operators has a {==need to see collective-level pathologies as they emerge, immediately correlate them with other system metrics==}, such that further recovery action can be taken.

Open Mosaic is built around this shift.
It provides always-on, near real-time observability for collective communication, treating collective communication as first-class OpenTelemetry data.
Open Mosaic streams these signals directly into a standard Grafana LGTM stack, where they can be viewed alongside other GPU and system metrics in a single dashboard.
This makes issues like slow ranks, imbalanced collectives, or network-induced stalls visible while workloads are still running.

!!! info "Key Takeaway"
    As GPU scale, failures and inefficiencies are inevitable.
    What matters is whether you can see and understand them early enough to act.
    Open Mosaic shifts GPU observability from forensic debugging to continuous operations, where real reliability gains and cost savings are made.
