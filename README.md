<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# Mosaic: Always-on GPU Collective Observability

[![Integration Test](https://github.com/open-mosaic/mosaic/actions/workflows/integration-tests.yml/badge.svg?branch=main)](https://github.com/open-mosaic/mosaic/actions/workflows/integration-tests.yml)
[![Unit Test](https://github.com/open-mosaic/mosaic/actions/workflows/nccl-profiler-plugin.yml/badge.svg?branch=main)](https://github.com/open-mosaic/mosaic/actions/workflows/nccl-profiler-plugin.yml)
[![codecov](https://codecov.io/gh/open-mosaic/mosaic/graph/badge.svg?token=58K1U9KEWD)](https://codecov.io/gh/open-mosaic/mosaic)

Mosaic is an open-source always-on observability tool for GPU collective communication,
providing near real-time visibility into performance and reliability issues in large-scale AI workloads.
It treats collective communication as first-class OpenTelemetry data, enabling correlation with GPU, network, and system signals in a single view.

At GPU scale, where failures and inefficiencies are inevitable,
Mosaic makes these issues visible early, shifting observability from postmortem analysis to continuous operations.

No offline tracing, no bespoke pipelines, and no invasive instrumentation.

### Getting Started
To get Mosaic up and running in your environment, follow our [Quick Start Guide](https://openmosaic.ai/latest/quickstart/).

### Documentation
For a deep dive into core concepts, architecture, and advanced configuration, visit [Mosaic Documentation](https://openmosaic.ai)
