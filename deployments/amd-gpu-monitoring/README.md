<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# AMD GPU Monitoring

This directory contains a Docker Compose configuration that provides additional metrics sources specifically designed to analyze AMD GPU activity.

## Overview

The `docker-compose.yml` file orchestrates multiple monitoring services that collect and expose metrics related to AMD GPU performance, device status, PCIe activity, and system-level metrics. These metrics sources can be scraped by monitoring systems (such as Prometheus) to provide comprehensive visibility into AMD GPU workloads and system behavior.

## Services

### amd-device-exporter
- **Image**: `rocm/device-metrics-exporter:v1.4.0`
- **Port**: `5053`
- **Profile**: `amd-device-exporter` (use `--profdile amd-device-exporter` to enable)
- **Purpose**: Exposes AMD device-specific metrics from ROCm-enabled GPUs
- **Device Access**: Requires access to `/dev/kfd` and `/dev/dri` for GPU communication
- **Note**: This service is only started when using the `amd-device-exporter` profile
- **Source**: https://github.com/ROCm/device-metrics-exporter

### gpu-pcie-exporter
- **Build**: `open-mosaic/gpu-pcie-exporter:latest`
- **Port**: `5052`
- **Purpose**: Provides information to map hostname and GPU ID to PCIe port number
- **Device Access**: Requires access to `/dev/kfd` and `/dev/dri`
- **Dependencies**: Mounts `/opt/rocm` from the host to access `rocm-smi` tools
- **Source**: [gpu_pcie_exporter](../gpu_pcie_exporter/README.md)

### node-exporter
- **Image**: `prom/node-exporter:v1.10.2`
- **Port**: `9100`
- **Profile**: `node-exporter` (use `--profile node-exporter` to enable)
- **Purpose**: Collects system-level metrics (CPU, memory, disk, network, etc.)
- **Note**: This service is only started when using the `node-exporter` profile
- **Source**: https://github.com/prometheus/node_exporter

### process-exporter
- **Image**: `ncabatoff/process-exporter:v0.8.7`
- **Port**: `9256`
- **Purpose**: Exports process-level metrics for detailed workload analysis
- **Configuration**: Uses `process-exporter.config.yml` for process filtering
- **Source:** https://github.com/ncabatoff/process-exporter

## Usage

To start services:

```bash
docker compose up -d
```

To start with the node-exporter (node-exporter profile):

```bash
docker compose --profile=node-exporter up -d
```

To start with the node-exporter and amd-device-exporter (node-exporter and amd-device-exporter profiles):

```bash
docker compose --profile=node-exporter --profile=amd-device-exporter up -d
```

To stop all services:

```bash
docker compose down
```

## Metrics Endpoints

Once running, metrics are available at:
- AMD Device Metrics: `http://localhost:5053/metrics` (if enabled)
- GPU PCIe Metrics: `http://localhost:5052/metrics`
- Node Metrics: `http://localhost:9100/metrics` (if enabled)
- Process Metrics: `http://localhost:9256/metrics`

## Requirements

- Docker and Docker Compose
- AMD GPUs with ROCm support
- Access to `/dev/kfd` and `/dev/dri` devices
- ROCm installation at `/opt/rocm` (for gpu-pcie-exporter)

## Integration

These metrics endpoints can be configured as scrape targets in Prometheus or other monitoring systems to collect and analyze AMD GPU activity over time.
