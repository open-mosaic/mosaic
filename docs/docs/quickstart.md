---
icon: fontawesome/solid/rocket
title: Quick Start
---

<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

This guide will help you set up Mosaic to monitor collective metrics across multiple GPUs using vLLM and Ray.

# Prerequisites

## Hardware

A minimum of 2 GPUs is required to generate collective metrics.
This tutorial assumes a single server equipped with 2 NVIDIA GPUs.

## Software

A Docker runtime with the NVIDIA Container Toolkit is required.

- [Install Docker Engine](https://docs.docker.com/engine/install/)
- [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

#  Environment Setup

Fetch the source repository and the pre-built images.

``` bash title="Clone the Mosaic repository"
git clone https://github.com/open-mosaic/mosaic.git
cd mosaic
```

# Generate Mosaic Metrics Scraping Configuration

We need to generate one metric scraping configuration per GPU server in production deployment.
For this quick start tutorial, we only need to do it on one host:

``` bash title="Scrape configuration from localhost"
./deployments/file_sd_configs/file-sd-config-generate.sh host -i 172.17.0.1
```

#  Launch Mosaic Infrastructure

Mosaic relies on the LGTM (Loki, Grafana, Tempo, Mimir) stack and specific hardware exporters to collect and visualize data.

``` bash title="Launch LGTM stack"
docker compose -f ./deployments/docker-compose.yml up -d
```

``` bash title="Launch GPU and System exporters"
docker compose -f ./deployments/nvidia-gpu-monitoring/docker-compose.yml \
  --profile=node-exporter \
  --profile=nvidia-device-exporter \
  --profile=process-exporter \
  up -d
```

!!! tip
    If you encounter a `buildx isn't installed` error, disable BuildKit by running `export DOCKER_BUILDKIT=0`

#  Deploy Reference Workload

The following command will launch a vLLM container with Mosaic profiler plugin and serve `Qwen/Qwen3-8B` model.
Metrics are configured to stream to Mosaic while the inference is running.

``` bash title="Launch vLLM"
docker compose -f ./deployments/docker-compose-vllm.yml up -d
```

!!! tip
    If `docker logs mosaic-vllm` shows `(APIServer pid=1) DEBUG 03-06 10:18:50 [v1/engine/utils.py:982] Waiting for 1 local, 0 remote core engine proc(s) to start.` and does not proceed further,
    try adding the following environment variable to `/deployments/docker-compose-vllm.yml`

    ```diff
        environment:
          - NCCL_PROFILER_OTEL_TELEMETRY_ENDPOINT=http://mosaic-otel-lgtm:4318
          - VLLM_LOGGING_LEVEL=DEBUG
    +      - NCCL_P2P_DISABLE=1
    ```

#  Verification

## Confirm Model Status

Verify that the model is being served correctly:

``` bash
curl -s localhost:8080/v1/models | jq '.data[].root'
```

Expected Output: `"Qwen/Qwen3-8B"`

## Verify Metrics Generation

Trigger an inference request to generate workload and populate the Mosaic metrics.

```bash
curl -s localhost:8080/v1/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-8B",
  "prompt": "Once upon a time",
  "max_tokens": 512
}' | jq
```

After the request completes, you can observe the updated metrics via Grafana dashboard at [http://localhost:3000](http://localhost:3000).
You should see metrics being reported on the dashboards. Here's an example:

![Sample Grafana Dashboard](images/dashboard-sample.png)

# Congratulations

You have successfully:

- Deployed the Mosaic observability stack.
- Launched a multi-GPU Ray cluster.
- Served a large language model using vLLM.
- Triggered and verified live metric collection.

Interested in learning more about Mosaic?
Visit [Architecture](./architecture.md) and [Profiler Design](./design.md) for more details.
