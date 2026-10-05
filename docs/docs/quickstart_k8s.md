---
icon: fontawesome/solid/dharmachakra
title: Quick Start on Kubernetes
---

<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

This guide deploys Mosaic on a Kubernetes cluster and monitors collective metrics from a vLLM reference workload.
It is the Kubernetes equivalent of the [Quick Start](./quickstart.md), which uses docker compose.

Mosaic does not ship its own Helm chart.
It uses upstream charts configured by the values files in `deployments/k8s/`:

| Component | Chart | Values file |
|-----------|-------|-------------|
| LGTM stack (`grafana/otel-lgtm`) and pipeline analyzer | [`bjw-s/app-template`](https://bjw-s-labs.github.io/helm-charts/docs/app-template/) | `otel-lgtm-values.yaml` |
| vLLM with the Mosaic profiler plugin | [`vllm/vllm-stack`](https://github.com/vllm-project/production-stack) | `vllm-values.yaml` |

# Prerequisites

## Hardware

A minimum of 2 GPUs is required to generate collective metrics.
This tutorial assumes a node equipped with 2 NVIDIA GPUs.

## Software

- A Kubernetes cluster, version 1.25 or later
- [Helm](https://helm.sh/docs/intro/install/) 3 or later
- [kubectl](https://kubernetes.io/docs/tasks/tools/) configured for the cluster
- The [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html), which provides the GPU device plugin and the `nvidia` RuntimeClass

# Environment Setup

``` bash title="Clone the Mosaic repository"
git clone https://github.com/open-mosaic/mosaic.git
cd mosaic
```

``` bash title="Add the Helm repositories"
helm repo add bjw-s https://bjw-s-labs.github.io/helm-charts
helm repo add vllm https://vllm-project.github.io/production-stack
helm repo update
```

``` bash title="Create the namespace"
kubectl create namespace mosaic
```

# Launch the LGTM Stack

The LGTM (Loki, Grafana, Tempo, Mimir) configuration and the Grafana dashboards are loaded from ConfigMaps built from the files in `deployments/`, the same files that docker compose uses.

``` bash title="Create the configuration ConfigMaps"
kubectl -n mosaic create configmap mosaic-otel-lgtm-config \
  --from-file=otelcol-config.yaml=deployments/otel-collector-config.yaml \
  --from-file=prometheus.yaml=deployments/k8s/prometheus-k8s.yaml \
  --from-file=deployments/loki-config.yaml \
  --from-file=deployments/recording-rules.yaml \
  --from-file=deployments/grafana.ini \
  --from-file=deployments/dashboards/dashboards.yml \
  --dry-run=client -o yaml | kubectl apply --server-side -f -

for f in deployments/dashboards/*.json; do
  name="mosaic-dashboard-$(basename "$f" .json | tr _ -)"
  kubectl -n mosaic create configmap "$name" --from-file="$f" \
    --dry-run=client -o yaml | kubectl apply --server-side -f -
done
```

!!! note
    Each dashboard has its own ConfigMap to stay under the 1 MiB ConfigMap size limit.
    `--server-side` is required because some dashboards are too large for the annotation that a client-side `kubectl apply` adds.
    When adding a dashboard, also add its ConfigMap to the `dashboards` volume in `deployments/k8s/otel-lgtm-values.yaml`.

``` bash title="Install the LGTM stack"
helm install mosaic-otel-lgtm bjw-s/app-template --version 5.2.1 \
  -n mosaic -f deployments/k8s/otel-lgtm-values.yaml
```

The pipeline analyzer runs as a sidecar in the same pod.
It reads the OTel collector metrics, and is scraped by Prometheus, over `localhost`.

The service is named `mosaic-otel-lgtm`.
Prometheus discovers pods in the `mosaic` namespace that have the `mosaic.io/scrape: "true"` annotation, so no scrape configuration needs to be generated.

!!! tip
    The ConfigMaps are mounted with `subPath`, so changes are not picked up automatically.
    After updating a ConfigMap, restart the pod with `kubectl -n mosaic rollout restart deploy/mosaic-otel-lgtm`.

# Launch vLLM

The vLLM pod serves `Qwen/Qwen3-8B` with a tensor parallel size of 2, using the `openmosaic/mosaic-vllm` image that includes the Mosaic profiler plugin.

``` bash title="Install vLLM"
helm install mosaic-vllm vllm/vllm-stack --version 0.1.13 \
  -n mosaic -f deployments/k8s/vllm-values.yaml
```

!!! tip
    For gated models, create a secret with your Hugging Face token and uncomment `hf_token` in `deployments/k8s/vllm-values.yaml`:

    ``` bash
    kubectl -n mosaic create secret generic hf-token --from-literal=token=<token>
    ```

Downloading and loading the model can take several minutes:

``` bash title="Wait for vLLM"
kubectl -n mosaic rollout status deploy/mosaic-vllm-qwen3-8b-deployment-vllm --timeout=30m
```

!!! tip
    If `kubectl -n mosaic logs deploy/mosaic-vllm-qwen3-8b-deployment-vllm` shows `Waiting for 1 local, 0 remote core engine proc(s) to start.` and does not proceed further,
    uncomment `NCCL_P2P_DISABLE` in `deployments/k8s/vllm-values.yaml` and apply it with
    `helm upgrade mosaic-vllm vllm/vllm-stack --version 0.1.13 -n mosaic -f deployments/k8s/vllm-values.yaml`.

!!! note
    The Mosaic images use the `latest` tag with `imagePullPolicy: Always`.
    Pushing a new image does not restart the pods, so restart them to pick it up:

    ``` bash
    kubectl -n mosaic rollout restart deploy/mosaic-vllm-qwen3-8b-deployment-vllm
    kubectl -n mosaic rollout restart deploy/mosaic-otel-lgtm
    ```

# Verification

## Confirm Model Status

Forward the vLLM and Grafana ports to your machine:

``` bash
kubectl -n mosaic port-forward svc/mosaic-vllm-qwen3-8b-engine-service 8080:80 &
kubectl -n mosaic port-forward svc/mosaic-otel-lgtm 3000:3000 &
```

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

# Monitor Your Own Workloads

Point the Mosaic profiler plugin of any workload in the cluster at the LGTM service:

``` bash
NCCL_PROFILER_OTEL_TELEMETRY_ENDPOINT=http://mosaic-otel-lgtm.mosaic.svc:4318
```

# Limitations

The node, process and GPU exporters are not yet deployed on Kubernetes.

Multi-node workloads need pods to reach the backend network for RDMA, usually with the [Multus](https://github.com/k8snetworkplumbingwg/multus-cni) CNI and the [SR-IOV network device plugin](https://github.com/k8snetworkplumbingwg/sriov-network-device-plugin).
Support for this is tracked in [issue #30](https://github.com/open-mosaic/mosaic/issues/30).

# Clean Up

``` bash
helm uninstall mosaic-vllm -n mosaic
helm uninstall mosaic-otel-lgtm -n mosaic
kubectl delete namespace mosaic
```
