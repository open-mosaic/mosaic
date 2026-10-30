---
icon: simple/kubernetes
title: Quick Start on Kubernetes
---

<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

This guide will demonstrate how to deploy Open Mosaic on a Kubernetes cluster and monitor collective metrics from a vLLM reference workload.
It is the Kubernetes equivalent of the [Quick Start on Docker](./quickstart.md), which uses docker compose.

!!! note
    This quick start guide is simplified for tutorial purposes and is not recommended for production use.

The Kubernetes quick start guide uses the the Grafana LGTMA stack (Loki, Grafana, Tempo, Mimir and Alloy), each component in its own pod for the observability backend.
Upstream charts are configured with values files in `deployments/k8s/`, and the pipeline analyzer is deployed with a chart from the Open Mosaic repository:

| Component | Chart | Values file |
|-----------|-------|-------------|
| Mimir (metrics) | [`grafana/mimir-distributed`](https://github.com/grafana/mimir/tree/main/operations/helm/charts/mimir-distributed) | `mimir-values.yaml` |
| Loki (logs) | [`grafana-community/loki`](https://github.com/grafana-community/helm-charts/tree/main/charts/loki) | `loki-values.yaml` |
| Tempo (traces) | [`grafana-community/tempo-distributed`](https://github.com/grafana-community/helm-charts/tree/main/charts/tempo-distributed) | `tempo-values.yaml` |
| Alloy (collector) | [`grafana/alloy`](https://github.com/grafana/alloy/tree/main/operations/helm/charts/alloy) | `alloy-values.yaml` |
| Grafana | [`grafana-community/grafana`](https://github.com/grafana-community/helm-charts/tree/main/charts/grafana) | `grafana-values.yaml` |
| Pipeline analyzer | `deployments/k8s/charts/pipeline-analyzer` | `charts/pipeline-analyzer/values.yaml` |
| vLLM with the Mosaic profiler plugin | [`vllm/vllm-stack`](https://github.com/vllm-project/production-stack) | `vllm-values.yaml` |

Workloads send OTLP telemetry to Alloy, which forwards metrics to Mimir and the pipeline analyzer, logs to Loki, and traces to Tempo.
Alloy also scrapes the pipeline analyzer and the annotated pods in the `mosaic` namespace.
Grafana reads from Mimir, Loki, and Tempo.

# Prerequisites

## Hardware

A minimum of 2 GPUs is required to generate collective metrics.
This tutorial assumes a node equipped with 2 NVIDIA GPUs.

!!! note
    Multi-node workloads that communicate over an RDMA backend network are out of scope for this tutorial.
    See [Limitations](#limitations).

## Software

- A Kubernetes cluster, version 1.25 or later, with a default StorageClass for the persistent volumes of Mimir, Loki and Tempo
- [Helm](https://helm.sh/docs/intro/install/) 3 or later
- [kubectl](https://kubernetes.io/docs/tasks/tools/) configured for the cluster
- The [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html), which provides the GPU device plugin and the `nvidia` RuntimeClass

# Environment Setup

``` bash title="Clone the Mosaic repository"
git clone https://github.com/open-mosaic/mosaic.git
cd mosaic
```

``` bash title="Add the Helm repositories"
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add grafana-community https://grafana-community.github.io/helm-charts
helm repo add vllm https://vllm-project.github.io/production-stack
helm repo update
```

``` bash title="Create the namespace"
kubectl create namespace mosaic
```

# Launch the LGTMA Stack

The configuration of each component, including the Alloy pipeline, is in its values file.
The Mimir recording rules are loaded from `deployments/recording-rules.yaml`, the same file that docker compose uses, with `--set-file`.
Install the backends first, then the pipeline analyzer and Alloy, which send data to them, and Grafana last.

``` bash title="Install Mimir, Loki and Tempo"
helm install mosaic-mimir grafana/mimir-distributed --version 6.2.1 \
  -n mosaic -f deployments/k8s/mimir-values.yaml \
  --set-file recordingRules=deployments/recording-rules.yaml

helm install mosaic-loki grafana-community/loki --version 18.15.1 \
  -n mosaic -f deployments/k8s/loki-values.yaml

helm install mosaic-tempo grafana-community/tempo-distributed --version 2.26.2 \
  -n mosaic -f deployments/k8s/tempo-values.yaml
```

!!! note
    The 3.x versions of the `tempo-distributed` chart require an external Kafka, so Tempo is pinned to the 2.x chart.

The pipeline analyzer chart is not currently published to a Helm repository.
Install it from your clone of the Mosaic repository:

``` bash title="Install the pipeline analyzer"
helm install mosaic-pipeline-analyzer ./deployments/k8s/charts/pipeline-analyzer -n mosaic
```

The pipeline analyzer pod includes an OpenTelemetry Collector sidecar.
Alloy forwards the NCCL profiler metrics to the sidecar, which exposes them in the Prometheus format that the pipeline analyzer reads over `localhost`.

``` bash title="Install Alloy"
helm install mosaic-alloy grafana/alloy --version 1.13.1 \
  -n mosaic -f deployments/k8s/alloy-values.yaml
```

Alloy receives telemetry on the `mosaic-alloy` service, on port 4317 (OTLP/gRPC) and 4318 (OTLP/HTTP).
It also scrapes pods in the `mosaic` namespace that have the `mosaic.io/scrape: "true"` annotation, so no scrape configuration needs to be generated.

## Grafana Dashboards

The Grafana dashboards are loaded from ConfigMaps, one for each dashboard in `deployments/dashboards/`, the same files that docker compose uses.

``` bash title="Create the dashboard ConfigMaps"
for f in deployments/dashboards/*.json; do
  name="mosaic-dashboard-$(basename "$f" .json | tr _ -)"
  kubectl -n mosaic create configmap "$name" --from-file="$f" \
    --dry-run=client -o yaml | kubectl apply --server-side -f -
done
```

!!! note
    Each dashboard has its own ConfigMap to stay under the 1 MiB ConfigMap size limit.
    `--server-side` is required because some dashboards are too large for the annotation that a client-side `kubectl apply` adds.
    When adding a dashboard, also add its ConfigMap to `extraContainerVolumes` and its provider to `dashboardProviders` in `deployments/k8s/grafana-values.yaml`.

``` bash title="Install Grafana"
helm install mosaic-grafana grafana-community/grafana --version 13.4.0 \
  -n mosaic -f deployments/k8s/grafana-values.yaml
```

Wait for all the pods to be ready:

``` bash
kubectl -n mosaic get pods
```

!!! tip
    To change the configuration of a component, edit its values file and apply it with `helm upgrade`, using the same arguments as `helm install`, including `--set-file` so changes to `deployments/recording-rules.yaml` are applied.
    After updating a dashboard ConfigMap, restart Grafana with `kubectl -n mosaic rollout restart deploy/mosaic-grafana`.

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

# Verification

## Confirm Model Status

Forward the vLLM and Grafana ports to your machine:

``` bash
kubectl -n mosaic port-forward svc/mosaic-vllm-qwen3-8b-engine-service 8080:80 &
kubectl -n mosaic port-forward svc/mosaic-grafana 3000:80 &
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

!!! tip
    Anonymous users can only view the dashboards.
    To edit them, sign in as `admin` with the password that the Grafana chart generated:

    ``` bash
    kubectl -n mosaic get secret mosaic-grafana -o jsonpath='{.data.admin-password}' | base64 -d; echo
    ```

!!! tip
    To check the metrics and the recording rules in Mimir directly, forward the Mimir gateway port:

    ``` bash
    kubectl -n mosaic port-forward svc/mosaic-mimir-gateway 8081:80 &
    curl -s 'localhost:8081/prometheus/api/v1/query?query=nccl_profiler_collective_count_count' | jq '.data.result | length'
    curl -s localhost:8081/prometheus/api/v1/rules | jq '.data.groups[].name'
    ```

# Monitor Your Own Workloads

Point the Mosaic profiler plugin of any workload in the cluster at the Alloy service:

``` bash
NCCL_PROFILER_OTEL_TELEMETRY_ENDPOINT=http://mosaic-alloy.mosaic.svc:4318
```

# Limitations

The node, process, and GPU exporters are not part of the Kubernetes quick start.

Multi-node workloads need pods to reach the backend network for RDMA, usually with the [Multus](https://github.com/k8snetworkplumbingwg/multus-cni) CNI and the [SR-IOV network device plugin](https://github.com/k8snetworkplumbingwg/sriov-network-device-plugin).
Support for this is tracked in [issue #30](https://github.com/open-mosaic/mosaic/issues/30).

# Clean Up

``` bash
helm uninstall mosaic-vllm -n mosaic
helm uninstall mosaic-grafana mosaic-alloy mosaic-pipeline-analyzer -n mosaic
helm uninstall mosaic-tempo mosaic-loki mosaic-mimir -n mosaic
kubectl delete namespace mosaic
```

Deleting the namespace also deletes the persistent volume claims of Mimir, Loki, and Tempo.
