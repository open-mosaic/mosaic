<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# GPU PCIe Exporter

Prometheus exporter for GPU to PCIe port mapping. Discovers GPUs using `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD) and exposes metrics with labels for host, gpu_id, gpu_uuid, pcie_port, and vendor. The PICe exporter provides mapping information so GPUs can be located at a physical PCIe slot on a host. The gpu_id and gpu_uuid values provided by the AMD and NVIDIA services do not correspond to a physically identifying value, but the PCIe bus ID does.

## Building the Docker Image

From the `gpu_pcie_exporter` directory:

```bash
docker build -t gpu-pcie-exporter .
```

Or from the parent directory using docker-compose:

```bash
docker compose build gpu-pcie-exporter
```

**Note:** If you encounter import errors after updating dependencies, rebuild without cache:

```bash
docker compose build --no-cache gpu-pcie-exporter
```

## Running

### Using Docker Compose

The service is configured in `amd-gpu-monitoring/docker-compose.yml`. Start it with:

```bash
docker compose up -d gpu-pcie-exporter
```

### Using Docker Directly

```bash
docker run -d \
  --name gpu-pcie-exporter \
  -p 8000:8000 \
  -v /opt/rocm:/opt/rocm:ro \
  -v /usr/bin:/host/usr/bin:ro \
  -e PATH=/opt/rocm/bin:/usr/bin:/usr/local/bin:/usr/sbin:/bin:/sbin \
  -e LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib:/usr/local/lib \
  gpu-pcie-exporter
```

## Accessing Metrics

Once running, metrics are available at:

```
http://localhost:8000/metrics
```

## Configuration

Command-line options:

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | 8000 | Port for the Prometheus metrics HTTP server |
| `--update-interval` | 60 | Seconds between refreshing GPU mappings |
| `--max-uptime` | 3600 | Exit after this many seconds so the container is restarted by Docker (e.g. to avoid the exporter stopping reporting metrics). Use `0` to disable. |
| `--test` | — | Print GPU mappings and exit (no server) |

The exporter exits after `--max-uptime` seconds (default: 1 hour). With Docker’s `restart: always` policy, the container is restarted automatically, which keeps the `gpu_pcie_port` metric reporting reliably.

To customize options when using Docker Compose, override the command in `docker-compose.yml`, for example:

```yaml
command: ["python3", "gpu_pcie_exporter.py", "--max-uptime", "7200"]
```

## Requirements

The container needs access to:
- `/opt/rocm/bin/rocm-smi` (for AMD GPUs)
- `/usr/bin/nvidia-smi` (for NVIDIA GPUs)

These are mounted from the host system via volumes in docker-compose.yml.


### Example docker-compose when monitoring NVIDIA GPUs via `nvidia-smi`:

```yaml
gpu-pcie-exporter:
  build:
    context: .
    dockerfile: Dockerfile
  container_name: gpu-pcie-exporter
  ports:
    - "5052:8000"
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
  volumes:
    - /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro
  restart: always
```

### Example docker-compose when monitoring AMD GPUs via `rocm-smi`:

```yaml
gpu-pcie-exporter:
  build:
    context: .
    dockerfile: Dockerfile
  container_name: gpu-pcie-exporter
  ports:
    - "5052:8000"
  devices:
    - /dev/kfd
    - /dev/dri
  volumes:
    - /opt/rocm:/opt/rocm:ro
  environment:
    - PATH=/usr/local/bin:/opt/rocm/bin:/usr/bin:/usr/sbin:/bin:/sbin
    - LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib:/usr/local/lib
  restart: always
```
