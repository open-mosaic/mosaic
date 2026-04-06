#!/bin/bash
# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
#
# Generate Prometheus file service discovery config for a single host.
#
# Usage: ./file-sd-config-generate.sh HOST_NAME [OPTIONS]
#
# Required:
#   HOST_NAME              Host name (used for labels and for targets if ip_address not set)
#
# Optional:
#   -i, --ip-address ADDR  Use this address for targets instead of host_name
#   --gpu-exporter-port N  GPU exporter port (default: 9400)
#   --node-exporter-port N Node exporter port (default: 9100)
#   --process-exporter-port N  Process exporter port (default: 9256)
#   --vllm-port N          vLLM Prometheus metrics port (default: 8100)
#   -o, --output FILE      Output file (default: <script_dir>/<HOST_NAME>.yaml)
#   --help                 Show this help message

set -e

# Default ports
DEFAULT_GPU_EXPORTER_PORT=9400
DEFAULT_NODE_EXPORTER_PORT=9100
DEFAULT_PROCESS_EXPORTER_PORT=9256
DEFAULT_VLLM_PORT=8100

usage() {
    sed -n '5,19p' "$0"
    exit 0
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Required
HOST_NAME=""
# Optional
IP_ADDRESS=""
GPU_EXPORTER_PORT="$DEFAULT_GPU_EXPORTER_PORT"
NODE_EXPORTER_PORT="$DEFAULT_NODE_EXPORTER_PORT"
PROCESS_EXPORTER_PORT="$DEFAULT_PROCESS_EXPORTER_PORT"
VLLM_PORT="$DEFAULT_VLLM_PORT"
OUTPUT_FILE=""

# Parse positional HOST_NAME first, then options
if [[ $# -eq 0 ]]; then
    echo "Error: HOST_NAME is required"
    usage
fi
if [[ "$1" == "--help" ]]; then
    usage
fi
HOST_NAME="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--ip-address)
            IP_ADDRESS="$2"
            shift 2
            ;;
        --gpu-exporter-port)
            GPU_EXPORTER_PORT="$2"
            shift 2
            ;;
        --node-exporter-port)
            NODE_EXPORTER_PORT="$2"
            shift 2
            ;;
        --process-exporter-port)
            PROCESS_EXPORTER_PORT="$2"
            shift 2
            ;;
        --vllm-port)
            VLLM_PORT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Target address: ip_address if set, else host_name
TARGET_ADDR="${IP_ADDRESS:-$HOST_NAME}"

# Default output to <script_dir>/<HOST_NAME>.yaml
OUTPUT_FILE="${OUTPUT_FILE:-$SCRIPT_DIR/${HOST_NAME}.yaml}"

# Emit one target group in YAML
emit_target() {
    local port="$1"
    local job="$2"
    local extra_labels="${3:-}"

    echo "- targets:"
    echo "  - \"${TARGET_ADDR}:${port}\""
    echo "  labels:"
    echo "    job: ${job}-${HOST_NAME}"
    echo "    host: ${HOST_NAME}"
    if [[ -n "$extra_labels" ]]; then
        echo "$extra_labels"
    fi
}

# Generate the YAML file
{
    # gpu_exporter
    emit_target "$GPU_EXPORTER_PORT" "gpu_exporter" "    __scrape_interval__: \"10s\"
    __scrape_timeout__: \"5s\""

    # node_exporter
    emit_target "$NODE_EXPORTER_PORT" "node_exporter"

    # process_exporter
    emit_target "$PROCESS_EXPORTER_PORT" "process_exporter"

    # vllm (Prometheus /metrics on the API port)
    emit_target "$VLLM_PORT" "vllm" "    __scrape_interval__: \"10s\"
    __scrape_timeout__: \"5s\""
} > "$OUTPUT_FILE"

echo "Generated: $OUTPUT_FILE"
