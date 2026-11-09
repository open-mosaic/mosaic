---
icon: fontawesome/solid/bug-slash
title: Troubleshoot
---

<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# Common Issues

1. **Plugin not loading**:
    - Verify `NCCL_PROFILER_PLUGIN=otel` is set
    - Check that `LD_LIBRARY_PATH` includes the plugin directory
    - Ensure the plugin library exists and is executable
    - With `NCCL_DEBUG=INFO`, look for `PROFILER/Plugin: libnccl-profiler-otel.so: ... cannot open shared object file`,
      which means a library the plugin depends on is missing. Run `ldd` on the plugin to find it.

2. **No metrics being sent**:
    - Verify `NCCL_PROFILER_OTEL_TELEMETRY_ENDPOINT` is correctly set
    - Check that `NCCL_PROFILER_OTEL_ENABLE=1` (default)
    - Check that `NCCL_PROFILER_OTEL_TELEMETRY_ENABLE=1` (default)
    - Verify OpenTelemetry collector is running and accessible

3. **Connection issues**:
    - Verify the OpenTelemetry collector is running and accessible
    - Check network connectivity to the collector endpoint
    - Ensure the endpoint URL format is correct (`http://host:port`)

4. **Buffer overflow warnings**:
    - Increase window trigger settings if events are generated too quickly
    - Check that telemetry thread is processing windows in time
    - Verify system has sufficient CPU for background processing

5. **No collective metrics with vLLM 0.29 or later**:
    - vLLM has several all-reduce backends and only falls back to NCCL when none of them applies.
      `--disable-custom-all-reduce` only disables vLLM's custom all-reduce, not the backends below.
    - Since vLLM 0.29.0, FlashInfer all-reduce is enabled by default and bypasses NCCL.
      Set `VLLM_ALLREDUCE_USE_FLASHINFER=0`.
    - PyTorch symmetric-memory all-reduce is also enabled by default and bypasses NCCL on GPUs with
      compute capability 9.0, 10.0, 10.3 or 10.7 (for example H100, H200, B200, GB200, B300).
      On these GPUs, also set `VLLM_ALLREDUCE_USE_SYMM_MEM=0`.
      Check the compute capability with `nvidia-smi --query-gpu=name,compute_cap --format=csv`.
    - Routing all-reduces through NCCL can be slower than these backends for small messages.

# Debug Logging

Enable trace logging for detailed debugging:

```bash
# Build with trace logging
make TRACE=1

# Run with NCCL debug output
export NCCL_DEBUG=INFO
export NCCL_PROFILER_PLUGIN=otel
# ... other environment variables
```

This will provide detailed logs about plugin initialization, metric collection, and telemetry export.
