# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

# Add Ray, OpenTelemetry, and NCCL OTEL Profiler Plugin to the official vLLM OpenAI image

# To build:
#   docker build -t openmosaic/mosaic-vllm:dev -f Dockerfile .
# CUDA 13.0 base (vllm/vllm-openai:*-cu130):
#   docker build -t openmosaic/mosaic-vllm:cu130 --build-arg VLLM_BASE_TAG=v0.19.0-cu130 -f Dockerfile .
# To build with specific NCCL tag:
#   docker build -t openmosaic/mosaic-vllm:dev --build-arg NCCL_TAG=<nccl-tag> -f Dockerfile .
# To run:
#   docker run -it --rm openmosaic/mosaic-vllm:dev ray ...

ARG NCCL_TAG=2.29.2-1
ARG OTEL_CPP_VERSION=v1.25.0
ARG VLLM_BASE_TAG=v0.19.0

# =============================================================================
# vLLM base + build OTel, NCCL, plugin + install Ray
# =============================================================================
FROM vllm/vllm-openai:${VLLM_BASE_TAG} AS builder
ARG NCCL_TAG
ARG OTEL_CPP_VERSION

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# --- Install all build dependencies (OTel, NCCL, plugin) ---
RUN apt-get update && apt-get install -y \
      libssl-dev \
      libgrpc++-dev \
      protobuf-compiler-grpc \
      cmake \
      build-essential \
      git \
      python3 && \
    rm -rf /var/lib/apt/lists/*

# --- Build OpenTelemetry C++ SDK ---
WORKDIR /tmp
RUN git clone --branch ${OTEL_CPP_VERSION} --depth 1 https://github.com/open-telemetry/opentelemetry-cpp.git && \
    cd opentelemetry-cpp && \
    mkdir build && cd build && \
    cmake \
      -DCMAKE_INSTALL_PREFIX=/tmp/opentelemetry-build \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_SHARED_LIBS=ON \
      -DWITH_OTLP_GRPC=ON \
      -DWITH_OTLP_HTTP=ON \
      -DBUILD_TESTING=OFF \
      -DOPENTELEMETRY_INSTALL=ON \
      -DWITH_PROMETHEUS=ON \
      -DWITH_BENCHMARK=OFF \
      .. && \
    cmake --build . --parallel $(nproc) && \
    cmake --install .

# --- Build NCCL ---
RUN git clone --branch v${NCCL_TAG} --depth 1 https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    make -j"$(nproc)" src.build NVCC_GENCODE+=-Wno-deprecated-gpu-targets

# --- Build NCCL Profiler OTEL Plugin ---
ENV NCCL_PATH=/tmp/nccl
ENV OTEL_PATH=/tmp/opentelemetry-build

COPY . /tmp/nccl-profiler-otel-plugin/
RUN cd /tmp/nccl-profiler-otel-plugin && /bin/bash -lc 'make clean && make TRACE=1 NCCL_PATH=/tmp/nccl'

FROM vllm/vllm-openai:${VLLM_BASE_TAG} AS runner
ARG NCCL_TAG
ARG OTEL_CPP_VERSION
ARG VLLM_BASE_TAG

# --- Upgrade pip ---
RUN pip install --upgrade pip

# --- Upgrade packages to the latest version ---
RUN apt-get update && apt-get upgrade -y

# --- Manually install Ray dependencies to avoid CVE-2026-0994 ---
RUN pip install --upgrade protobuf==6.33.5

# --- Manually install cryptography and pillow to avoid CVE-2026-26007, CVE-2026-25990 ---
# FIXME: Remove this once components are updated to the latest version
RUN pip install cryptography==46.0.5 pillow==12.1.1

# --- Install Ray (Python) ---
RUN pip install "ray[default]==2.54.0"

RUN mkdir -p /usr/local/plugins/lib

COPY --from=builder /tmp/opentelemetry-build/lib/* /usr/local/lib/
COPY --from=builder /tmp/nccl/build/lib/* /usr/local/plugins/lib/
COPY --from=builder /tmp/nccl-profiler-otel-plugin/build/libnccl-profiler-otel.so /usr/local/plugins/lib/

RUN ( ldconfig || true )

# --- Set environment variables ---
ENV NCCL_VERSION=${NCCL_TAG}
ENV LD_LIBRARY_PATH=/usr/local/plugins/lib:/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
ENV NCCL_PROFILER_PLUGIN=otel

WORKDIR /
ENTRYPOINT ["/bin/bash", "-c"]

# Set by workflow
#
# Note: ARG GIT_HASH is placed here (not at the top) to maximize Docker layer caching.
ARG GIT_HASH=unknown
LABEL maintainer="Mosaic Project <mosaic@openmosaic.ai>"
LABEL org.opencontainers.image.title="Mosaic Reference Implementation"
LABEL org.opencontainers.image.description="vLLM OpenAI with Mosaic Profiler"
LABEL org.opencontainers.image.url="https://github.com/open-mosaic/mosaic"
LABEL org.opencontainers.image.source="https://github.com/open-mosaic/mosaic"
LABEL org.opencontainers.image.revision="${GIT_HASH}"
LABEL ai.openmosaic.image.nccl.version="${NCCL_TAG}"
LABEL ai.openmosaic.image.otel.version="${OTEL_CPP_VERSION}"
LABEL ai.openmosaic.image.vllm.base="${VLLM_BASE_TAG}"
