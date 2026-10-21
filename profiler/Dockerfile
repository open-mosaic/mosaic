# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

# Add Ray, OpenTelemetry, and NCCL OTEL Profiler Plugin to the official vLLM OpenAI image

# To build:
#   docker build -t openmosaic/mosaic-vllm:dev -f Dockerfile .
# The default vLLM base (vllm/vllm-openai:<version>) is CUDA 13.0.
# CUDA 12.9 base (vllm/vllm-openai:*-cu129):
#   docker build -t openmosaic/mosaic-vllm:cu129 --build-arg VLLM_BASE_TAG=v0.32.0-cu129 -f Dockerfile .
# To build with specific NCCL tag:
#   docker build -t openmosaic/mosaic-vllm:dev --build-arg NCCL_TAG=<nccl-tag> -f Dockerfile .
# To run:
#   docker run -it --rm openmosaic/mosaic-vllm:dev ray ...

ARG NCCL_TAG=2.32.3-1
ARG OTEL_CPP_VERSION=v1.25.0
ARG VLLM_BASE_TAG=v0.32.0

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

# --- Runtime libraries for the profiler plugin (protobuf comes from libgrpc++-dev in the builder) ---
RUN apt-get install -y --no-install-recommends libprotobuf32t64 && \
    rm -rf /var/lib/apt/lists/*

# --- Manually install Ray dependencies to avoid CVE-2026-0994 ---
RUN pip install --upgrade protobuf==6.33.5

# --- Manually install cryptography to avoid CVE-2026-69247, CVE-2026-69249, GHSA-537c-gmf6-5ccf ---
# FIXME: Remove this once components are updated to the latest version
RUN pip install cryptography==50.0.2

# --- Additional forced dependency upgrades for CVEs: 2026-53539, 2026-97687, 2026-97689, and advisories: GHSA-4w2j-m93h-cj5j (uv), GHSA-6q48-6c7w-9gw2 (certifi) ---
# FIXME: Remove this once components are updated to the latest version
RUN pip install --upgrade python-multipart==0.0.32 urllib3==2.8.0 uv==0.13.4 certifi==2026.10.19

# --- Install Ray (Python) ---
RUN pip install "ray[default]==2.59.0"

# --- Pin the pip NCCL wheel (nvidia-nccl-cu12/cu13) to match the NCCL built above ---
# torch loads its own pip NCCL, which may take precedence over /usr/local/plugins/lib.
# --no-deps leaves torch untouched; pip will warn that torch pins an older NCCL.
RUN NCCL_PKG=$(pip list --format=freeze | grep -o '^nvidia-nccl-cu[0-9]*') && \
    test -n "${NCCL_PKG}" && \
    pip install --no-deps "${NCCL_PKG}==${NCCL_TAG%-*}"

RUN mkdir -p /usr/local/plugins/lib

COPY --from=builder /tmp/opentelemetry-build/lib/* /usr/local/lib/
COPY --from=builder /tmp/nccl/build/lib/* /usr/local/plugins/lib/
COPY --from=builder /tmp/nccl-profiler-otel-plugin/build/libnccl-profiler-otel.so /usr/local/plugins/lib/

RUN ( ldconfig || true )

# --- Fail the build if the plugin has unresolved shared libraries ---
RUN ! LD_LIBRARY_PATH=/usr/local/plugins/lib ldd /usr/local/plugins/lib/libnccl-profiler-otel.so | grep "not found"

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
