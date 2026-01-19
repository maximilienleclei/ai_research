ARG MODE=cpu
ARG IMAGE_CPU="docker.io/library/ubuntu:24.04"
ARG IMAGE_ROCM="docker.io/rocm/dev-ubuntu-24.04:6.4.4-complete"
ARG IMAGE_CUDA="docker.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04"

FROM ${IMAGE_CPU} AS base-cpu
FROM ${IMAGE_ROCM} AS base-7800xt
FROM ${IMAGE_CUDA} AS base-cuda

FROM base-${MODE} AS final

ARG MODE
ARG ROCM_VERSION=6.4.4
ARG AMD_GPU_ARCH=gfx1101
ARG CAUSAL_CONV1D_VERSION=v1.5.3.post1
ARG MAMBA_VERSION=v2.2.6.post3

ENV PYTHONPYCACHEPREFIX=/.cache/python/
ENV DEBIAN_FRONTEND=noninteractive
ENV ROCM_PATH=/opt/rocm-${ROCM_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv python-is-python3 \
    swig wget curl git openssh-client default-jre build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --break-system-packages uv

COPY requirements.txt .
RUN case "${MODE}" in \
      "cpu")    export URL_BIT="cpu" ;; \
      "cuda")   export URL_BIT="cu130" ;; \
      "7800xt") export URL_BIT="rocm${ROCM_VERSION%.*}" ;; \
      *) echo "Invalid MODE: ${MODE}"; exit 1 ;; \
    esac \
    && uv pip install --system --no-cache --break-system-packages \
       --index-url "https://download.pytorch.org/whl/${URL_BIT}" \
       torch torchvision torchaudio \
    && uv pip install --system --no-cache --break-system-packages -r requirements.txt

# AMD GPU Arch Spoofing
RUN if [ "${MODE}" = "7800xt" ]; then \
        AMDGPU_BIN="${ROCM_PATH}/lib/llvm/bin/amdgpu-arch" && \
        mv "$AMDGPU_BIN" "${AMDGPU_BIN}.real" && \
        printf '#!/bin/bash\necho %s\n' "${AMD_GPU_ARCH}" > "$AMDGPU_BIN" && \
        chmod +x "$AMDGPU_BIN"; \
    fi
RUN if [ "${MODE}" = "7800xt" ] || [ "${MODE}" = "cuda" ]; then \
        pip install --break-system-packages --no-cache-dir --no-build-isolation \
        packaging \
        "git+https://github.com/Dao-AILab/causal-conv1d.git@${CAUSAL_CONV1D_VERSION}" \
        "git+https://github.com/state-spaces/mamba.git@${MAMBA_VERSION}"; \
    fi
# Revert Arch Spoof
RUN if [ "${MODE}" = "7800xt" ]; then \
        AMDGPU_BIN="${ROCM_PATH}/lib/llvm/bin/amdgpu-arch" && \
        mv "${AMDGPU_BIN}.real" "$AMDGPU_BIN"; \
    fi