# syntax=docker/dockerfile:1
FROM ubuntu:noble@sha256:e3f92abc0967a6c19d0dfa2d55838833e947b9d74edbcb0113e48535ad4be12a

ARG MODE=cpu
ARG UV_VERSION=0.9.17
# ROCm GPU architecture (e.g., gfx90a for MI200, gfx942 for MI300, gfx1100 for RX7900)
ARG ROCM_ARCH=gfx1101
# ROCM versions
ENV CUDA_VERSION=12.8
ENV ROCM_VERSION=6.4.2
ENV AMDGPU_VERSION=6.4.60402

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    python-is-python3 \
    # Box2D dependency
    swig \
    git \
    wget \
    curl \
    # Hydra fork build dependency
    default-jre \
    build-essential \
    && if [ "${MODE}" = "cuda" ]; then \
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
        && dpkg -i cuda-keyring_1.1-1_all.deb \
        && apt-get update \
        && CUDA_PKG_VERSION=$(echo "$CUDA_VERSION" | tr '.' '-') \
        && apt-get install -y cuda-toolkit-${CUDA_PKG_VERSION} \
        && rm cuda-keyring_1.1-1_all.deb; \
    fi \
    && if [ "${MODE}" = "rocm" ]; then \
        wget -q https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/noble/amdgpu-install_${AMDGPU_VERSION}-1_all.deb \
        && apt-get install -y ./amdgpu-install_${AMDGPU_VERSION}-1_all.deb \
        && amdgpu-install -y --usecase=rocm --no-dkms \
        && rm amdgpu-install_${AMDGPU_VERSION}-1_all.deb; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /ai_research
COPY pyproject.toml .

COPY uv.lock.cpu uv.lock.cuda uv.lock.rocm ./
RUN cp uv.lock.${MODE} uv.lock
RUN uv sync --frozen --no-dev --no-install-project --active

# ROCM patch
RUN if [ "${MODE}" = "rocm" ]; then \
        location=$(python3 -c "import torch; print(torch.__path__[0])"); \
        cd "${location}/lib/" && \
        rm -f libhsa-runtime64.so*; \
    fi

# Mamba (requires specifying GPU arch for ROCm since no GPU is present during build)
ARG ROCM_ARCH=gfx1101
# Create mock amdgpu-arch that returns target arch (real one fails without GPU)
RUN if [ "${MODE}" = "rocm" ]; then \
        mv /opt/rocm-6.4.2/lib/llvm/bin/amdgpu-arch /opt/rocm-6.4.2/lib/llvm/bin/amdgpu-arch.real && \
        echo '#!/bin/bash' > /opt/rocm-6.4.2/lib/llvm/bin/amdgpu-arch && \
        echo "echo ${ROCM_ARCH}" >> /opt/rocm-6.4.2/lib/llvm/bin/amdgpu-arch && \
        chmod +x /opt/rocm-6.4.2/lib/llvm/bin/amdgpu-arch; \
    fi
RUN if [ "${MODE}" = "rocm" ]; then \
        uv pip install --no-build-isolation \
        "git+https://github.com/Dao-AILab/causal-conv1d.git@v1.5.3.post1" \
        "git+https://github.com/state-spaces/mamba.git@v2.2.6.post3"; \
    elif [ "${MODE}" = "cuda" ]; then \
        uv pip install --no-build-isolation \
        "git+https://github.com/Dao-AILab/causal-conv1d.git@v1.5.3.post1" \
        "git+https://github.com/state-spaces/mamba.git@v2.2.6.post3"; \
    fi
# Restore real amdgpu-arch for runtime
RUN if [ "${MODE}" = "rocm" ]; then \
        mv /opt/rocm-6.4.2/lib/llvm/bin/amdgpu-arch.real /opt/rocm-6.4.2/lib/llvm/bin/amdgpu-arch; \
    fi

COPY . .
RUN uv pip install --no-deps -e .

CMD ["/bin/bash"]