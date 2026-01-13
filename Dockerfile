FROM ubuntu:24.04

ARG MODE=cpu

# __pycache__/ and .pyc/ out of the project folder
ENV PYTHONPYCACHEPREFIX=/.cache/python/

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    python-is-python3 \
    # Box2D dependency
    swig \
    # Installs
    wget \
    curl \
    git \
    # Dev
    openssh-client \
    # Hydra fork build dependency
    default-jre \
    build-essential \
    && if [ "${MODE}" = "cuda" ]; then \
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
        && dpkg -i cuda-keyring_1.1-1_all.deb \
        && apt-get update \
        && CUDA_PKG_VERSION=$(echo "$CUDA_VERSION" | tr '.' '-') \
        && apt-get install -y cuda-toolkit-12-8 \
        && rm cuda-keyring_1.1-1_all.deb; \
    fi \
    && if [ "${MODE}" = "7800xt" ]; then \
        wget -q https://repo.radeon.com/amdgpu-install/6.4.2/ubuntu/noble/amdgpu-install_6.4.60402-1_all.deb \
        && apt-get install -y ./amdgpu-install_6.4.60402-1_all.deb \
        && amdgpu-install -y --usecase=rocm --no-dkms \
        && rm amdgpu-install_6.4.60402-1_all.deb; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Python packages except Mamba
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY requirements.txt .
RUN case "${MODE}" in \
      "cpu") export URL_BIT="cpu" ;; \
      "cuda") export URL_BIT="cu128" ;; \
      "7800xt") export URL_BIT="rocm6.4" ;; \
      *) echo "Invalid MODE: ${MODE}"; exit 1 ;; \
    esac \
    && echo "Selected URL_BIT: ${URL_BIT}" \
    && uv pip install --system --no-cache --break-system-packages \
       --index-url "https://download.pytorch.org/whl/${URL_BIT}" \
       torch torchvision torchaudio triton pytorch-triton-rocm \
    && uv pip install --system --no-cache --break-system-packages -r requirements.txt

# Delete `libhsa-runtime64.so` to give `torch` GPU visibility
RUN if [ "${MODE}" = "7800xt" ]; then \
        location=$(python3 -c "import torch; print(torch.__path__[0])"); \
        cd "${location}/lib/" && \
        rm -f libhsa-runtime64.so*; \
    fi

# GPU arch trick
RUN if [ "${MODE}" = "7800xt" ]; then \
        mv /opt/rocm-6.4.2/lib/llvm/bin/amdgpu-arch /opt/rocm-6.4.2/lib/llvm/bin/amdgpu-arch.real && \
        echo '#!/bin/bash' > /opt/rocm-6.4.2/lib/llvm/bin/amdgpu-arch && \
        echo "echo gfx1101" >> /opt/rocm-6.4.2/lib/llvm/bin/amdgpu-arch && \
        chmod +x /opt/rocm-6.4.2/lib/llvm/bin/amdgpu-arch; \
    fi
# Mamba
RUN if [ "${MODE}" = "7800xt" ] || [ "${MODE}" = "cuda" ]; then \
        pip install --break-system-packages --no-cache-dir --no-build-isolation \
        packaging \
        "git+https://github.com/Dao-AILab/causal-conv1d.git@v1.5.3.post1" \
        "git+https://github.com/state-spaces/mamba.git@v2.2.6.post3"; \
    fi
# Reverse GPU arch trick
RUN if [ "${MODE}" = "7800xt" ]; then \
        mv /opt/rocm-6.4.2/lib/llvm/bin/amdgpu-arch.real /opt/rocm-6.4.2/lib/llvm/bin/amdgpu-arch; \
    fi
