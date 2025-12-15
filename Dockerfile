# ==============================================================================
# 0. GLOBAL CONFIGURATION
# ==============================================================================
ARG UBUNTU_SHA=sha256:e3f92abc0967a6c19d0dfa2d55838833e947b9d74edbcb0113e48535ad4be12a

# Hardware Versions
ARG CUDA_VERSION=12.8.1
ARG ROCM_VERSION=6.4.2
ARG AMDGPU_VERSION=6.4.60402

# ==============================================================================
# STAGE 1: BUILDER
# ==============================================================================
FROM ubuntu:noble@${UBUNTU_SHA} AS builder

ARG MODE=cpu
ARG CUDA_VERSION
ARG ROCM_VERSION
ARG AMDGPU_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python-is-python3 \
    git \
    wget \
    # for box2d
    swig \
    # Java to build our fork of Hydra
    default-jre \
    # Media processing libraries
    ffmpeg \
    sox \
    libavdevice-dev \
    libffi8 \
    # Required for uv
    curl \
    # Clean up
    && rm -rf /var/lib/apt/lists/*
    && if [ "$MODE" = "cuda" ]; then \
        apt-get install -y --no-install-recommends gnupg2; \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb; \
        dpkg -i cuda-keyring_1.1-1_all.deb; \
        apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-12-8; \
    elif [ "$MODE" = "7800xt" ]; then \
        apt-get install -y --no-install-recommends gnupg2 software-properties-common; \
        wget https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/noble/amdgpu-install_${AMDGPU_VERSION}-1_all.deb; \
        apt-get install -y ./*.deb; \
        amdgpu-install --usecase=rocm -y && rm *.deb; \
    fi

RUN if [ "$MODE" = "cpu" ]; then \
        pip3 install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/cpu; \
    elif [ "$MODE" = "cuda" ]; then \
        CUDA_TAG=$(echo "$CUDA_VERSION" | cut -d. -f1,2 | tr -d '.'); \
        pip3 install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/cu${CUDA_TAG}; \
    elif [ "$MODE" = "7800xt" ]; then \
        ROCM_TAG=$(echo "$ROCM_VERSION" | cut -d. -f1,2); \
        pip3 install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/rocm${ROCM_TAG}; \
    fi

RUN pip3 install --no-cache-dir \
    numpy==1.26.4 \
    scikit-learn==1.7.2 \
    matplotlib==3.10.7 \
    datasets==4.4.1 \
    einops==0.8.1 \
    jaxtyping==0.3.3 \
    beartype==0.22.7 \
    ordered_set==4.1.0 \
    gymnasium[classic-control,box2d]==1.2.2

# ------------------------------------------------------------------------------
# 5. SPECIAL BUILD: MAMBA & CAUSAL-CONV1D
# ------------------------------------------------------------------------------
# Mamba requires proper compiler toolchains for each hardware mode
# Install from source for GPU modes only (cuda, 7800xt)
RUN if [ "$MODE" = "cpu" ]; then \
        echo ">>> Skipping Mamba for CPU mode..."; \
    elif [ "$MODE" = "cuda" ]; then \
        echo ">>> Building Mamba from source for CUDA..."; \
        export CUDA_HOME=/usr/local/cuda; \
        export PATH=$CUDA_HOME/bin:$PATH; \
        pip3 install "git+https://github.com/Dao-AILab/causal-conv1d.git@22a4577d8ace9d5703daea91a7fb56695492152b"; \
        pip3 install "git+https://github.com/state-spaces/mamba.git@35e927b20fd674f0b30a799a6408b7aac6ffe642"; \
    elif [ "$MODE" = "7800xt" ]; then \
        echo ">>> Building Mamba from source for 7800 XT (gfx1101)..."; \
        export HIP_HOME=/opt/rocm; \
        export ROCM_PATH=/opt/rocm; \
        export PATH=$HIP_HOME/bin:$PATH; \
        export HIP_ARCHITECTURES="gfx1101"; \
        export PYTORCH_ROCM_ARCH="gfx1101"; \
        echo ">>> Checking hipcc version..."; \
        hipcc --version || echo "hipcc not found in PATH"; \
        pip3 install \
           "git+https://github.com/Dao-AILab/causal-conv1d.git@22a4577d8ace9d5703daea91a7fb56695492152b"; \
        pip3 install \
           "git+https://github.com/state-spaces/mamba.git@35e927b20fd674f0b30a799a6408b7aac6ffe642"; \
    fi



# ==============================================================================
# STAGE 2: FINAL
# ==============================================================================
FROM ubuntu:noble@${UBUNTU_SHA} AS final

ARG ROCM_VERSION
ARG MODE=cpu
ENV DEBIAN_FRONTEND=noninteractive

# 1. Runtime Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libgomp1 \
    libgl1

# 2. Install ROCm Runtime libs (Only for 7800xt)
RUN if [ "$MODE" = "7800xt" ]; then \
        apt-get update && apt-get install -y --no-install-recommends wget gnupg2; \
        mkdir --parents --mode=0755 /etc/apt/keyrings; \
        wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null; \
        ROCM_TAG=$(echo "$ROCM_VERSION" | cut -d. -f1,2); \
        echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_TAG} noble main" | tee /etc/apt/sources.list.d/rocm.list; \
        apt-get update && apt-get install -y --no-install-recommends hip-runtime-amd rocm-device-libs; \
    fi

COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin