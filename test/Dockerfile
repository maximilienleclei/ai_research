# Reverted to the original base image as requested.
FROM ubuntu:noble@sha256:e3f92abc0967a6c19d0dfa2d55838833e947b9d74edbcb0113e48535ad4be12a

ARG MODE=cpu
# GPU
ENV CUDA_VERSION=12.8.1
ENV ROCM_VERSION=6.4.2
ENV AMDGPU_VERSION=6.4.60402
# PIP
ENV UV_VERSION=0.9.17
# TORCH
ENV TRITON_VERSION=3.2.0
ENV TORCH_VERSION=2.7.1
ENV TORCHAUDIO_VERSION=2.7.1
ENV TORCHVISION_VERSION=0.22.1

# SYSTEM PACKAGES
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python-is-python3 \
    # Box2D dependencies
    swig \
    git \
    wget \
    # Java to build Hydra fork
    default-jre \
    # Install ROCm only if MODE is set to 'rocm'
    && if [ "${MODE}" = "rocm" ]; then \
        wget -q https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/noble/amdgpu-install_${AMDGPU_VERSION}-1_all.deb \
        && apt-get install -y ./amdgpu-install_${AMDGPU_VERSION}-1_all.deb \
        && amdgpu-install -y --usecase=rocm --no-dkms \
        && rm amdgpu-install_${AMDGPU_VERSION}-1_all.deb; \
    fi \
    # Clean up apt lists for all cases to reduce image size
    && rm -rf /var/lib/apt/lists/*

# PIP
RUN pip install uv==${UV_VERSION} --break-system-packages

# TORCH
RUN if [ "${MODE}" = "cpu" ]; then \
        INDEX_URL="https://download.pytorch.org/whl/cpu"; \
    elif [ "${MODE}" = "cuda" ]; then \
        CUDA_TAG=$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr -d '.'); \
        INDEX_URL="https://download.pytorch.org/whl/cu${CUDA_TAG}"; \
    elif [ "${MODE}" = "rocm" ]; then \
        ROCM_TAG=$(echo "${ROCM_VERSION}" | cut -d. -f1,2); \
        INDEX_URL="https://download.pytorch.org/whl/rocm${ROCM_TAG}"; \
    else \
        echo "Error: Unknown MODE '${MODE}'" && exit 1; \
    fi && \
    uv pip install --system \
        triton==${TRITON_VERSION} \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        --index-url "${INDEX_URL}" \
    && if [ "${MODE}" = "rocm" ]; then \
        location=$(pip show torch | grep Location | awk -F ": " '{print $2}'); \
        cd "${location}/torch/lib/" && \
        rm -f libhsa-runtime64.so*; \
    fi

# mamba-ssm
RUN if [ "${MODE}" = "cpu" ]; then \
    # FIX 4: Replaced invalid command 'nothing' with a valid shell command.
    echo "CPU Mode: Skipping Mamba/Causal-Conv1d"; \
    elif [ "${MODE}" = "cuda" ]; then \
    # FIX 5: Corrected flags to --no-build-isolation and added --system.
    uv pip install --system \
        --no-build-isolation \
        causal-conv1d==1.5.3.post1 \
        mamba-ssm==2.2.6.post3; \
    elif [ "${MODE}" = "rocm" ]; then \
    # FIX 6: Added 'git+' prefix to repository URLs.
    uv pip install --system git+https://github.com/Dao-AILab/causal-conv1d.git@52949d0891770c65243ed24db8abb66ef37741b7 \
    && uv pip install --system git+https://github.com/state-spaces/mamba.git@10b5d6358f27966f6a40e4bf0baa17a460688128; \
    fi


ADD pyproject.toml /ai_research/pyproject.toml
WORKDIR /ai_research
RUN uv lock --all
RUN uv pip install --system --no-cache-dir --no-build-isolation -r uv.lock -e .
