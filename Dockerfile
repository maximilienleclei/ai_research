# syntax=docker/dockerfile:1
FROM ubuntu:noble@sha256:e3f92abc0967a6c19d0dfa2d55838833e947b9d74edbcb0113e48535ad4be12a

ARG MODE=cpu
# ROCM versions
ENV ROCM_VERSION=6.4.2
ENV AMDGPU_VERSION=6.4.60402
# Python packages
ENV UV_VERSION=0.5.11

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
    && if [ "${MODE}" = "rocm" ]; then \
        wget -q https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/noble/amdgpu-install_${AMDGPU_VERSION}-1_all.deb \
        && apt-get install -y ./amdgpu-install_${AMDGPU_VERSION}-1_all.deb \
        && amdgpu-install -y --usecase=rocm --no-dkms \
        && rm amdgpu-install_${AMDGPU_VERSION}-1_all.deb; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /ai_research
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --extra ${MODE} --output-file requirements.txt
# mamba-ssm|causal-conv1d: no build isolation
RUN grep -vE "mamba-ssm|causal-conv1d" requirements.txt > requirements_stage1.txt && \
    grep -E "mamba-ssm|causal-conv1d" requirements.txt > requirements_stage2.txt || true
RUN uv pip install -r requirements_stage1.txt
# ROCM patch
RUN if [ "${MODE}" = "rocm" ]; then \
        location=$(python3 -c "import torch; print(torch.__path__[0])"); \
        cd "${location}/lib/" && \
        rm -f libhsa-runtime64.so*; \
    fi
RUN if [ -s requirements_stage2.txt ]; then \
        uv pip install --no-build-isolation -r requirements_stage2.txt; \
    fi
COPY . .
RUN uv pip install --no-deps -e .

CMD ["/bin/bash"]