UV_VERSION ?= 0.9.17
# ROCm GPU architecture (gfx90a=MI200, gfx942=MI300, gfx1100=RX7900)
ROCM_ARCH ?= gfx1101

# --- 1. LOCKING HELPERS ---
# usage: make _gen_lock MODE=cpu OUT=uv.lock.cpu
_gen_lock:
	podman build --build-arg MODE=$(MODE) --build-arg UV_VERSION=$(UV_VERSION) -f Dockerfile.lock -t uv-lock:$(MODE) .
	podman run --rm uv-lock:$(MODE) cat /app/uv.lock > $(OUT)
	podman rmi uv-lock:$(MODE)

# Public locking targets
lock-cpu:
	$(MAKE) _gen_lock MODE=cpu OUT=uv.lock.cpu

lock-cuda:
	$(MAKE) _gen_lock MODE=cuda OUT=uv.lock.cuda

lock-rocm:
	$(MAKE) _gen_lock MODE=rocm OUT=uv.lock.rocm

lock-all: lock-cpu lock-cuda lock-rocm

# --- 2. BUILD TARGETS ---
build-cpu:
	# Ensure lock exists
	@test -f uv.lock.cpu || $(MAKE) lock-cpu
	podman build --build-arg MODE=cpu --build-arg UV_VERSION=$(UV_VERSION) -t ai-research:cpu .

build-cuda:
	@test -f uv.lock.cuda || $(MAKE) lock-cuda
	podman build --build-arg MODE=cuda --build-arg UV_VERSION=$(UV_VERSION) -t ai-research:cuda .

build-rocm:
	@test -f uv.lock.rocm || $(MAKE) lock-rocm
	podman build --build-arg MODE=rocm --build-arg UV_VERSION=$(UV_VERSION) --build-arg ROCM_ARCH=$(ROCM_ARCH) -t ai-research:rocm .