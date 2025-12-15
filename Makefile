# Default to the simple method, but swap to the 'Heavy Duty' command 
# if you hit the torch/mamba build errors.
lock:
	podman run --rm \
		-v "$$(pwd):/app" \
		-w /app \
		ghcr.io/astral-sh/uv:0.5.11 \
		lock

# Build the docker image using the lockfile we just generated
build-cuda:
	podman build --build-arg MODE=cuda -t ai-research:cuda .

build-rocm:
	podman build --build-arg MODE=rocm -t ai-research:rocm .