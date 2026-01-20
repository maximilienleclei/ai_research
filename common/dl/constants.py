"""Constants used throughout the deep learning framework.

This module centralizes magic numbers and configuration constants
to improve code readability and maintainability.
"""

# =============================================================================
# Batch Size Tuning Constants
# =============================================================================

# Minimum throughput improvement (as multiplier) required to prefer larger batch size.
# If larger batch is not at least 5% faster, prefer smaller for better convergence.
BATCH_SIZE_EFFICIENCY_THRESHOLD = 1.05

# Minimum batch size for throughput comparison testing
MIN_BATCH_SIZE_FOR_THROUGHPUT_CHECK = 8

# =============================================================================
# Worker Tuning Constants
# =============================================================================

# Buffer factor for worker saturation check.
# Workers are considered "saturated" if they can produce batches within 10% of GPU time.
WORKER_SATURATION_BUFFER = 1.1

# Default maximum number of batches to iterate when testing worker configurations
DEFAULT_MAX_DATA_PASSES = 50

# =============================================================================
# Torch Compile Constants
# =============================================================================

# Minimum CUDA compute capability required for torch.compile()
TORCH_COMPILE_MINIMUM_CUDA_VERSION = 7
