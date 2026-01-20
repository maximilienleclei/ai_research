"""Visualization utilities for Lightning modules.

Provides functions for converting tensors to W&B-compatible media objects.

Shape abbreviations:
    TSL: Target sequence length
    CSL: Conditioning sequence length
    TNC: Target number of channels
    CNC: Conditioning number of channels (i.e. number of features)
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import wandb
from jaxtyping import Float
from torch import Tensor

log = logging.getLogger(__name__)

# Maximum number of channels to treat as line plot vs heatmap
MAX_CHANNELS_FOR_LINE_PLOT = 4


def to_wandb_image(
    data: Float[Tensor, "TNC TSL"] | Float[Tensor, "CNC CSL"],
) -> wandb.Image:
    """Convert a tensor to a W&B Image for logging.

    For tensors with few channels (<=4), creates a line plot with each channel
    as a separate line. For tensors with more channels, creates a heatmap.

    Args:
        data: 2D tensor of shape (num_channels, sequence_length).
            Values are clipped to [-1, 1] for visualization.

    Returns:
        W&B Image object suitable for logging to tables.
    """
    plt.figure()
    log.debug(f"data.shape: {data.shape}")
    data = data.clip(-1, 1)
    if data.shape[0] <= MAX_CHANNELS_FOR_LINE_PLOT:
        for i in range(data.shape[0]):
            plt.plot(np.linspace(0, len(data[i]) - 1, len(data[i])), data[i])
        plt.ylim(-1.1, 1.1)
    else:  # Conditioning
        plt.imshow(data)
    plt.axis("off")
    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    image = data.reshape(canvas.get_width_height()[::-1] + (4,))
    plt.close()
    return wandb.Image(image[:, :, [1, 2, 3, 0]])
