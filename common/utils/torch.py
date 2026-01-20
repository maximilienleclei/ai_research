"""PyTorch utilities for data processing."""

import torch
from jaxtyping import Float32
from torch import Tensor


class RunningStandardization:
    """Online standardization using Welford's algorithm.

    Computes running mean and standard deviation, then standardizes
    each input to zero mean and unit variance.
    """

    def __init__(self: "RunningStandardization", x_size: int) -> None:
        """Initialize running statistics.

        Args:
            x_size: Dimension of input vectors.
        """
        self.mean: Float32[Tensor, "x_size"] = torch.zeros(size=(x_size,))
        self.var: Float32[Tensor, "x_size"] = torch.zeros(size=(x_size,))
        self.std: Float32[Tensor, "x_size"] = torch.zeros(size=(x_size,))
        self.n: Float32[Tensor, "1"] = torch.zeros(size=(1,))

    def __call__(
        self: "RunningStandardization", x: Float32[Tensor, "x_size"]
    ) -> Float32[Tensor, "x_size"]:
        """Update statistics and return standardized input."""
        self.n += torch.ones(size=(1,))
        new_mean: Float32[Tensor, "x_size"] = (
            self.mean + (x - self.mean) / self.n
        )
        new_var: Float32[Tensor, "x_size"] = self.var + (x - self.mean) * (
            x - new_mean
        )
        new_std: Float32[Tensor, "x_size"] = torch.sqrt(new_var / self.n)
        self.mean, self.var, self.std = new_mean, new_var, new_std
        standardized_x: Float32[Tensor, "x_size"] = (x - self.mean) / (
            self.std + self.std.eq(0)
        )
        return standardized_x
