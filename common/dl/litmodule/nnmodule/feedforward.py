"""Shapes:

BS: Batch size
NIF: Number of input features
NOF: Number of output features
"""

import logging
from dataclasses import dataclass
from typing import Annotated as An

from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from common.utils.beartype import ge

log = logging.getLogger(__name__)


@dataclass
class FNNConfig:
    input_size: An[int, ge(1)]
    output_size: An[int, ge(1)]
    num_layers: An[int, ge(1)]
    hidden_size: An[int, ge(1)] | None = None
    flatten: bool = True

    def __post_init__(self: "FNNConfig") -> None:
        if self.num_layers != 1 and self.hidden_size is None:
            error_msg = "`num_layers` must be 1 if `hidden_size` is `None`."
            raise ValueError(error_msg)


class FNN(nn.Module):
    def __init__(self: "FNN", config: FNNConfig) -> None:
        super().__init__()
        self.config = config
        inner_dims = (
            []
            if config.hidden_size is None
            else [config.hidden_size] * (config.num_layers - 1)
        )
        dims = [config.input_size, *inner_dims, config.output_size]
        layers: list[nn.Module] = []
        for i in range(config.num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # ReLU after all but the last layer
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(
        self: "FNN",
        x: Float[Tensor, " BS *_ NIF"],
    ) -> Float[Tensor, " BS *_ NOF"]:
        if self.config.flatten:
            x = rearrange(x, "BS ... -> BS (...)")
        log.debug(f"x.shape: {x.shape}")
        x: Float[Tensor, " BS *_ NOF"] = self.model(x)
        log.debug(f"x.shape: {x.shape}")
        return x
