""":class:`.FNN` & its config.

---

Shapes:
    - BS: Batch size
    - NIF: Number of input features
        (:paramref:`.FNNConfig.input_size`)
    - NOF: Number of output features
        (:paramref:`.FNNConfig.output_size`)
"""

import logging
from dataclasses import dataclass
from typing import Annotated as An

from jaxtyping import Float
from torch import Tensor, nn

from common.utils.beartype import ge

log = logging.getLogger(__name__)


@dataclass
class FNNConfig:
    """Holds :class:`FNN` config values.

    Args:
        input_size
        output_size
        num_layers
        hidden_size
    """

    input_size: An[int, ge(1)]
    output_size: An[int, ge(1)]
    num_layers: An[int, ge(1)]
    hidden_size: An[int, ge(1)] | None = None

    def __post_init__(self: "FNNConfig") -> None:
        """Post-initialization checks."""
        if self.num_layers != 1 and self.hidden_size is None:
            error_msg = "`num_layers` must be 1 if `hidden_size` is `None`."
            raise ValueError(error_msg)


class FNN(nn.Module):
    """A Feedforward Neural Network with ReLU activations.

    Args:
        config

    Attributes:
        config (FNNConfig)
        model (torch.nn.Sequential)
    """

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
        """.

        Args:
            x
        """
        log.debug(f"x.shape: {x.shape}")
        x: Float[Tensor, " BS *_ NOF"] = self.model(x)
        log.debug(f"x.shape: {x.shape}")
        return x
