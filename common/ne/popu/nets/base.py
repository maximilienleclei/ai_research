"""Base classes for network populations in neuroevolution.

Networks are the evolved entities in neuroevolution. They take observations
as input and produce actions (or action logits) as output. A population of
networks evolves together through mutation and selection.

Note:
    BaseNetsConfig uses Hydra's interpolation to extract num_inputs/num_outputs
    from the evaluator at config resolution time. The evaluator reference is then
    deleted to prevent circular serialization. This is intentional - the config
    should only contain the extracted dimensions, not the full evaluator.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from common.ne.eval.base import BaseEval


@dataclass
class BaseNetsConfig:
    """Configuration for network populations.

    Attributes:
        device: Device for tensor operations (e.g., "cpu", "cuda:0").
            Interpolated from config.device.
        num_nets: Number of networks in the population.
            Interpolated from popu.config.size.
        num_inputs: Network input dimension (observation size).
            Computed from evaluator in __post_init__.
        num_outputs: Network output dimension (action size).
            Computed from evaluator in __post_init__.

    Note:
        The `eval` field is used only during initialization to extract dimensions.
        It is deleted after extraction to prevent circular references during
        serialization and to keep the config focused on network parameters.
    """

    device: str = "${config.device}"
    num_nets: int = "${popu.config.size}"
    eval: "BaseEval" = field(default="${eval}", repr=False)  # type: ignore[assignment]
    # Computed fields - set in __post_init__
    num_inputs: int = field(init=False, default=0)
    num_outputs: int = field(init=False, default=0)

    def __post_init__(self: "BaseNetsConfig") -> None:
        """Extract network dimensions from evaluator and clean up reference."""
        self.num_inputs, self.num_outputs = (
            self.eval.retrieve_num_inputs_outputs()
        )
        # Remove eval reference to prevent circular serialization
        # The dimensions are now stored in num_inputs/num_outputs
        del self.eval


class BaseNets(ABC):
    """Abstract base class for network populations.

    Networks are the evolved entities in neuroevolution. This base class
    defines the interface that all network implementations must follow.

    Attributes:
        config: Configuration containing device, dimensions, and population size.

    Subclasses:
        - StaticNets: Fixed-architecture networks (feedforward, recurrent)
        - DynamicNets: Networks with evolving topology (node growth/pruning)
    """

    def __init__(self: "BaseNets", config: BaseNetsConfig) -> None:
        """Initialize network population with configuration.

        Args:
            config: Network configuration with device and dimension info.
        """
        self.config = config

    @abstractmethod
    def mutate(self: "BaseNets") -> None:
        """Apply mutations to all networks in the population.

        This method modifies networks in-place. For static networks, this
        typically means perturbing weights. For dynamic networks, this may
        also include architectural changes (node growth/pruning).
        """
        ...

    @abstractmethod
    def resample(self: "BaseNets", indices: Tensor) -> None:
        """Resample (clone) networks according to selection indices.

        Args:
            indices: Index tensor of shape (num_nets,) specifying which network
                each position should be cloned from.

        Note:
            This implements the selection step of evolution. Indices typically
            come from a selection algorithm (e.g., top-50% duplication in SimpleGA).
        """
        ...

    def reset(self: "BaseNets") -> None:
        """Reset network state between episodes.

        Override this for stateful networks (e.g., recurrent networks that
        maintain hidden states). Default implementation does nothing.

        Note:
            For DynamicNets, reset() intentionally does NOT reset running
            standardization statistics, as those are learned normalizations
            that should persist across episodes.
        """
        pass

    @abstractmethod
    def __call__(self: "BaseNets", x: Tensor) -> Tensor:
        """Forward pass through all networks in the population.

        Args:
            x: Input observations with shape (num_nets, input_dim).

        Returns:
            Output actions/logits with shape (num_nets, output_dim).
        """
        ...
