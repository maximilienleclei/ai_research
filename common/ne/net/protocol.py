"""Network protocol defining common interface for all network types.

This module defines protocol interfaces that establish contracts without requiring
inheritance. All network types (BatchedFeedforward, BatchedRecurrent, DynamicNetPopulation)
implement these protocols to enable clean polymorphism in the Population adapter.
"""

from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor


@runtime_checkable
class NetworkProtocol(Protocol):
    """Base protocol that all network types must implement.

    Defines the minimal interface needed by the Population wrapper for basic
    operations: forward pass, mutation, and checkpointing.

    All networks in ne/net/ implement this protocol.
    """

    # Required attributes
    num_nets: int  # Or pop_size for DynamicNetPopulation
    device: str

    def forward_batch(
        self, x: Float[Tensor, "N input_size"]
    ) -> Float[Tensor, "num_nets N output_size"]:
        """Batched forward pass for all networks in parallel.

        Args:
            x: Input tensor [N, input_size]

        Returns:
            Output tensor [num_nets, N, output_size]
        """
        ...

    def mutate(self) -> None:
        """Apply mutations to all networks in the population."""
        ...

    def get_state_dict(self) -> dict:
        """Get state dictionary for checkpointing.

        Returns:
            Dict containing all state needed to reconstruct this network
        """
        ...

    def load_state_dict(self, state: dict) -> None:
        """Restore network from checkpoint state.

        Args:
            state: State dict from get_state_dict()
        """
        ...


@runtime_checkable
class ParameterizableNetwork(NetworkProtocol, Protocol):
    """Networks with flat parameter operations (for ES/CMA-ES optimizers).

    Only BatchedFeedforward and BatchedRecurrent implement this protocol.
    DynamicNetPopulation does NOT support flat parameters due to variable topology.

    This protocol enables parameter-averaging optimizers (ES, CMA-ES) to work
    with networks that have fixed architecture.
    """

    def get_parameters_flat(self) -> Float[Tensor, "num_nets num_params"]:
        """Get flattened parameter vectors for all networks.

        Returns:
            Flat parameters [num_nets, num_params] where num_params is the
            total number of parameters (weights + biases)
        """
        ...

    def set_parameters_flat(
        self, params: Float[Tensor, "num_nets num_params"]
    ) -> None:
        """Set network parameters from flat vectors.

        Args:
            params: Flat parameters [num_nets, num_params]
        """
        ...

    def clone_network(self, indices: Tensor) -> None:
        """Clone networks at specified indices to fill population.

        Used by GA selection: top performers are cloned to replace losers.

        Args:
            indices: Indices of networks to clone [num_nets]
                     (indices may repeat for duplication)
        """
        ...


@runtime_checkable
class StructuralNetwork(NetworkProtocol, Protocol):
    """Networks with evolving topology (for GA-only optimization).

    Only DynamicNetPopulation implements this protocol. These networks have
    variable structure that evolves over time, making parameter averaging
    (used by ES/CMA-ES) meaningless.

    These networks can only use GA optimizer which doesn't require parameter
    operations.
    """

    def select_and_duplicate(self, fitness: Float[Tensor, "num_nets"]) -> None:
        """Select top performers and duplicate to fill population.

        This method combines selection and duplication in one step because
        topology evolution requires special handling during network copying.

        Args:
            fitness: Fitness values [num_nets] (lower is better)
        """
        ...
