"""Dynamic network models with evolving topology (GA-exclusive).

This module provides a wrapper for dynamic networks that implements
the batched population interface for efficient GPU-parallel computation.

Note: Dynamic networks are GA-exclusive as topology mutations are not differentiable.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


class DynamicNetPopulation:
    """Wrapper for dynamic network population with batched population interface.

    Uses dynamic networks with graph-based recurrence (via cycles and multiple passes).
    Properly implements batched computation following compute_test.py pattern.

    Note: This model is GA-exclusive (not compatible with SGD) because topology
    mutations are not differentiable.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        pop_size: int,
        device: str = "cpu",
    ) -> None:
        """Initialize dynamic network population.

        Args:
            input_size: Input dimension
            output_size: Output dimension
            pop_size: Population size
            device: Device to run computation on
        """
        from .evolution import Net
        from .compute_test import WelfordRunningStandardizer

        self.input_size: int = input_size
        self.output_size: int = output_size
        self.pop_size: int = pop_size
        self.device: str = device

        # Create networks
        self.nets: list[Net] = [
            Net(input_size, output_size, device=self.device) for _ in range(pop_size)
        ]

        # Initialize networks with one mutation each
        for net in self.nets:
            net.mutate()

        print(f"  Initialized {pop_size} dynamic networks")

        # Build computation infrastructure
        self._build_computation_infrastructure()

    def _build_computation_infrastructure(self) -> None:
        """Build batched computation tensors following compute_test.py pattern."""
        from .compute_test import WelfordRunningStandardizer

        # Get number of nodes for each network
        nets_num_nodes: Int[Tensor, "pop_size"] = torch.tensor(
            [len(net.nodes.all) for net in self.nets], device=self.device
        )

        # Concatenate n_mean_m2_x_z with zero entry at front
        self.n_mean_m2_x_z: Float[Tensor, "TNNplus1 5"] = torch.cat(
            [torch.zeros(1, 5, device=self.device)]
            + [net.n_mean_m2_x_z.to(self.device) for net in self.nets]
        )

        # Create WelfordRunningStandardizer
        self.wrs: WelfordRunningStandardizer = WelfordRunningStandardizer(
            self.n_mean_m2_x_z, verbose=False
        )

        # Build index mappings
        self.input_nodes_start_indices: Int[Tensor, "pop_size"] = (
            torch.cat(
                (
                    torch.tensor([0], device=self.device),
                    torch.cumsum(nets_num_nodes[:-1], dim=0),
                )
            )
            + 1
        )

        self.input_nodes_indices: Int[Tensor, "input_sizexpop_size"] = (
            self.input_nodes_start_indices.unsqueeze(1)
            + torch.arange(self.input_size, device=self.device)
        ).flatten()

        output_nodes_start_indices: Int[Tensor, "pop_size"] = (
            self.input_nodes_start_indices + self.input_size
        )

        self.output_nodes_indices: Int[Tensor, "output_sizexpop_size"] = (
            output_nodes_start_indices.unsqueeze(1)
            + torch.arange(self.output_size, device=self.device)
        ).flatten()

        nodes_indices: Int[Tensor, "TNN"] = torch.arange(
            1, len(self.n_mean_m2_x_z), device=self.device
        )
        self.mutable_nodes_indices: Int[Tensor, "TNMN"] = nodes_indices[
            ~torch.isin(nodes_indices, self.input_nodes_indices)
        ]

        # Build in_nodes_indices and weights
        nets_num_mutable_nodes: Int[Tensor, "pop_size"] = (
            nets_num_nodes - self.input_size
        )
        nets_cum_num_mutable_nodes: Int[Tensor, "pop_size"] = torch.cumsum(
            nets_num_mutable_nodes, 0
        )

        in_nodes_indices: Int[Tensor, "TNMN 3"] = torch.empty(
            (nets_num_mutable_nodes.sum(), 3), dtype=torch.int32, device=self.device
        )

        for i in range(self.pop_size):
            start: int = 0 if i == 0 else nets_cum_num_mutable_nodes[i - 1].item()
            end: int = nets_cum_num_mutable_nodes[i].item()
            net_in_nodes_indices: Int[Tensor, "NET_NMN 3"] = self.nets[
                i
            ].in_nodes_indices
            in_nodes_indices[start:end] = (
                net_in_nodes_indices + (net_in_nodes_indices >= 0) * self.input_nodes_start_indices[i]
            ).to(self.device)

        in_nodes_indices = torch.relu(in_nodes_indices)  # Map -1s to 0s
        self.flat_in_nodes_indices: Int[Tensor, "TNMNx3"] = in_nodes_indices.flatten()

        self.weights: Float[Tensor, "TNMN 3"] = torch.cat(
            [torch.tensor(net.weights, device=self.device) for net in self.nets]
        )

        # Build mask for variable num_network_passes_per_input
        num_network_passes_per_input: Int[Tensor, "pop_size"] = torch.tensor(
            [net.num_network_passes_per_input for net in self.nets],
            device=self.device,
        )
        self.max_num_network_passes_per_input: int = max(
            num_network_passes_per_input
        ).item()

        self.num_network_passes_per_input_mask: Float[Tensor, "max_passes TNMN"] = (
            torch.zeros(
                (self.max_num_network_passes_per_input, nets_num_mutable_nodes.sum()),
                device=self.device,
            )
        )

        for i in range(self.max_num_network_passes_per_input):
            for j in range(self.pop_size):
                if self.nets[j].num_network_passes_per_input > i:
                    start: int = (
                        0 if j == 0 else nets_cum_num_mutable_nodes[j - 1].item()
                    )
                    end: int = nets_cum_num_mutable_nodes[j].item()
                    self.num_network_passes_per_input_mask[i][start:end] = 1

        self.num_network_passes_per_input_mask = (
            self.num_network_passes_per_input_mask.bool()
        )

    def forward_batch(
        self, observations: Float[Tensor, "N input_size"]
    ) -> Float[Tensor, "pop_size N output_size"]:
        """Batched forward pass for all networks.

        Args:
            observations: Batch of observations [N, input_size]

        Returns:
            logits: Output logits for all networks [pop_size, N, output_size]
        """
        batch_size: int = len(observations)
        all_outputs: list[Float[Tensor, "pop_size output_size"]] = []

        with torch.no_grad():
            for i in range(batch_size):
                obs: Float[Tensor, " input_size"] = observations[i]

                # Flatten observation for all networks
                flat_obs: Float[Tensor, "input_sizexpop_size"] = obs.repeat(
                    self.pop_size
                )

                # Initialize output with previous z-scores
                out: Float[Tensor, "TNNplus1"] = self.n_mean_m2_x_z[:, 4].clone()

                # Set input nodes
                out[self.input_nodes_indices] = flat_obs

                # Apply Welford standardization
                out = self.wrs(out)

                # Multiple passes through the graph
                for pass_idx in range(self.max_num_network_passes_per_input):
                    # Map inputs for each mutable node
                    mapped_out: Float[Tensor, "TNMN 3"] = torch.gather(
                        out, 0, self.flat_in_nodes_indices
                    ).reshape(-1, 3)

                    # Compute weighted sum
                    matmuld_mapped_out: Float[Tensor, "TNMN"] = (
                        mapped_out * self.weights
                    ).sum(dim=1)

                    # Update only nodes that need this pass
                    out[self.mutable_nodes_indices] = torch.where(
                        self.num_network_passes_per_input_mask[pass_idx],
                        matmuld_mapped_out,
                        out[self.mutable_nodes_indices],
                    )

                    # Apply Welford standardization
                    out = self.wrs(out)

                # Extract output nodes
                outputs: Float[Tensor, "pop_size output_size"] = out[
                    self.output_nodes_indices
                ].reshape(self.pop_size, self.output_size)

                all_outputs.append(outputs)

        # Stack to get [pop_size, N, output_size]
        logits: Float[Tensor, "pop_size N output_size"] = torch.stack(
            all_outputs, dim=1
        )

        return logits

    def evaluate(
        self,
        observations: Float[Tensor, "N input_size"],
        actions: Int[Tensor, " N"],
    ) -> Float[Tensor, " pop_size"]:
        """Evaluate fitness (cross-entropy) of all networks.

        Args:
            observations: Batch of observations
            actions: Batch of actions

        Returns:
            fitness: Cross-entropy for each network [pop_size]
        """
        with torch.no_grad():
            # Get logits for all networks
            logits: Float[Tensor, "pop_size N output_size"] = self.forward_batch(
                observations
            )

            # Expand actions for all networks
            actions_expanded: Int[Tensor, "pop_size N"] = actions.unsqueeze(0).expand(
                self.pop_size, -1
            )

            # Reshape for cross_entropy
            flat_logits: Float[Tensor, "pop_sizexN output_size"] = logits.reshape(
                -1, self.output_size
            )
            flat_actions: Int[Tensor, " pop_sizexN"] = actions_expanded.reshape(-1)

            # Compute per-sample CE then reshape and mean per network
            per_sample_ce: Float[Tensor, " pop_sizexN"] = F.cross_entropy(
                flat_logits, flat_actions, reduction="none"
            )
            per_network_ce: Float[Tensor, "pop_size N"] = per_sample_ce.view(
                self.pop_size, -1
            )
            fitness: Float[Tensor, " pop_size"] = per_network_ce.mean(dim=1)

        return fitness

    def mutate(self) -> None:
        """Apply mutations to all networks in population."""
        for net in self.nets:
            net.mutate()
        # Rebuild computation infrastructure after topology changes
        self._build_computation_infrastructure()

    def select_and_duplicate(self, fitness: Float[Tensor, " pop_size"]) -> None:
        """Select top performers and duplicate to fill population.

        This method combines selection and duplication in one step because
        topology evolution requires special handling during network copying.

        Args:
            fitness: Fitness values [pop_size] (lower is better)
        """
        # Sort by fitness (minimize CE)
        sorted_indices: list[int] = torch.argsort(fitness).cpu().tolist()

        # Top 50% survive
        num_survivors: int = self.pop_size // 2
        survivor_indices: list[int] = sorted_indices[:num_survivors]

        # Create new population
        new_nets: list = []
        for idx in survivor_indices:
            new_nets.append(self.nets[idx])

        # Duplicate survivors to fill population
        num_losers: int = self.pop_size - num_survivors
        for i in range(num_losers):
            survivor_idx: int = survivor_indices[i % num_survivors]
            # Clone the network (uses proper clone() method instead of deepcopy)
            new_nets.append(self.nets[survivor_idx].clone())

        self.nets = new_nets
        # Rebuild after selection
        self._build_computation_infrastructure()

    def get_state_dict(self) -> dict:
        """Get state dict for checkpointing.

        Serializes all networks in the population along with metadata.

        Returns:
            State dict containing all network states and population metadata
        """
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "pop_size": self.pop_size,
            "device": self.device,
            # Serialize each network using its get_state_dict method
            "net_states": [net.get_state_dict() for net in self.nets],
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state dict from checkpoint.

        Reconstructs all networks from their serialized states.

        Args:
            state: State dict from get_state_dict()
        """
        from .evolution import Net

        self.input_size = state["input_size"]
        self.output_size = state["output_size"]
        self.pop_size = state["pop_size"]
        self.device = state["device"]

        # Reconstruct each network from its state
        self.nets = []
        for net_state in state["net_states"]:
            net = Net(
                num_inputs=net_state["num_inputs"],
                num_outputs=net_state["num_outputs"],
                device=self.device,
            )
            net.load_state_dict(net_state)
            self.nets.append(net)

        # Rebuild computation infrastructure
        self._build_computation_infrastructure()
