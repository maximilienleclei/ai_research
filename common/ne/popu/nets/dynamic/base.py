"""Shapes:

TNN: Total number of nodes (in all networks).
TNMN: Total number of mutable (hidden and output) nodes (in all networks).
NN: Number of networks
NO: Number of outputs (per network)
NI: Number of inputs (per network)
"""

import logging

import torch
from jaxtyping import Float, Int
from torch import Tensor

from common.ne.popu.nets.base import BaseNets, BaseNetsConfig
from common.ne.popu.nets.dynamic.evolution import Net
from common.ne.popu.nets.dynamic.utils import WelfordRunningStandardizer

log = logging.getLogger(__name__)


class DynamicNets(BaseNets):

    def __init__(self: "DynamicNets", config: BaseNetsConfig):
        super().__init__(config)
        self.nets: list[Net] = [
            Net(
                self.config.num_inputs,
                self.config.num_outputs,
                self.config.device,
            )
            for _ in range(self.config.num_nets)
        ]
        self.have_mutated = False
        self._net_data: dict[int, dict] = {}

    def mutate(self: "DynamicNets") -> None:
        """Mutate all networks in the population."""
        if self.have_mutated:
            self._update_nets_standardization_values()

        # Mutate and cache data for fast prepare_for_computation
        self._net_data = {}
        for i, net in enumerate(self.nets):
            net.mutate()
            self._net_data[i] = {
                "num_nodes": len(net.nodes.all),
                "n_mean_m2_x_z": net.n_mean_m2_x_z,
                "in_nodes_indices": net.in_nodes_indices,
                "weights": net.weights,
                "num_network_passes_per_input": net.num_network_passes_per_input,
            }

        self.have_mutated = True
        self._prepare_for_computation()

    def _update_nets_standardization_values(self: "DynamicNets") -> None:
        """Update each net's n_mean_m2_x_z from the batched tensor."""
        for i in range(self.config.num_nets):
            start = self._input_nodes_start_indices[i]
            end = (
                None
                if i + 1 == self.config.num_nets
                else self._input_nodes_start_indices[i + 1]
            )
            self.nets[i].n_mean_m2_x_z = self._wrs.n_mean_m2_x_z[start:end]

    def _prepare_for_computation(self: "DynamicNets") -> None:
        """Build all tensors needed for batched computation."""
        device = self.config.device

        nets_num_nodes: Int[Tensor, "NN"] = torch.tensor(
            [self._net_data[i]["num_nodes"] for i in range(self.config.num_nets)],
            device=device,
        )

        # We add a value at the front to aid computation. This value at index 0
        # will always output 0. Empty in-node slots map to 0, meaning that node.
        n_mean_m2_x_z: Float[Tensor, "TNNplus1 5"] = torch.cat(
            [torch.zeros(1, 5, device=device)]
            + [
                self._net_data[i]["n_mean_m2_x_z"].to(device)
                for i in range(self.config.num_nets)
            ]
        )
        self._wrs = WelfordRunningStandardizer(n_mean_m2_x_z)

        self._input_nodes_start_indices: Int[Tensor, "NN"] = (
            torch.cat(
                (torch.tensor([0], device=device), torch.cumsum(nets_num_nodes[:-1], dim=0))
            )
            + 1
        )

        self._input_nodes_indices: Int[Tensor, "NIxNN"] = (
            self._input_nodes_start_indices.unsqueeze(1)
            + torch.arange(self.config.num_inputs, device=device)
        ).flatten()

        output_nodes_start_indices: Int[Tensor, "NN"] = (
            self._input_nodes_start_indices + self.config.num_inputs
        )

        self._output_nodes_indices: Int[Tensor, "NOxNN"] = (
            output_nodes_start_indices.unsqueeze(1)
            + torch.arange(self.config.num_outputs, device=device)
        ).flatten()

        nodes_indices = torch.arange(1, len(n_mean_m2_x_z), device=device)
        self._mutable_nodes_indices: Int[Tensor, "TNMN"] = nodes_indices[
            ~torch.isin(nodes_indices, self._input_nodes_indices)
        ]

        nets_num_mutable_nodes: Int[Tensor, "NN"] = (
            nets_num_nodes - self.config.num_inputs
        )
        nets_cum_num_mutable_nodes: Int[Tensor, "NN"] = torch.cumsum(
            nets_num_mutable_nodes, 0
        )
        # Convert to Python list for fast indexing in loops
        nets_cum_num_mutable_nodes_list: list[int] = nets_cum_num_mutable_nodes.tolist()

        in_nodes_indices: Int[Tensor, "TNMN 3"] = torch.empty(
            (nets_num_mutable_nodes.sum(), 3), dtype=torch.int32, device=device
        )
        for i in range(self.config.num_nets):
            start: int = 0 if i == 0 else nets_cum_num_mutable_nodes_list[i - 1]
            end: int = nets_cum_num_mutable_nodes_list[i]
            net_in_nodes_indices: Int[Tensor, "NET_NMN 3"] = self._net_data[i][
                "in_nodes_indices"
            ].to(device)
            in_nodes_indices[start:end] = (
                net_in_nodes_indices
                + (net_in_nodes_indices >= 0)
                * self._input_nodes_start_indices[i]
            )
        in_nodes_indices = torch.clamp(in_nodes_indices, min=0)
        self._flat_in_nodes_indices: Int[Tensor, "TNMNx3"] = in_nodes_indices.flatten()

        self._weights: Float[Tensor, "TNMN 3"] = torch.cat(
            [self._net_data[i]["weights"].to(device) for i in range(self.config.num_nets)]
        )

        # Cache as Python list for fast access in nested loop
        num_network_passes_list: list[int] = [
            self._net_data[i]["num_network_passes_per_input"]
            for i in range(self.config.num_nets)
        ]
        self._max_num_network_passes_per_input: int = max(num_network_passes_list)
        self._num_network_passes_per_input_mask = torch.zeros(
            (
                self._max_num_network_passes_per_input,
                nets_num_mutable_nodes.sum(),
            ),
            device=device,
        )

        for i in range(self._max_num_network_passes_per_input):
            for j in range(self.config.num_nets):
                if num_network_passes_list[j] > i:
                    start = 0 if j == 0 else nets_cum_num_mutable_nodes_list[j - 1]
                    end = nets_cum_num_mutable_nodes_list[j]
                    self._num_network_passes_per_input_mask[i][start:end] = 1
        self._num_network_passes_per_input_mask = (
            self._num_network_passes_per_input_mask.bool()
        )

    def resample(self: "DynamicNets", indices: Tensor) -> None:
        """Resample networks according to the provided indices."""
        indices_list = indices.tolist()
        self.nets = [self.nets[i].clone() for i in indices_list]

        # Update cache after cloning
        self._net_data = {}
        for i, net in enumerate(self.nets):
            self._net_data[i] = {
                "num_nodes": len(net.nodes.all),
                "n_mean_m2_x_z": net.n_mean_m2_x_z,
                "in_nodes_indices": net.in_nodes_indices,
                "weights": net.weights,
                "num_network_passes_per_input": net.num_network_passes_per_input,
            }

        self.have_mutated = False
        self._prepare_for_computation()

    # reset() intentionally inherits no-op from BaseNets.
    # Unlike recurrent nets (episode-specific hidden states), DynamicNets'
    # running standardizer stats are learned normalization values that should
    # persist. Zeroing them + calling _prepare_for_computation() was destroying
    # accumulated stats multiple times per generation, preventing learning.

    def __call__(
        self: "DynamicNets", x: Float[Tensor, "NN NI"]
    ) -> Float[Tensor, "NN NO"]:
        """Forward pass through the network population."""
        flat_obs: Float[Tensor, "NNxNI"] = x.flatten()

        out: Float[Tensor, "TNNplus1"] = self._wrs.n_mean_m2_x_z[:, 4].clone()
        out[self._input_nodes_indices] = flat_obs
        out: Float[Tensor, "TNNplus1"] = self._wrs(out)

        for j in range(self._max_num_network_passes_per_input):
            mapped_out: Float[Tensor, "TNMN 3"] = torch.gather(
                out, 0, self._flat_in_nodes_indices
            ).reshape(-1, 3)

            matmuld_mapped_out: Float[Tensor, "TNMN"] = (
                mapped_out * self._weights
            ).sum(dim=1)

            out[self._mutable_nodes_indices] = torch.where(
                self._num_network_passes_per_input_mask[j],
                matmuld_mapped_out,
                out[self._mutable_nodes_indices],
            )

            out: Float[Tensor, "TNNplus1"] = self._wrs(out)

        output: Float[Tensor, "NN NO"] = out[
            self._output_nodes_indices
        ].reshape(self.config.num_nets, self.config.num_outputs)

        return output
