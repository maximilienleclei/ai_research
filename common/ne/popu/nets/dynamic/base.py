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

from common.ne.popu.nets.base import BaseNets
from common.ne.popu.nets.dynamic.evolution import Net
from common.ne.popu.nets.dynamic.utils import WelfordRunningStandardizer

log = logging.getLogger(__name__)


class DynamicNets(BaseNets):

    def __init__(self: "DynamicNets", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nets: list[Net] = [
            Net(
                self.config.num_inputs,
                self.config.num_outputs,
                self.config.device,
            )
            for _ in range(self.config.num_nets)
        ]
        self.have_mutated = False

    def mutate(self: "DynamicNets") -> None:
        """Mutate all networks in the population."""
        if self.have_mutated:
            self.update_nets_standardization_values()
        for net in self.nets:
            net.mutate()
        self.have_mutated = True
        self.prepare_for_computation()

    def update_nets_standardization_values(self: "DynamicNets") -> None:
        """Update each net's n_mean_m2_x_z from the batched tensor."""
        for i in range(self.config.num_nets):
            start = self.input_nodes_start_indices[i]
            end = (
                None
                if i + 1 == self.config.num_nets
                else self.input_nodes_start_indices[i + 1]
            )
            self.nets[i].n_mean_m2_x_z = self.wrs.n_mean_m2_x_z[start:end]

    def prepare_for_computation(self: "DynamicNets") -> None:
        """Build all tensors needed for batched computation."""
        nets_num_nodes: Int[Tensor, "NN"] = torch.tensor(
            [len(net.nodes.all) for net in self.nets]
        )
        log.debug("1. nets_num_nodes")
        log.debug(nets_num_nodes)

        # We add a value at the front to aid computation. This value at index 0
        # will always output 0. Empty in-node slots map to 0, meaning that node.
        n_mean_m2_x_z: Float[Tensor, "TNNplus1 5"] = torch.cat(
            ([torch.zeros(1, 5)] + [net.n_mean_m2_x_z for net in self.nets])
        )
        self.wrs = WelfordRunningStandardizer(n_mean_m2_x_z)
        log.debug("3. n_mean_m2_x_z")
        log.debug(n_mean_m2_x_z)
        log.debug(n_mean_m2_x_z.shape)

        self.input_nodes_start_indices: Int[Tensor, "NN"] = (
            torch.cat(
                (torch.tensor([0]), torch.cumsum(nets_num_nodes[:-1], dim=0))
            )
            + 1
        )
        log.debug("4. input_nodes_start_indices")
        log.debug(self.input_nodes_start_indices)

        self.input_nodes_indices: Int[Tensor, "NIxNN"] = (
            self.input_nodes_start_indices.unsqueeze(1)
            + torch.arange(self.config.num_inputs)
        ).flatten()
        log.debug("5. input_nodes_indices")
        log.debug(self.input_nodes_indices)

        output_nodes_start_indices: Int[Tensor, "NN"] = (
            self.input_nodes_start_indices + self.config.num_inputs
        )
        log.debug("6. output_nodes_start_indices")
        log.debug(output_nodes_start_indices)

        self.output_nodes_indices: Int[Tensor, "NOxNN"] = (
            output_nodes_start_indices.unsqueeze(1)
            + torch.arange(self.config.num_outputs)
        ).flatten()
        log.debug("7. output_nodes_indices")
        log.debug(self.output_nodes_indices)

        nodes_indices = torch.arange(1, len(n_mean_m2_x_z))
        self.mutable_nodes_indices: Int[Tensor, "TNMN"] = nodes_indices[
            ~torch.isin(nodes_indices, self.input_nodes_indices)
        ]
        log.debug("8. mutable_nodes_indices")
        log.debug(self.mutable_nodes_indices)

        nets_num_mutable_nodes: Int[Tensor, "4"] = (
            nets_num_nodes - self.config.num_inputs
        )
        nets_cum_num_mutable_nodes: Int[Tensor, "4"] = torch.cumsum(
            nets_num_mutable_nodes, 0
        )
        in_nodes_indices: Int[Tensor, "TNMN 3"] = torch.empty(
            (nets_num_mutable_nodes.sum(), 3), dtype=torch.int32
        )
        for i in range(self.config.num_nets):
            start: int = (
                0 if i == 0 else nets_cum_num_mutable_nodes[i - 1].item()
            )
            end: int = nets_cum_num_mutable_nodes[i].item()
            net_in_nodes_indices: Int[Tensor, "NET_NMN 3"] = self.nets[
                i
            ].in_nodes_indices
            in_nodes_indices[start:end] = (
                net_in_nodes_indices
                + (net_in_nodes_indices >= 0)
                * self.input_nodes_start_indices[i]
            )
        in_nodes_indices = torch.relu(in_nodes_indices)  # Map the -1s to 0s
        self.flat_in_nodes_indices: Int[Tensor, "TNMNx3"] = (
            in_nodes_indices.flatten()
        )
        log.debug("9. in_nodes_indices")
        log.debug(in_nodes_indices)
        log.debug(in_nodes_indices.shape)
        log.debug(self.flat_in_nodes_indices)

        self.weights: Float[Tensor, "TNMN 3"] = torch.cat(
            [net.weights for net in self.nets]
        )
        log.debug("10. weights")
        log.debug(self.weights)
        log.debug(self.weights.shape)

        num_network_passes_per_input: Int[Tensor, "NN"] = torch.tensor(
            [net.num_network_passes_per_input for net in self.nets]
        )
        self.max_num_network_passes_per_input: int = max(
            num_network_passes_per_input
        ).item()
        self.num_network_passes_per_input_mask = torch.zeros(
            (
                self.max_num_network_passes_per_input,
                nets_num_mutable_nodes.sum(),
            )
        )
        log.debug("11. num_network_passes_per_input")
        log.debug(num_network_passes_per_input)

        for i in range(self.max_num_network_passes_per_input):
            for j in range(self.config.num_nets):
                if self.nets[j].num_network_passes_per_input > i:
                    start = 0 if j == 0 else nets_cum_num_mutable_nodes[j - 1]
                    end = nets_cum_num_mutable_nodes[j]
                    self.num_network_passes_per_input_mask[i][start:end] = 1
        self.num_network_passes_per_input_mask = (
            self.num_network_passes_per_input_mask.bool()
        )
        log.debug("12. num_network_passes_per_input_mask")
        log.debug(self.num_network_passes_per_input_mask)
        log.debug(self.num_network_passes_per_input_mask.shape)

    def resample(self: "DynamicNets", indices: Tensor) -> None:
        """Resample networks according to the provided indices."""
        indices_list = indices.tolist()
        self.nets = [self.nets[i].clone() for i in indices_list]
        self.have_mutated = False
        self.prepare_for_computation()

    def reset(self: "DynamicNets") -> None:
        for net in self.nets:
            net.n_mean_m2_x_z = torch.zeros_like(net.n_mean_m2_x_z)
        self.prepare_for_computation()

    def __call__(
        self: "DynamicNets", x: Float[Tensor, "NN NI"]
    ) -> Float[Tensor, "NN NO"]:
        """Forward pass through the network population."""
        log.debug("13. obs")
        log.debug(x)
        log.debug(x.shape)

        flat_obs: Float[Tensor, "NNxNI"] = x.flatten()
        log.debug("14. flat_obs")
        log.debug(flat_obs)
        log.debug(flat_obs.shape)

        out: Float[Tensor, "TNNplus1"] = self.wrs.n_mean_m2_x_z[
            :, 4
        ].clone()  # z-score
        out[self.input_nodes_indices] = flat_obs
        log.debug("15. out")
        log.debug(out)
        log.debug(out.shape)

        out: Float[Tensor, "TNNplus1"] = self.wrs(out)
        log.debug("16. out")
        log.debug(out)
        log.debug(out.shape)

        for j in range(self.max_num_network_passes_per_input):

            log.debug(f"Pass {j}")
            log.debug("17. num_network_passes_per_input_mask[j]")
            log.debug(self.num_network_passes_per_input_mask[j])

            mapped_out: Float[Tensor, "TNMN 3"] = torch.gather(
                out, 0, self.flat_in_nodes_indices
            ).reshape(-1, 3)
            log.debug("18. mapped_out")
            log.debug(mapped_out)
            log.debug(mapped_out.shape)

            matmuld_mapped_out: Float[Tensor, "TNMN"] = (
                mapped_out * self.weights
            ).sum(dim=1)
            log.debug("19. matmuld_mapped_out")
            log.debug(matmuld_mapped_out)
            log.debug(matmuld_mapped_out.shape)
            log.debug("19a. out[mutable_nodes_indices]")
            log.debug(out[self.mutable_nodes_indices])
            log.debug(out[self.mutable_nodes_indices].shape)
            log.debug("19b. num_network_passes_per_input_mask[j]")
            log.debug(self.num_network_passes_per_input_mask[j])
            log.debug(self.num_network_passes_per_input_mask[j].shape)

            out[self.mutable_nodes_indices] = torch.where(
                self.num_network_passes_per_input_mask[j],
                matmuld_mapped_out,
                out[self.mutable_nodes_indices],
            )
            log.debug("20. out")
            log.debug(out)
            log.debug(out.shape)

            out: Float[Tensor, "TNNplus1"] = self.wrs(out)
            log.debug("21. out")
            log.debug(out)
            log.debug(out.shape)

        output: Float[Tensor, "NN NO"] = out[
            self.output_nodes_indices
        ].reshape(self.config.num_nets, self.config.num_outputs)
        log.debug("22. output")
        log.debug(output)
        log.debug(output.shape)

        return output
