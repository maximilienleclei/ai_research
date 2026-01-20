"""Shapes:

NN: num_nets
BS: batch_size
NI: num_inputs
NO: num_outputs
LiIS: layer_i_in_size
LiOS: layer_i_out_size
"""

import torch
from jaxtyping import Float
from torch import Tensor

from common.ne.popu.nets.static.base import BaseStaticNets, StaticNetsConfig


class RecurrentStaticNets(BaseStaticNets):

    def __init__(
        self: "RecurrentStaticNets", config: StaticNetsConfig
    ) -> None:
        super().__init__(config)
        layer_sizes: list[int] = (
            [self.config.num_inputs]
            + self.config.hidden_layer_sizes
            + [self.config.num_outputs]
        )
        self.W_ih_weights: list[Float[Tensor, "NN LiOS LiIS"]] = []
        self.W_ih_biases: list[Float[Tensor, "NN 1 LiOS"]] = []
        self.W_hh_weights: list[Float[Tensor, "NN LiOS LiOS"]] = []
        self.W_hh_biases: list[Float[Tensor, "NN 1 LiOS"]] = []
        self.num_layers: int = len(layer_sizes) - 1
        for j in range(self.num_layers):
            layer_j_in_size: int = layer_sizes[j]
            layer_j_out_size: int = layer_sizes[j + 1]
            layer_j_ih_std: float = (1.0 / layer_j_in_size) ** 0.5
            layer_j_hh_std: float = (1.0 / layer_j_out_size) ** 0.5
            layer_j_W_ih: Float[Tensor, "NN LiOS LiIS"] = (
                torch.randn(
                    self.config.num_nets, layer_j_out_size, layer_j_in_size,
                    device=self.config.device
                )
                * layer_j_ih_std
            )
            layer_j_b_ih: Float[Tensor, "NN 1 LiOS"] = (
                torch.randn(self.config.num_nets, 1, layer_j_out_size, device=self.config.device)
                * layer_j_ih_std
            )
            layer_j_W_hh: Float[Tensor, "NN LiOS LiOS"] = (
                torch.randn(
                    self.config.num_nets, layer_j_out_size, layer_j_out_size,
                    device=self.config.device
                )
                * layer_j_hh_std
            )
            layer_j_b_hh: Float[Tensor, "NN 1 LiOS"] = (
                torch.randn(self.config.num_nets, 1, layer_j_out_size, device=self.config.device)
                * layer_j_hh_std
            )
            self.W_ih_weights.append(layer_j_W_ih)
            self.W_ih_biases.append(layer_j_b_ih)
            self.W_hh_weights.append(layer_j_W_hh)
            self.W_hh_biases.append(layer_j_b_hh)
        if self.config.sigma_sigma is not None:
            self.W_ih_weight_sigmas: list[Float[Tensor, "NN LiOS LiIS"]] = []
            self.W_ih_bias_sigmas: list[Float[Tensor, "NN 1 LiOS"]] = []
            self.W_hh_weight_sigmas: list[Float[Tensor, "NN LiOS LiOS"]] = []
            self.W_hh_bias_sigmas: list[Float[Tensor, "NN 1 LiOS"]] = []
            for layer_j_W_ih, layer_j_b_ih, layer_j_W_hh, layer_j_b_hh in zip(
                self.W_ih_weights,
                self.W_ih_biases,
                self.W_hh_weights,
                self.W_hh_biases,
            ):
                self.W_ih_weight_sigmas.append(
                    torch.full_like(layer_j_W_ih, self.config.sigma)
                )
                self.W_ih_bias_sigmas.append(
                    torch.full_like(layer_j_b_ih, self.config.sigma)
                )
                self.W_hh_weight_sigmas.append(
                    torch.full_like(layer_j_W_hh, self.config.sigma)
                )
                self.W_hh_bias_sigmas.append(
                    torch.full_like(layer_j_b_hh, self.config.sigma)
                )

    def mutate(self: "RecurrentStaticNets") -> None:
        for j in range(self.num_layers):
            if self.config.sigma_sigma is not None:
                layer_j_xi_W_ih: Float[Tensor, "NN LiOS LiIS"] = (
                    torch.randn_like(self.W_ih_weight_sigmas[j])
                    * self.config.sigma_sigma
                )
                self.W_ih_weight_sigmas[j]: Float[Tensor, "NN LiOS LiIS"] = (
                    self.W_ih_weight_sigmas[j] * (1 + layer_j_xi_W_ih)
                )
                layer_j_W_ih_sigma: Float[Tensor, "NN LiOS LiIS"] = (
                    self.W_ih_weight_sigmas[j]
                )
                layer_j_xi_b_ih: Float[Tensor, "NN 1 LiOS"] = (
                    torch.randn_like(self.W_ih_bias_sigmas[j])
                    * self.config.sigma_sigma
                )
                self.W_ih_bias_sigmas[j]: Float[Tensor, "NN 1 LiOS"] = (
                    self.W_ih_bias_sigmas[j] * (1 + layer_j_xi_b_ih)
                )
                layer_j_b_ih_sigma: Float[Tensor, "NN 1 LiOS"] = (
                    self.W_ih_bias_sigmas[j]
                )
                layer_j_xi_W_hh: Float[Tensor, "NN LiOS LiOS"] = (
                    torch.randn_like(self.W_hh_weight_sigmas[j])
                    * self.config.sigma_sigma
                )
                self.W_hh_weight_sigmas[j]: Float[Tensor, "NN LiOS LiOS"] = (
                    self.W_hh_weight_sigmas[j] * (1 + layer_j_xi_W_hh)
                )
                layer_j_W_hh_sigma: Float[Tensor, "NN LiOS LiOS"] = (
                    self.W_hh_weight_sigmas[j]
                )
                layer_j_xi_b_hh: Float[Tensor, "NN 1 LiOS"] = (
                    torch.randn_like(self.W_hh_bias_sigmas[j])
                    * self.config.sigma_sigma
                )
                self.W_hh_bias_sigmas[j]: Float[Tensor, "NN 1 LiOS"] = (
                    self.W_hh_bias_sigmas[j] * (1 + layer_j_xi_b_hh)
                )
                layer_j_b_hh_sigma: Float[Tensor, "NN 1 LiOS"] = (
                    self.W_hh_bias_sigmas[j]
                )
            else:
                layer_j_W_ih_sigma: float = self.config.sigma
                layer_j_b_ih_sigma: float = self.config.sigma
                layer_j_W_hh_sigma: float = self.config.sigma
                layer_j_b_hh_sigma: float = self.config.sigma
            self.W_ih_weights[j]: Float[Tensor, "NN LiOS LiIS"] = (
                self.W_ih_weights[j]
                + torch.randn_like(self.W_ih_weights[j]) * layer_j_W_ih_sigma
            )
            self.W_ih_biases[j]: Float[Tensor, "NN 1 LiOS"] = (
                self.W_ih_biases[j]
                + torch.randn_like(self.W_ih_biases[j]) * layer_j_b_ih_sigma
            )
            self.W_hh_weights[j]: Float[Tensor, "NN LiOS LiOS"] = (
                self.W_hh_weights[j]
                + torch.randn_like(self.W_hh_weights[j]) * layer_j_W_hh_sigma
            )
            self.W_hh_biases[j]: Float[Tensor, "NN 1 LiOS"] = (
                self.W_hh_biases[j]
                + torch.randn_like(self.W_hh_biases[j]) * layer_j_b_hh_sigma
            )

    def resample(self: "RecurrentStaticNets", indices: Tensor) -> None:
        for j in range(self.num_layers):
            self.W_ih_weights[j]: Float[Tensor, "NN LiOS LiIS"] = (
                self.W_ih_weights[j][indices]
            )
            self.W_ih_biases[j]: Float[Tensor, "NN 1 LiOS"] = self.W_ih_biases[
                j
            ][indices]
            self.W_hh_weights[j]: Float[Tensor, "NN LiOS LiOS"] = (
                self.W_hh_weights[j][indices]
            )
            self.W_hh_biases[j]: Float[Tensor, "NN 1 LiOS"] = self.W_hh_biases[
                j
            ][indices]
            if self.config.sigma_sigma is not None:
                self.W_ih_weight_sigmas[j]: Float[Tensor, "NN LiOS LiIS"] = (
                    self.W_ih_weight_sigmas[j][indices]
                )
                self.W_ih_bias_sigmas[j]: Float[Tensor, "NN 1 LiOS"] = (
                    self.W_ih_bias_sigmas[j][indices]
                )
                self.W_hh_weight_sigmas[j]: Float[Tensor, "NN LiOS LiOS"] = (
                    self.W_hh_weight_sigmas[j][indices]
                )
                self.W_hh_bias_sigmas[j]: Float[Tensor, "NN 1 LiOS"] = (
                    self.W_hh_bias_sigmas[j][indices]
                )

    def reset(self: "RecurrentStaticNets") -> None:
        layer_sizes: list[int] = (
            [self.config.num_inputs]
            + self.config.hidden_layer_sizes
            + [self.config.num_outputs]
        )
        self.hidden_states: list[Float[Tensor, "NN BS LiOS"]] = []
        for j in range(self.num_layers):
            layer_j_out_size: int = layer_sizes[j + 1]
            layer_j_h: Float[Tensor, "NN BS LiOS"] = torch.zeros(
                self.config.num_nets, 1, layer_j_out_size, device=self.config.device
            )
            self.hidden_states.append(layer_j_h)

    def __call__(
        self: "RecurrentStaticNets", x: Float[Tensor, "NN BS NI"]
    ) -> Float[Tensor, "NN BS NO"]:
        for j in range(self.num_layers):
            layer_j_ih: Float[Tensor, "NN BS LiOS"] = torch.bmm(
                x, self.W_ih_weights[j].transpose(-1, -2)
            )
            layer_j_ih: Float[Tensor, "NN BS LiOS"] = (
                layer_j_ih + self.W_ih_biases[j]
            )
            layer_j_hh: Float[Tensor, "NN BS LiOS"] = torch.bmm(
                self.hidden_states[j], self.W_hh_weights[j].transpose(-1, -2)
            )
            layer_j_hh: Float[Tensor, "NN BS LiOS"] = (
                layer_j_hh + self.W_hh_biases[j]
            )
            layer_j_h_new: Float[Tensor, "NN BS LiOS"] = torch.tanh(
                layer_j_ih + layer_j_hh
            )
            x = self.hidden_states[j] = layer_j_h_new
        return x
