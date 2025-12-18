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


class FeedforwardStaticNets(BaseStaticNets):

    def __init__(
        self: "FeedforwardStaticNets", config: StaticNetsConfig
    ) -> None:
        super().__init__(config)
        layer_sizes: list[int] = (
            [self.config.num_inputs]
            + self.config.hidden_layer_sizes
            + [self.config.num_outputs]
        )
        self.weights: list[Float[Tensor, "NN LiIS LiOS"]] = []
        self.biases: list[Float[Tensor, "NN 1 LiOS"]] = []
        self.num_layers: int = len(layer_sizes) - 1
        for j in range(self.num_layers):
            layer_j_in_size: int = layer_sizes[j]
            layer_j_out_size: int = layer_sizes[j + 1]
            layer_j_std: float = (1.0 / layer_j_in_size) ** 0.5
            layer_j_weight: Float[Tensor, "NN LiIS LiOS"] = (
                torch.randn(
                    self.config.num_nets, layer_j_in_size, layer_j_out_size
                )
                * layer_j_std
            )
            layer_j_bias: Float[Tensor, "NN 1 LiOS"] = (
                torch.randn(self.config.num_nets, 1, layer_j_out_size)
                * layer_j_std
            )
            layer_j_weight.requires_grad = False
            layer_j_bias.requires_grad = False
            self.weights.append(layer_j_weight)
            self.biases.append(layer_j_bias)
        if self.config.sigma_sigma is not None:
            self.weight_sigmas: list[Float[Tensor, "NN LiIS LiOS"]] = []
            self.bias_sigmas: list[Float[Tensor, "NN 1 LiOS"]] = []
            for layer_j_weight, layer_j_bias in zip(self.weights, self.biases):
                layer_j_weight_sigma: Float[Tensor, "NN LiIS LiOS"] = (
                    torch.full_like(layer_j_weight, self.config.sigma)
                )
                layer_j_bias_sigma: Float[Tensor, "NN 1 LiOS"] = (
                    torch.full_like(layer_j_bias, self.config.sigma)
                )
                layer_j_weight_sigma.requires_grad = False
                layer_j_bias_sigma.requires_grad = False
                self.weight_sigmas.append(layer_j_weight_sigma)
                self.bias_sigmas.append(layer_j_bias_sigma)

    def mutate(self: "FeedforwardStaticNets") -> None:
        for j in range(self.num_layers):
            if self.config.sigma_sigma is not None:
                layer_j_xi: Float[Tensor, "NN LiIS LiOS"] = (
                    torch.randn_like(self.weight_sigmas[j])
                    * self.config.sigma_sigma
                )
                self.weight_sigmas[j]: Float[Tensor, "NN LiIS LiOS"] = (
                    self.weight_sigmas[j] * (1 + layer_j_xi)
                )
                layer_j_weight_sigma: Float[Tensor, "NN LiIS LiOS"] = (
                    self.weight_sigmas[j]
                )
                layer_j_xi_bias: Float[Tensor, "NN 1 LiOS"] = (
                    torch.randn_like(self.bias_sigmas[j])
                    * self.config.sigma_sigma
                )
                self.bias_sigmas[j]: Float[Tensor, "NN 1 LiOS"] = (
                    self.bias_sigmas[j] * (1 + layer_j_xi_bias)
                )
                layer_j_bias_sigma: Float[Tensor, "NN 1 LiOS"] = (
                    self.bias_sigmas[j]
                )
            else:
                layer_j_weight_sigma: float = self.config.sigma
                layer_j_bias_sigma: float = self.config.sigma
            self.weights[j]: Float[Tensor, "NN LiIS LiOS"] = (
                self.weights[j]
                + torch.randn_like(self.weights[j]) * layer_j_weight_sigma
            )
            self.biases[j]: Float[Tensor, "NN 1 LiOS"] = (
                self.biases[j]
                + torch.randn_like(self.biases[j]) * layer_j_bias_sigma
            )

    def resample(self: "FeedforwardStaticNets", indices: Tensor) -> None:
        for j in range(self.num_layers):
            self.weights[j]: Float[Tensor, "NN LiIS LiOS"] = self.weights[j][
                indices
            ]
            self.biases[j]: Float[Tensor, "NN 1 LiOS"] = self.biases[j][
                indices
            ]
            if self.config.sigma_sigma is not None:
                self.weight_sigmas[j]: Float[Tensor, "NN LiIS LiOS"] = (
                    self.weight_sigmas[j][indices]
                )
                self.bias_sigmas[j]: Float[Tensor, "NN 1 LiOS"] = (
                    self.bias_sigmas[j][indices]
                )

    def __call__(
        self: "FeedforwardStaticNets", x: Float[Tensor, "NN BS NI"]
    ) -> Float[Tensor, "NN BS NO"]:
        for j in range(self.num_layers):
            x: Float[Tensor, "NN BS LiOS"] = torch.bmm(x, self.weights[j])
            x: Float[Tensor, "NN BS LiOS"] = x + self.biases[j]
            if j < self.num_layers - 1:
                x: Float[Tensor, "NN BS LiOS"] = torch.tanh(x)
        return x
