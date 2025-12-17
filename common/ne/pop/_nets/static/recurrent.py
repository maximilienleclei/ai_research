"""Shapes:

NN: num_nets
BS: batch_size
ID: in_dim
OD: out_dim
LiID: layer_i_in_dim
LiOD: layer_i_out_dim
"""

import torch
from jaxtyping import Float
from torch import Tensor
from common.ne.pop._nets.static.base import BaseStaticNets, StaticNetsConfig

class RecurrentStaticNets(BaseStaticNets):

    def __init__(self, config: StaticNetsConfig):
        self.config: StaticNetsConfig = config
        self.W_ih_weights: list[Float[Tensor, "NN LiOD LiID"]] = []
        self.W_ih_biases: list[Float[Tensor, "NN 1 LiOD"]] = []
        self.W_hh_weights: list[Float[Tensor, "NN LiOD LiOD"]] = []
        self.W_hh_biases: list[Float[Tensor, "NN 1 LiOD"]] = []
        self.num_layers: int = len(self.config.layer_dims) - 1
        for j in range(self.num_layers):
            layer_j_in_dim: int = self.config.layer_dims[j]
            layer_j_out_dim: int = self.config.layer_dims[j + 1]
            layer_j_ih_std: float = (1.0 / layer_j_in_dim) ** 0.5
            layer_j_hh_std: float = (1.0 / layer_j_out_dim) ** 0.5
            layer_j_W_ih: Float[Tensor, "NN LiOD LiID"] = torch.randn(self.config.num_nets, layer_j_out_dim, layer_j_in_dim) * layer_j_ih_std
            layer_j_b_ih: Float[Tensor, "NN 1 LiOD"] = torch.randn(self.config.num_nets, 1, layer_j_out_dim) * layer_j_ih_std
            layer_j_W_hh: Float[Tensor, "NN LiOD LiOD"] = torch.randn(self.config.num_nets, layer_j_out_dim, layer_j_out_dim) * layer_j_hh_std
            layer_j_b_hh: Float[Tensor, "NN 1 LiOD"] = torch.randn(self.config.num_nets, 1, layer_j_out_dim) * layer_j_hh_std
            self.W_ih_weights.append(layer_j_W_ih)
            self.W_ih_biases.append(layer_j_b_ih)
            self.W_hh_weights.append(layer_j_W_hh)
            self.W_hh_biases.append(layer_j_b_hh)
        if self.config.sigma_sigma is not None:
            self.W_ih_weight_sigmas: list[Float[Tensor, "NN LiOD LiID"]] = []
            self.W_ih_bias_sigmas: list[Float[Tensor, "NN 1 LiOD"]] = []
            self.W_hh_weight_sigmas: list[Float[Tensor, "NN LiOD LiOD"]] = []
            self.W_hh_bias_sigmas: list[Float[Tensor, "NN 1 LiOD"]] = []
            for layer_j_W_ih, layer_j_b_ih, layer_j_W_hh, layer_j_b_hh in zip(self.W_ih_weights, self.W_ih_biases, self.W_hh_weights, self.W_hh_biases):
                self.W_ih_weight_sigmas.append(torch.full_like(layer_j_W_ih, self.config.sigma))
                self.W_ih_bias_sigmas.append(torch.full_like(layer_j_b_ih, self.config.sigma))
                self.W_hh_weight_sigmas.append(torch.full_like(layer_j_W_hh, self.config.sigma))
                self.W_hh_bias_sigmas.append(torch.full_like(layer_j_b_hh, self.config.sigma))

    def mutate(self) -> None:
        for j in range(self.num_layers):
            if self.config.sigma_sigma is not None:
                layer_j_xi_W_ih: Float[Tensor, "NN LiOD LiID"] = torch.randn_like(self.W_ih_weight_sigmas[j]) * self.config.sigma_sigma
                self.W_ih_weight_sigmas[j]: Float[Tensor, "NN LiOD LiID"] = self.W_ih_weight_sigmas[j] * (1 + layer_j_xi_W_ih)
                layer_j_W_ih_sigma: Float[Tensor, "NN LiOD LiID"] = self.W_ih_weight_sigmas[j]
                layer_j_xi_b_ih: Float[Tensor, "NN 1 LiOD"] = torch.randn_like(self.W_ih_bias_sigmas[j]) * self.config.sigma_sigma
                self.W_ih_bias_sigmas[j]: Float[Tensor, "NN 1 LiOD"] = self.W_ih_bias_sigmas[j] * (1 + layer_j_xi_b_ih)
                layer_j_b_ih_sigma: Float[Tensor, "NN 1 LiOD"] = self.W_ih_bias_sigmas[j]
                layer_j_xi_W_hh: Float[Tensor, "NN LiOD LiOD"] = torch.randn_like(self.W_hh_weight_sigmas[j]) * self.config.sigma_sigma
                self.W_hh_weight_sigmas[j]: Float[Tensor, "NN LiOD LiOD"] = self.W_hh_weight_sigmas[j] * (1 + layer_j_xi_W_hh)
                layer_j_W_hh_sigma: Float[Tensor, "NN LiOD LiOD"] = self.W_hh_weight_sigmas[j]
                layer_j_xi_b_hh: Float[Tensor, "NN 1 LiOD"] = torch.randn_like(self.W_hh_bias_sigmas[j]) * self.config.sigma_sigma
                self.W_hh_bias_sigmas[j]: Float[Tensor, "NN 1 LiOD"] = self.W_hh_bias_sigmas[j] * (1 + layer_j_xi_b_hh)
                layer_j_b_hh_sigma: Float[Tensor, "NN 1 LiOD"] = self.W_hh_bias_sigmas[j]
            else:
                layer_j_W_ih_sigma: float = self.config.sigma
                layer_j_b_ih_sigma: float = self.config.sigma
                layer_j_W_hh_sigma: float = self.config.sigma
                layer_j_b_hh_sigma: float = self.config.sigma
            self.W_ih_weights[j]: Float[Tensor, "NN LiOD LiID"] = self.W_ih_weights[j] + torch.randn_like(self.W_ih_weights[j]) * layer_j_W_ih_sigma
            self.W_ih_biases[j]: Float[Tensor, "NN 1 LiOD"] = self.W_ih_biases[j] + torch.randn_like(self.W_ih_biases[j]) * layer_j_b_ih_sigma
            self.W_hh_weights[j]: Float[Tensor, "NN LiOD LiOD"] = self.W_hh_weights[j] + torch.randn_like(self.W_hh_weights[j]) * layer_j_W_hh_sigma
            self.W_hh_biases[j]: Float[Tensor, "NN 1 LiOD"] = self.W_hh_biases[j] + torch.randn_like(self.W_hh_biases[j]) * layer_j_b_hh_sigma

    def __call__(self, x: Float[Tensor, "NN BS ID"]) -> Float[Tensor, "NN BS OD"]:
        hidden_states: list[Float[Tensor, "NN BS LiOD"]] = []
        for j in range(self.num_layers):
            layer_j_out_dim: int = self.config.layer_dims[j + 1]
            layer_j_h: Float[Tensor, "NN BS LiOD"] = torch.zeros(self.config.num_nets, x.shape[1], layer_j_out_dim)
            hidden_states.append(layer_j_h)
        layer_input: Float[Tensor, "NN BS dim"] = x
        for j in range(self.num_layers):
            layer_j_ih: Float[Tensor, "NN BS LiOD"] = torch.bmm(layer_input, self.W_ih_weights[j].transpose(-1, -2))
            layer_j_ih: Float[Tensor, "NN BS LiOD"] = layer_j_ih + self.W_ih_biases[j]
            layer_j_hh: Float[Tensor, "NN BS LiOD"] = torch.bmm(hidden_states[j], self.W_hh_weights[j].transpose(-1, -2))
            layer_j_hh: Float[Tensor, "NN BS LiOD"] = layer_j_hh + self.W_hh_biases[j]
            layer_j_h_new: Float[Tensor, "NN BS LiOD"] = torch.tanh(layer_j_ih + layer_j_hh)
            hidden_states[j] = layer_j_h_new
            layer_input = layer_j_h_new
        return hidden_states[-1]
