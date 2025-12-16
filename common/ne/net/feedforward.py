"""Feedforward neural network models for neuroevolution.

Batched feedforward MLPs for efficient GPU-parallel computation across num_nets networks.
"""

import torch
from jaxtyping import Float
from torch import Tensor


class BatchedFeedforward:
    """Batched population of feedforward MLPs for efficient GPU-parallel computation.

    Stores num_nets networks as batched tensors for parallel forward passes and mutations.
    Used by evolutionary optimizers (GA/ES/CMA-ES).
    """

    def __init__(
        self,
        dimensions: list[int],
        num_nets: int,
        sigma_init: float = 1e-3,
        sigma_noise: float | None = 1e-2,
        device: str = "cpu",
    ) -> None:
        """Initialize batched feedforward population.

        Args:
            dimensions: Layer dimensions [input_size, hidden1, hidden2, ..., output_size]
            num_nets: Number of networks in batch
            sigma_init: Initial mutation strength (used for fixed sigma if sigma_noise=None)
            sigma_noise: Noise level for adaptive sigma. If None, uses fixed sigma_init.
            device: Device to run computation on
        """
        self.num_nets: int = num_nets
        self.dimensions: list[int] = dimensions
        self.num_layers: int = len(dimensions) - 1
        self.sigma_init: float = sigma_init
        self.sigma_noise: float | None = sigma_noise
        self.device: str = device

        # Initialize batched parameters for each layer
        self.weights: list[Float[Tensor, "num_nets out_dim in_dim"]] = []
        self.biases: list[Float[Tensor, "num_nets out_dim"]] = []

        for i in range(self.num_layers):
            in_dim: int = dimensions[i]
            out_dim: int = dimensions[i + 1]

            # Xavier initialization
            std: float = (1.0 / in_dim) ** 0.5

            weight: Float[Tensor, "num_nets out_dim in_dim"] = (
                torch.randn(num_nets, out_dim, in_dim, device=device) * std
            )
            bias: Float[Tensor, "num_nets out_dim"] = (
                torch.randn(num_nets, out_dim, device=device) * std
            )

            self.weights.append(weight)
            self.biases.append(bias)

        # Initialize adaptive sigmas if sigma_noise is provided
        if sigma_noise is not None:
            self.weight_sigmas: list[Float[Tensor, "num_nets out_dim in_dim"]] = []
            self.bias_sigmas: list[Float[Tensor, "num_nets out_dim"]] = []

            for weight, bias in zip(self.weights, self.biases):
                self.weight_sigmas.append(torch.full_like(weight, sigma_init))
                self.bias_sigmas.append(torch.full_like(bias, sigma_init))

    def forward_batch(
        self, x: Float[Tensor, "N input_size"]
    ) -> Float[Tensor, "num_nets N output_size"]:
        """Batched forward pass for all networks in parallel.

        Args:
            x: Input observations [N, input_size]

        Returns:
            Logits for all networks [num_nets, N, output_size]
        """
        # x: [N, input_size] -> expand to [num_nets, N, input_size]
        h: Float[Tensor, "num_nets N dim"] = x.unsqueeze(0).expand(
            self.num_nets, -1, -1
        )

        # Forward through each layer
        for i in range(self.num_layers):
            # Linear: [num_nets, N, in_dim] @ [num_nets, in_dim, out_dim]
            h = torch.bmm(h, self.weights[i].transpose(-1, -2))
            # Add bias: [num_nets, N, out_dim] + [num_nets, 1, out_dim]
            h = h + self.biases[i].unsqueeze(1)

            # Activation (tanh) for all layers except the last
            if i < self.num_layers - 1:
                h = torch.tanh(h)

        return h

    def mutate(self) -> None:
        """Apply mutations to all networks in parallel using adaptive or fixed sigma."""
        if self.sigma_noise is not None:
            # Adaptive sigma mutation - update sigmas then apply noise
            for i in range(self.num_layers):
                # Update weight sigma
                xi: Float[Tensor, "num_nets out_dim in_dim"] = (
                    torch.randn_like(self.weight_sigmas[i]) * self.sigma_noise
                )
                self.weight_sigmas[i] = self.weight_sigmas[i] * (1 + xi)
                eps: Float[Tensor, "num_nets out_dim in_dim"] = (
                    torch.randn_like(self.weights[i]) * self.weight_sigmas[i]
                )
                self.weights[i] = self.weights[i] + eps

                # Update bias sigma
                xi_bias: Float[Tensor, "num_nets out_dim"] = (
                    torch.randn_like(self.bias_sigmas[i]) * self.sigma_noise
                )
                self.bias_sigmas[i] = self.bias_sigmas[i] * (1 + xi_bias)
                eps_bias: Float[Tensor, "num_nets out_dim"] = (
                    torch.randn_like(self.biases[i]) * self.bias_sigmas[i]
                )
                self.biases[i] = self.biases[i] + eps_bias
        else:
            # Fixed sigma mutation
            for i in range(self.num_layers):
                self.weights[i] = (
                    self.weights[i] + torch.randn_like(self.weights[i]) * self.sigma_init
                )
                self.biases[i] = (
                    self.biases[i] + torch.randn_like(self.biases[i]) * self.sigma_init
                )

    def get_state_dict(self) -> dict:
        """Get network state for checkpointing.

        Returns:
            Dictionary containing all network state including dimensions, parameters, and sigmas
        """
        state = {
            "dimensions": self.dimensions,
            "num_nets": self.num_nets,
            "sigma_init": self.sigma_init,
            "sigma_noise": self.sigma_noise,
            "weights": self.weights,
            "biases": self.biases,
        }
        if self.sigma_noise is not None:
            state["weight_sigmas"] = self.weight_sigmas
            state["bias_sigmas"] = self.bias_sigmas
        return state

    def load_state_dict(self, state: dict) -> None:
        """Restore network state from checkpoint.

        Args:
            state: State dictionary from get_state_dict()
        """
        self.dimensions = state["dimensions"]
        self.num_nets = state["num_nets"]
        self.sigma_init = state["sigma_init"]
        self.sigma_noise = state["sigma_noise"]
        self.num_layers = len(self.dimensions) - 1
        self.weights = state["weights"]
        self.biases = state["biases"]
        if "weight_sigmas" in state:
            self.weight_sigmas = state["weight_sigmas"]
            self.bias_sigmas = state["bias_sigmas"]

    def get_parameters_flat(self) -> Float[Tensor, "num_nets num_params"]:
        """Get flattened parameter vectors for all networks.

        Used by ES/CMA-ES optimizers for parameter averaging.

        Returns:
            Flat parameters [num_nets, num_params] where num_params is the
            total number of parameters (weights + biases across all layers)
        """
        params_list: list[Float[Tensor, "num_nets ..."]] = []

        for i in range(self.num_layers):
            # Flatten weight matrix: [num_nets, out_dim, in_dim] -> [num_nets, out_dim * in_dim]
            params_list.append(self.weights[i].flatten(start_dim=1))
            # Biases are already [num_nets, out_dim]
            params_list.append(self.biases[i])

        # Concatenate all parameters: [num_nets, num_params]
        return torch.cat(params_list, dim=1)

    def set_parameters_flat(
        self, flat_params: Float[Tensor, "num_nets num_params"]
    ) -> None:
        """Set network parameters from flat vectors.

        Used by ES/CMA-ES optimizers to set averaged parameters.

        Args:
            flat_params: Flat parameters [num_nets, num_params]
        """
        idx: int = 0

        for i in range(self.num_layers):
            # Extract and reshape weights
            w_size: int = self.weights[i][0].numel()  # out_dim * in_dim
            w_shape: tuple[int, int, int] = self.weights[i].shape  # [num_nets, out_dim, in_dim]
            self.weights[i] = flat_params[:, idx : idx + w_size].reshape(w_shape).clone()
            idx += w_size

            # Extract biases
            b_size: int = self.biases[i][0].numel()  # out_dim
            self.biases[i] = flat_params[:, idx : idx + b_size].clone()
            idx += b_size

    def clone_network(self, indices: Tensor) -> None:
        """Clone networks at specified indices to fill population.

        Used by GA selection: top performers are cloned to replace losers.

        Args:
            indices: Indices of networks to clone [num_nets]
                     (indices may repeat for duplication)
        """
        for i in range(self.num_layers):
            self.weights[i] = self.weights[i][indices].clone()
            self.biases[i] = self.biases[i][indices].clone()

            if self.sigma_noise is not None:
                self.weight_sigmas[i] = self.weight_sigmas[i][indices].clone()
                self.bias_sigmas[i] = self.bias_sigmas[i][indices].clone()
