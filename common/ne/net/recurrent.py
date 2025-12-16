"""Recurrent neural network models for neuroevolution.

Batched stacked RNNs with either frozen reservoir or trainable rank-1 recurrent weights
for efficient GPU-parallel computation across num_nets networks.
"""

import math

import torch
from jaxtyping import Float
from torch import Tensor


class BatchedRecurrent:
    """Batched population of stacked recurrent MLPs for efficient GPU-parallel computation.

    Supports two recurrent architectures:
    - 'reservoir': Frozen random recurrent weights (echo state network)
    - 'trainable': Trainable rank-1 factorized recurrent weights (u ⊗ v^T)

    All layers (including output) are recurrent.
    Stores num_nets networks as batched tensors for parallel forward passes and mutations.
    Used by evolutionary optimizers (GA/ES/CMA-ES).
    """

    def __init__(
        self,
        dimensions: list[int],
        num_nets: int,
        model_type: str = "reservoir",
        sigma_init: float = 1e-3,
        sigma_noise: float | None = 1e-2,
        device: str = "cpu",
    ) -> None:
        """Initialize batched stacked recurrent population.

        Args:
            dimensions: Layer dimensions [input_size, layer1, layer2, ..., output_size]
                       All layers (including output) are recurrent
            num_nets: Number of networks in batch
            model_type: 'reservoir' (frozen W_hh) or 'trainable' (trainable rank-1 W_hh)
            sigma_init: Initial mutation strength (used for fixed sigma if sigma_noise=None)
            sigma_noise: Noise level for adaptive sigma. If None, uses fixed sigma_init.
            device: Device to run computation on
        """
        self.num_nets: int = num_nets
        self.dimensions: list[int] = dimensions
        self.num_layers: int = len(dimensions) - 1  # All layers are recurrent
        self.model_type: str = model_type
        self.sigma_init: float = sigma_init
        self.sigma_noise: float | None = sigma_noise
        self.device: str = device

        # Initialize input-to-hidden and recurrent weights for each layer
        self.W_ih_weights: list[Float[Tensor, "num_nets layer_size input_size"]] = []
        self.W_ih_biases: list[Float[Tensor, "num_nets layer_size"]] = []

        # Recurrent connections - structure depends on model_type
        if model_type == "reservoir":
            self.W_hh: list[Float[Tensor, "num_nets layer_size layer_size"]] = []
        elif model_type == "trainable":
            self.u: list[Float[Tensor, "num_nets layer_size"]] = []
            self.v: list[Float[Tensor, "num_nets layer_size"]] = []
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Initialize each RNN layer (including output layer)
        for layer_idx in range(self.num_layers):
            in_size: int = dimensions[layer_idx]
            layer_size: int = dimensions[layer_idx + 1]

            # Input-to-hidden weights
            ih_std: float = (1.0 / in_size) ** 0.5
            W_ih: Float[Tensor, "num_nets layer_size input_size"] = (
                torch.randn(num_nets, layer_size, in_size, device=device) * ih_std
            )
            b_ih: Float[Tensor, "num_nets layer_size"] = (
                torch.randn(num_nets, layer_size, device=device) * ih_std
            )
            self.W_ih_weights.append(W_ih)
            self.W_ih_biases.append(b_ih)

            # Recurrent connections
            hh_std: float = 1.0 / math.sqrt(layer_size)
            if model_type == "reservoir":
                W_hh: Float[Tensor, "num_nets layer_size layer_size"] = (
                    torch.randn(num_nets, layer_size, layer_size, device=device)
                    * hh_std
                )
                self.W_hh.append(W_hh)
            elif model_type == "trainable":
                u: Float[Tensor, "num_nets layer_size"] = (
                    torch.randn(num_nets, layer_size, device=device) * hh_std
                )
                v: Float[Tensor, "num_nets layer_size"] = (
                    torch.randn(num_nets, layer_size, device=device) * hh_std
                )
                self.u.append(u)
                self.v.append(v)

        # Initialize adaptive sigmas if sigma_noise is provided
        if sigma_noise is not None:
            self.W_ih_weight_sigmas: list[
                Float[Tensor, "num_nets layer_size input_size"]
            ] = []
            self.W_ih_bias_sigmas: list[Float[Tensor, "num_nets layer_size"]] = []

            for W_ih, b_ih in zip(self.W_ih_weights, self.W_ih_biases):
                self.W_ih_weight_sigmas.append(torch.full_like(W_ih, sigma_init))
                self.W_ih_bias_sigmas.append(torch.full_like(b_ih, sigma_init))

            # Recurrent weight sigmas (trainable mode only)
            if model_type == "trainable":
                self.u_sigmas: list[Float[Tensor, "num_nets layer_size"]] = []
                self.v_sigmas: list[Float[Tensor, "num_nets layer_size"]] = []
                for u_vec, v_vec in zip(self.u, self.v):
                    self.u_sigmas.append(torch.full_like(u_vec, sigma_init))
                    self.v_sigmas.append(torch.full_like(v_vec, sigma_init))

    def forward_batch_step(
        self,
        x: Float[Tensor, "num_nets BS input_size"],
        h_states: list[Float[Tensor, "num_nets BS layer_size"]],
    ) -> tuple[
        Float[Tensor, "num_nets BS output_size"],
        list[Float[Tensor, "num_nets BS layer_size"]],
    ]:
        """Single timestep forward pass for all networks in parallel.

        Args:
            x: Input observations [num_nets, batch_size, input_size]
            h_states: List of previous hidden states, one per layer
                     [num_nets, batch_size, layer_size_i] for each layer i

        Returns:
            output: Network output (last layer's activations) [num_nets, batch_size, output_size]
            h_new_states: List of new hidden states for each layer
        """
        h_new_states: list[Float[Tensor, "num_nets BS layer_size"]] = []
        layer_input: Float[Tensor, "num_nets BS dim"] = x

        # Process through each recurrent layer (including output layer)
        for layer_idx in range(self.num_layers):
            h_prev: Float[Tensor, "num_nets BS layer_size"] = h_states[layer_idx]

            # Input-to-hidden: [num_nets, BS, input_size] @ [num_nets, input_size, layer_size]
            ih_out: Float[Tensor, "num_nets BS layer_size"] = torch.bmm(
                layer_input, self.W_ih_weights[layer_idx].transpose(-1, -2)
            )
            ih_out = ih_out + self.W_ih_biases[layer_idx].unsqueeze(1)

            # Recurrent connection
            if self.model_type == "reservoir":
                # Frozen reservoir: h @ W_hh
                hh_out: Float[Tensor, "num_nets BS layer_size"] = torch.bmm(
                    h_prev, self.W_hh[layer_idx].transpose(-1, -2)
                )
            elif self.model_type == "trainable":
                # Rank-1: (u ⊗ v^T) @ h = u * (v^T @ h)
                v_expanded: Float[Tensor, "num_nets BS layer_size"] = (
                    self.v[layer_idx].unsqueeze(1).expand(-1, layer_input.shape[1], -1)
                )
                v_dot_h: Float[Tensor, "num_nets BS"] = (h_prev * v_expanded).sum(
                    dim=2
                )
                u_expanded: Float[Tensor, "num_nets BS layer_size"] = (
                    self.u[layer_idx].unsqueeze(1).expand(-1, layer_input.shape[1], -1)
                )
                hh_out: Float[Tensor, "num_nets BS layer_size"] = (
                    u_expanded * v_dot_h.unsqueeze(2)
                )

            # New hidden state with tanh activation
            h_new: Float[Tensor, "num_nets BS layer_size"] = torch.tanh(
                ih_out + hh_out
            )
            h_new_states.append(h_new)

            # Next layer's input is this layer's output
            layer_input = h_new

        # Output is the last layer's activation
        output: Float[Tensor, "num_nets BS output_size"] = h_new_states[-1]

        return output, h_new_states

    def forward_batch_sequence(
        self,
        x: Float[Tensor, "seq_len input_size"],
        h_0: list[Float[Tensor, "num_nets layer_size"]] | None = None,
    ) -> tuple[
        Float[Tensor, "num_nets seq_len output_size"],
        list[Float[Tensor, "num_nets layer_size"]],
    ]:
        """Batched forward pass for sequence across all networks.

        Args:
            x: Input sequence [seq_len, input_size]
            h_0: Initial hidden states as list (one per layer) or None
                 Each element: [num_nets, layer_size]

        Returns:
            outputs: Network outputs [num_nets, seq_len, output_size]
            h_final_states: List of final hidden states for each layer
        """
        seq_len: int = x.shape[0]

        # Initialize hidden states
        if h_0 is None:
            h_states: list[Float[Tensor, "num_nets 1 layer_size"]] = []
            for layer_idx in range(self.num_layers):
                layer_size: int = self.dimensions[layer_idx + 1]
                h_states.append(
                    torch.zeros(self.num_nets, 1, layer_size, device=self.device)
                )
        else:
            h_states = [h.unsqueeze(1) for h in h_0]

        # Process sequence timestep by timestep
        all_outputs: list[Float[Tensor, "num_nets 1 output_size"]] = []
        for t in range(seq_len):
            # Expand input for all networks: [input_size] -> [num_nets, 1, input_size]
            x_t: Float[Tensor, "num_nets 1 input_size"] = (
                x[t].unsqueeze(0).unsqueeze(0).expand(self.num_nets, 1, -1)
            )
            output_t, h_states = self.forward_batch_step(x_t, h_states)
            all_outputs.append(output_t)

        # Stack outputs: [num_nets, seq_len, output_size]
        outputs: Float[Tensor, "num_nets seq_len output_size"] = torch.cat(
            all_outputs, dim=1
        )

        # Final hidden states: squeeze batch dimension
        h_final_states: list[Float[Tensor, "num_nets layer_size"]] = [
            h.squeeze(1) for h in h_states
        ]

        return outputs, h_final_states

    def mutate(self) -> None:
        """Apply mutations to all networks in parallel using adaptive or fixed sigma."""
        if self.sigma_noise is not None:
            self._mutate_adaptive()
        else:
            self._mutate_fixed()

    def _mutate_adaptive(self) -> None:
        """Adaptive sigma mutation."""
        # Mutate each layer's parameters (including output layer)
        for layer_idx in range(self.num_layers):
            # W_ih_weight
            xi: Float[Tensor, "num_nets layer_size input_size"] = (
                torch.randn_like(self.W_ih_weight_sigmas[layer_idx]) * self.sigma_noise
            )
            self.W_ih_weight_sigmas[layer_idx] = self.W_ih_weight_sigmas[layer_idx] * (
                1 + xi
            )
            eps: Float[Tensor, "num_nets layer_size input_size"] = (
                torch.randn_like(self.W_ih_weights[layer_idx])
                * self.W_ih_weight_sigmas[layer_idx]
            )
            self.W_ih_weights[layer_idx] = self.W_ih_weights[layer_idx] + eps

            # W_ih_bias
            xi_bias: Float[Tensor, "num_nets layer_size"] = (
                torch.randn_like(self.W_ih_bias_sigmas[layer_idx]) * self.sigma_noise
            )
            self.W_ih_bias_sigmas[layer_idx] = self.W_ih_bias_sigmas[layer_idx] * (
                1 + xi_bias
            )
            eps_bias: Float[Tensor, "num_nets layer_size"] = (
                torch.randn_like(self.W_ih_biases[layer_idx])
                * self.W_ih_bias_sigmas[layer_idx]
            )
            self.W_ih_biases[layer_idx] = self.W_ih_biases[layer_idx] + eps_bias

            # Trainable recurrent weights (u, v) if applicable
            if self.model_type == "trainable":
                # u vector
                xi_u: Float[Tensor, "num_nets layer_size"] = (
                    torch.randn_like(self.u_sigmas[layer_idx]) * self.sigma_noise
                )
                self.u_sigmas[layer_idx] = self.u_sigmas[layer_idx] * (1 + xi_u)
                eps_u: Float[Tensor, "num_nets layer_size"] = (
                    torch.randn_like(self.u[layer_idx]) * self.u_sigmas[layer_idx]
                )
                self.u[layer_idx] = self.u[layer_idx] + eps_u

                # v vector
                xi_v: Float[Tensor, "num_nets layer_size"] = (
                    torch.randn_like(self.v_sigmas[layer_idx]) * self.sigma_noise
                )
                self.v_sigmas[layer_idx] = self.v_sigmas[layer_idx] * (1 + xi_v)
                eps_v: Float[Tensor, "num_nets layer_size"] = (
                    torch.randn_like(self.v[layer_idx]) * self.v_sigmas[layer_idx]
                )
                self.v[layer_idx] = self.v[layer_idx] + eps_v

    def _mutate_fixed(self) -> None:
        """Fixed sigma mutation."""
        # Mutate each layer's parameters (including output layer)
        for layer_idx in range(self.num_layers):
            self.W_ih_weights[layer_idx] = (
                self.W_ih_weights[layer_idx]
                + torch.randn_like(self.W_ih_weights[layer_idx]) * self.sigma_init
            )
            self.W_ih_biases[layer_idx] = (
                self.W_ih_biases[layer_idx]
                + torch.randn_like(self.W_ih_biases[layer_idx]) * self.sigma_init
            )

            if self.model_type == "trainable":
                self.u[layer_idx] = (
                    self.u[layer_idx]
                    + torch.randn_like(self.u[layer_idx]) * self.sigma_init
                )
                self.v[layer_idx] = (
                    self.v[layer_idx]
                    + torch.randn_like(self.v[layer_idx]) * self.sigma_init
                )

    def save_hidden_states(self) -> list[Float[Tensor, "num_nets layer_size"]]:
        """Save current hidden states for persistence across generations/episodes.

        Returns:
            List of hidden state tensors, one per layer [num_nets, layer_size_i]
        """
        if not hasattr(self, "_current_hidden_states") or self._current_hidden_states is None:
            # Initialize with zeros if no states exist
            return [
                torch.zeros(self.num_nets, self.dimensions[i + 1], device=self.device)
                for i in range(self.num_layers)
            ]
        return [h.clone() for h in self._current_hidden_states]

    def restore_hidden_states(
        self, states: list[Float[Tensor, "num_nets layer_size"]]
    ) -> None:
        """Restore hidden states from previous evaluation.

        Args:
            states: List of hidden state tensors from save_hidden_states()
        """
        self._current_hidden_states = [s.clone() for s in states]

    def reset_hidden_states(self) -> None:
        """Reset all hidden states to zero."""
        self._current_hidden_states = None

    def get_state_dict(self) -> dict:
        """Get full network state for checkpointing.

        Returns:
            Dictionary containing dimensions, parameters, sigmas, and optional hidden states
        """
        state = {
            "dimensions": self.dimensions,
            "num_nets": self.num_nets,
            "model_type": self.model_type,
            "sigma_init": self.sigma_init,
            "sigma_noise": self.sigma_noise,
            "W_ih_weights": self.W_ih_weights,
            "W_ih_biases": self.W_ih_biases,
        }

        if self.model_type == "reservoir":
            state["W_hh"] = self.W_hh
        elif self.model_type == "trainable":
            state["u"] = self.u
            state["v"] = self.v

        if self.sigma_noise is not None:
            state["W_ih_weight_sigmas"] = self.W_ih_weight_sigmas
            state["W_ih_bias_sigmas"] = self.W_ih_bias_sigmas
            if self.model_type == "trainable":
                state["u_sigmas"] = self.u_sigmas
                state["v_sigmas"] = self.v_sigmas

        # Save current hidden states if they exist
        if hasattr(self, "_current_hidden_states") and self._current_hidden_states is not None:
            state["hidden_states"] = self._current_hidden_states

        return state

    def load_state_dict(self, state: dict) -> None:
        """Restore network state from checkpoint.

        Args:
            state: State dictionary from get_state_dict()
        """
        self.dimensions = state["dimensions"]
        self.num_nets = state["num_nets"]
        self.model_type = state["model_type"]
        self.sigma_init = state["sigma_init"]
        self.sigma_noise = state["sigma_noise"]
        self.num_layers = len(self.dimensions) - 1
        self.W_ih_weights = state["W_ih_weights"]
        self.W_ih_biases = state["W_ih_biases"]

        if self.model_type == "reservoir":
            self.W_hh = state["W_hh"]
        elif self.model_type == "trainable":
            self.u = state["u"]
            self.v = state["v"]

        if "W_ih_weight_sigmas" in state:
            self.W_ih_weight_sigmas = state["W_ih_weight_sigmas"]
            self.W_ih_bias_sigmas = state["W_ih_bias_sigmas"]
            if self.model_type == "trainable":
                self.u_sigmas = state["u_sigmas"]
                self.v_sigmas = state["v_sigmas"]

        # Restore hidden states if present
        if "hidden_states" in state:
            self._current_hidden_states = state["hidden_states"]
        else:
            self._current_hidden_states = None

    def get_parameters_flat(self) -> Float[Tensor, "num_nets num_params"]:
        """Get flattened parameter vectors for all networks.

        Used by ES/CMA-ES optimizers for parameter averaging.

        Returns:
            Flat parameters [num_nets, num_params] where num_params is the
            total number of parameters (W_ih weights/biases + recurrent weights)
        """
        params_list: list[Float[Tensor, "num_nets ..."]] = []

        for layer_idx in range(self.num_layers):
            # Input-to-hidden weights and biases
            params_list.append(self.W_ih_weights[layer_idx].flatten(start_dim=1))
            params_list.append(self.W_ih_biases[layer_idx])

            # Recurrent weights (trainable mode only - reservoir is frozen)
            if self.model_type == "trainable":
                params_list.append(self.u[layer_idx])
                params_list.append(self.v[layer_idx])

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

        for layer_idx in range(self.num_layers):
            # Extract and reshape W_ih weights
            w_size: int = self.W_ih_weights[layer_idx][0].numel()
            w_shape: tuple[int, int, int] = self.W_ih_weights[layer_idx].shape
            self.W_ih_weights[layer_idx] = (
                flat_params[:, idx : idx + w_size].reshape(w_shape).clone()
            )
            idx += w_size

            # Extract W_ih biases
            b_size: int = self.W_ih_biases[layer_idx][0].numel()
            self.W_ih_biases[layer_idx] = flat_params[:, idx : idx + b_size].clone()
            idx += b_size

            # Extract recurrent weights (trainable mode only)
            if self.model_type == "trainable":
                u_size: int = self.u[layer_idx][0].numel()
                self.u[layer_idx] = flat_params[:, idx : idx + u_size].clone()
                idx += u_size

                v_size: int = self.v[layer_idx][0].numel()
                self.v[layer_idx] = flat_params[:, idx : idx + v_size].clone()
                idx += v_size

    def clone_network(self, indices: Tensor) -> None:
        """Clone networks at specified indices to fill population.

        Used by GA selection: top performers are cloned to replace losers.

        Args:
            indices: Indices of networks to clone [num_nets]
                     (indices may repeat for duplication)
        """
        for layer_idx in range(self.num_layers):
            self.W_ih_weights[layer_idx] = self.W_ih_weights[layer_idx][indices].clone()
            self.W_ih_biases[layer_idx] = self.W_ih_biases[layer_idx][indices].clone()

            if self.model_type == "reservoir":
                self.W_hh[layer_idx] = self.W_hh[layer_idx][indices].clone()
            elif self.model_type == "trainable":
                self.u[layer_idx] = self.u[layer_idx][indices].clone()
                self.v[layer_idx] = self.v[layer_idx][indices].clone()

            if self.sigma_noise is not None:
                self.W_ih_weight_sigmas[layer_idx] = (
                    self.W_ih_weight_sigmas[layer_idx][indices].clone()
                )
                self.W_ih_bias_sigmas[layer_idx] = (
                    self.W_ih_bias_sigmas[layer_idx][indices].clone()
                )
                if self.model_type == "trainable":
                    self.u_sigmas[layer_idx] = self.u_sigmas[layer_idx][indices].clone()
                    self.v_sigmas[layer_idx] = self.v_sigmas[layer_idx][indices].clone()

        # Clone hidden states if they exist
        if hasattr(self, "_current_hidden_states") and self._current_hidden_states is not None:
            self._current_hidden_states = [h[indices].clone() for h in self._current_hidden_states]
