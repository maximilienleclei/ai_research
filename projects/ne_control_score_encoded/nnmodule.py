"""Autoencoder neural network module for state encoding."""

from dataclasses import dataclass
from typing import Annotated as An

from jaxtyping import Float
from torch import Tensor, nn

from common.utils.beartype import ge


@dataclass
class AutoencoderConfig:
    state_dim: An[int, ge(1)] = 4  # Default for CartPole
    latent_dim: An[int, ge(1)] = 3  # state_dim - 1
    hidden_size: An[int, ge(1)] = 32
    num_hidden_layers: An[int, ge(0)] = 1


class Autoencoder(nn.Module):
    """Simple feedforward autoencoder for state compression."""

    def __init__(self: "Autoencoder", config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Build encoder layers
        encoder_layers: list[nn.Module] = []
        in_dim = config.state_dim
        for _ in range(config.num_hidden_layers):
            encoder_layers.append(nn.Linear(in_dim, config.hidden_size))
            encoder_layers.append(nn.ReLU())
            in_dim = config.hidden_size
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers
        decoder_layers: list[nn.Module] = []
        in_dim = config.latent_dim
        for _ in range(config.num_hidden_layers):
            decoder_layers.append(nn.Linear(in_dim, config.hidden_size))
            decoder_layers.append(nn.ReLU())
            in_dim = config.hidden_size
        decoder_layers.append(nn.Linear(in_dim, config.state_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(
        self: "Autoencoder",
        x: Float[Tensor, "BS SD"],
    ) -> Float[Tensor, "BS SD"]:
        """Forward pass: encode then decode."""
        latent = self.encode(x)
        return self.decode(latent)

    def encode(
        self: "Autoencoder",
        x: Float[Tensor, "BS SD"],
    ) -> Float[Tensor, "BS LD"]:
        """Encode states to latent representation."""
        return self.encoder(x)

    def decode(
        self: "Autoencoder",
        z: Float[Tensor, "BS LD"],
    ) -> Float[Tensor, "BS SD"]:
        """Decode latent representation to states."""
        return self.decoder(z)
