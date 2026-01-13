"""Lightning module for autoencoder training."""

from dataclasses import dataclass, field
from typing import Annotated as An
from typing import Any

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from common.dl.litmodule.base import BaseLitModule, BaseLitModuleConfig
from common.utils.beartype import one_of


@dataclass
class AutoencoderLitModuleConfig(BaseLitModuleConfig):
    wandb_column_names: list[str] = field(
        default_factory=lambda: ["reconstruction_loss"]
    )


class AutoencoderLitModule(BaseLitModule):
    """Lightning module for autoencoder training with MSE reconstruction loss."""

    def __init__(
        self: "AutoencoderLitModule",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config: AutoencoderLitModuleConfig

    def step(
        self: "AutoencoderLitModule",
        data: tuple[Float[Tensor, "BS SD"]],
        stage: An[str, one_of("train", "val", "test", "predict")],
    ) -> Float[Tensor, ""]:
        (x,) = data  # Unpack single element tuple
        x_reconstructed = self.nnmodule(x)
        loss = F.mse_loss(x_reconstructed, x)

        # Log to wandb
        wandb_data = self.wandb_train_data if stage == "train" else self.wandb_val_data
        if not wandb_data and self.global_rank == 0:
            for i in range(min(self.config.wandb_num_samples, len(x))):
                sample_loss = F.mse_loss(x_reconstructed[i], x[i]).item()
                wandb_data.append({"reconstruction_loss": sample_loss})

        return loss
