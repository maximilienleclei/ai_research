"""Shapes:

- BS: Batch size
- TSL: Target sequence length
- CNC: Number of channels for the conditioning data
- CSL: Conditioning data sequence length
"""

import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import Annotated as An
from typing import Any, final

from ema_pytorch import EMA
from jaxtyping import Float
from torch import Tensor

from common.utils.beartype import one_of

from ...litmodule.base import BaseLitModule, BaseLitModuleConfig
from .utils import to_wandb_image

log = logging.getLogger(__name__)


@dataclass
class BaseCond1DTarget1DPredLitModuleConfig(BaseLitModuleConfig):

    wandb_column_names: list[str] = field(
        default_factory=lambda: ["x", "y", "x_hat"],
    )


class BaseCond1DTarget1DPredLitModule(BaseLitModule, ABC):

    def __init__(
        self: "BaseCond1DTarget1DPredLitModule",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config: BaseCond1DTarget1DPredLitModuleConfig
        self.ema_nnmodule = EMA(
            model=self.nnmodule,
            include_online_model=False,
        )
        self.ema_nnmodule.ema_model.eval()

    @final
    def save_conditioning_target_features(
        self: "BaseCond1DTarget1DPredLitModule",
        stage: An[str, one_of("train", "val", "test")],
        y: Float[Tensor, " BS CNC CSL"],
        x: Float[Tensor, " BS HNC TSL"],
    ) -> None:
        data = (
            self.wandb_train_data if stage == "train" else self.wandb_val_data
        )
        if len(data) >= self.config.wandb_num_samples or self.global_rank != 0:
            return
        x, y = x.cpu(), y.cpu()
        for index in range(len(x)):
            data.append(
                {
                    "x": to_wandb_image(x[index]),
                    "x_raw": x[index],
                    "y": to_wandb_image(y[index]),
                    "y_raw": y[index],
                },
            )
            if len(data) >= self.config.wandb_num_samples:
                break

    def optimizer_step(
        self: "BaseCond1DTarget1DPredLitModule",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().optimizer_step(*args, **kwargs)
        self.ema_nnmodule.update()
