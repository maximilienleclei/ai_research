"""Shapes.

BS: Batch size
TSL: Target sequence length
HNC: Target signal number of channels (i.e. number of features)
CNC: Conditioning number of channels (i.e. number of features)
"""

import logging
from dataclasses import dataclass
from typing import Annotated as An
from typing import Any

import torch
import torch.nn.functional as f
import torchmetrics.functional as tmf
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from common.dl.litmodule._nnmodule.cond_autoreg.base import BaseCAM
from common.dl.litmodule.cond1d_target1d.base import (
    BaseCond1DTarget1DPredLitModule,
    BaseCond1DTarget1DPredLitModuleConfig,
)
from common.dl.litmodule.utils import to_wandb_image
from common.utils.beartype import one_of

log = logging.getLogger(__name__)


@dataclass
class Cond1DTarget1DAutoregressionLitModuleConfig(
    BaseCond1DTarget1DPredLitModuleConfig
):

    accuracy_binarization_threshold: float = 1e-2
    Cond1DTarget1D_num_channels: int = 4
    binarize_Cond1DTarget1D_data: bool = False
    temp_gt0_Cond1DTarget1D_data: bool = True

    def __post_init__(
        self: "Cond1DTarget1DAutoregressionLitModuleConfig",
    ) -> None:
        self.wandb_column_names.append("x_hat")


class Cond1DTarget1DAutoregressionLitModule(BaseCond1DTarget1DPredLitModule):

    def __init__(
        self: "Cond1DTarget1DAutoregressionLitModule",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config: Cond1DTarget1DAutoregressionLitModuleConfig
        self.nnmodule: BaseCAM

    def step(
        self: "Cond1DTarget1DAutoregressionLitModule",
        data: dict[str, Tensor],
        stage: An[str, one_of("train", "val", "test")],
    ) -> Float[Tensor, " "]:
        x: Float[Tensor, " BS TSL HNC"] = data["Cond1DTarget1Ds"]
        x = x[..., : self.config.Cond1DTarget1D_num_channels]
        if self.config.temp_gt0_Cond1DTarget1D_data:
            x = torch.clamp(x, min=0.0)
        if self.config.binarize_Cond1DTarget1D_data:
            x = (x.abs() > self.config.accuracy_binarization_threshold).float()
        y: Float[Tensor, " BS TSL CNC"] = data["audio_stfts"]
        self.save_conditioning_target_features(
            stage,
            rearrange(y, "BS TSL CNC -> BS CNC TSL"),
            rearrange(x, "BS TSL HNC -> BS HNC TSL"),
        )
        model = (
            self.nnmodule if stage == "train" else self.ema_nnmodule.ema_model
        )
        x_hat, loss = model.predict_sequence_features_and_compute_loss(
            y,
            x,
            "fitting",
        )
        self.compute_and_log_metrics(x, x_hat, loss, stage)
        return loss

    def update_wandb_data_before_log(
        self: "Cond1DTarget1DAutoregressionLitModule",
        data: list[dict[str, Any]],
        stage: An[str, one_of("train", "val")],
    ) -> None:
        if self.global_rank == 0:
            y = torch.zeros(
                (
                    len(data),
                    *rearrange(data[0]["y_raw"], "CNC TSL -> TSL CNC").shape,
                ),
            )
            x = torch.zeros(
                (
                    len(data),
                    *rearrange(data[0]["x_raw"], "HNC TSL -> TSL HNC").shape,
                ),
            )
            for i in range(len(data)):
                x[i] = rearrange(data[i]["x_raw"], "HNC TSL -> TSL HNC")
                y[i] = rearrange(data[i]["y_raw"], "CNC TSL -> TSL CNC")
            x = x.to(self.device)
            y = y.to(self.device)
            assert isinstance(
                self.ema_nnmodule.ema_model,
                BaseCAM,
            )  # Static type checking purposes
            x_hat, loss = (
                self.ema_nnmodule.ema_model.predict_sequence_features_and_compute_loss(
                    y,
                    x,
                    "inference",
                )
            )
            self.compute_and_log_metrics(x, x_hat, loss, stage + "_infer")
            x_hat = x_hat.cpu()
            for i, data_i in enumerate(data):
                data_i.update(
                    {
                        "x_hat": to_wandb_image(
                            rearrange(x_hat[i], "TSL HNC -> HNC TSL"),
                        ),
                    },
                )
                data_i.pop("x_raw")
                data_i.pop("y_raw")

    def compute_and_log_metrics(
        self: "Cond1DTarget1DAutoregressionLitModule",
        x: Float[Tensor, " BS TSL HNC"],
        x_hat: Float[Tensor, " BS TSL HNC"],
        loss: Float[Tensor, " "],
        stage: An[
            str,
            one_of(
                "train",
                "val",
                "test",
                "train_infer",
                "val_infer",
                "test_infer",
            ),
        ],
    ) -> None:
        self.log(name=f"{stage}/loss", value=loss)
        mse: Float[Tensor, " "] = f.mse_loss(x_hat, x)
        self.log(name=f"{stage}/mse", value=mse)
        binary_x: Float[Tensor, " BS TSL HNC"] = (
            x.abs() > self.config.accuracy_binarization_threshold
        ).float()
        binary_x_hat: Float[Tensor, " BS TSL HNC"] = (
            x_hat.abs() > self.config.accuracy_binarization_threshold
        ).float()
        acc: Float[Tensor, " "] = tmf.accuracy(
            preds=binary_x_hat,
            target=binary_x,
            task="binary",
        )
        self.log(name=f"{stage}/acc", value=acc)
        f1: Float[Tensor, " "] = tmf.f1_score(
            preds=binary_x_hat,
            target=binary_x,
            task="binary",
        )
        self.log(name=f"{stage}/f1", value=f1)
