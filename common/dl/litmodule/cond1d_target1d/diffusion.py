"""Shapes.

BS: Batch size
TSL: Cond1DTarget1D sequence length
HNC: Cond1DTarget1D number of channels
CSL: Conditioning sequence length
CNC: Conditioning number of channels (i.e. number of features)
"""

import contextlib
import logging
from dataclasses import dataclass, field
from typing import Annotated as An
from typing import Any

import torch
from jaxtyping import Float, Int
from torch import Tensor

from common.dl.litmodule.nnmodule.cond_diffusion.dit_1d_1d import DiT1D1D
from common.dl.litmodule.cond1d_target1d.base import (
    BaseCond1DTarget1DPredLitModule,
    BaseCond1DTarget1DPredLitModuleConfig,
)
from common.dl.litmodule.utils import to_wandb_image
from common.dl.utils.diffusion import create_diffusion
from common.utils.beartype import one_of

log = logging.getLogger(__name__)


@dataclass
class Cond1DTarget1DDiffusionLitModuleConfig(
    BaseCond1DTarget1DPredLitModuleConfig
):

    cfg_scales: list[float] = field(
        default_factory=lambda: [1.0, 1.5, 2.0, 4.0],
    )

    def __post_init__(self: "Cond1DTarget1DDiffusionLitModuleConfig") -> None:
        with contextlib.suppress(ValueError):
            self.wandb_column_names.remove("x_hat")
        self.wandb_column_names.extend(
            [f"x_hat_{cfg_scale}" for cfg_scale in self.cfg_scales],
        )


class Cond1DTarget1DDiffusionLitModule(BaseCond1DTarget1DPredLitModule):

    def __init__(
        self: "Cond1DTarget1DDiffusionLitModule",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config: Cond1DTarget1DDiffusionLitModuleConfig
        self.nnmodule: DiT1D1D
        self.diffusion = create_diffusion(timestep_respacing="")

    def step(
        self: "Cond1DTarget1DDiffusionLitModule",
        data: dict[str, Tensor],
        stage: An[str, one_of("train", "val", "test")],
    ) -> Float[Tensor, " "]:
        x: Float[Tensor, " BS 4 TSL"] = data["Cond1DTarget1Ds"]
        y: Float[Tensor, " BS CNC CSL"] = data["audio_stfts"]
        self.save_conditioning_target_features(stage, y, x)
        t: Int[Tensor, " BS"] = torch.randint(
            low=0,
            high=self.diffusion.num_timesteps,
            size=(x.shape[0],),
            device=self.device,
        )
        model = (
            self.nnmodule if stage == "train" else self.ema_nnmodule.ema_model
        )
        loss_dict = self.diffusion.training_losses(
            model,
            x,
            t,
            {"y": y},
        )
        loss: Float[Tensor, ""] = loss_dict["loss"].mean()
        return loss

    def update_wandb_data_before_log(
        self: "Cond1DTarget1DDiffusionLitModule",
        data: list[dict[str, Any]],
        stage: An[str, one_of("train", "val")],
    ) -> None:
        if self.global_rank == 0:
            num_cfg_scales = len(self.config.cfg_scales)
            cfg_scales = (
                torch.tensor(self.config.cfg_scales)
                .repeat(
                    self.config.wandb_num_samples,
                )
                .to(self.device)
            )
            x_big_t = torch.randn(
                self.config.wandb_num_samples * num_cfg_scales,
                4,
                self.nnmodule.input_size,
                device=self.device,
            ).to(self.device)
            y = torch.zeros(
                (
                    self.config.wandb_num_samples * num_cfg_scales,
                    *data[0]["y_raw"].shape,
                ),
            )
            for i in range(self.config.wandb_num_samples):
                y[i * num_cfg_scales : (i + 1) * num_cfg_scales] = data[i][
                    "y_raw"
                ]
            y = y.to(self.device)
            x_hat = (
                self.diffusion.p_sample_loop(
                    self.ema_nnmodule.ema_model.forward_with_cfg,
                    x_big_t.shape,
                    x_big_t,
                    clip_denoised=False,
                    model_kwargs={"y": y, "cfg_scales": cfg_scales},
                    progress=True,
                    device=self.device,
                )
                .squeeze()
                .cpu()
            )
            for i, data_i in enumerate(data):
                for j, cfg_scale_j in enumerate(self.config.cfg_scales):
                    data_i.update(
                        {
                            f"x_hat_{cfg_scale_j}": to_wandb_image(
                                x_hat[i * num_cfg_scales + j],
                            ),
                        },
                    )
                data_i.pop("x_raw")
                data_i.pop("y_raw")
