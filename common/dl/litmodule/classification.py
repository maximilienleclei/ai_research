from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated as An
from typing import Any, final

import torch
import torch.nn.functional as f
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics.classification import MulticlassAccuracy

from common.dl.litmodule.base import BaseLitModule, BaseLitModuleConfig
from common.utils.beartype import ge, one_of


@dataclass
class BaseClassificationLitModuleConfig(BaseLitModuleConfig):
    """Configuration for classification Lightning modules.

    Attributes:
        num_classes: Number of classes for classification. Must be at least 2.
        wandb_column_names: Column names for W&B table logging. Defaults to
            input, true label, predicted label, and logits columns.
    """

    num_classes: An[int, ge(2)] = 2
    wandb_column_names: list[str] = field(
        default_factory=lambda: ["x", "y", "y_hat", "logits"],
    )


class BaseClassificationLitModule(BaseLitModule, ABC):
    """Base Lightning module for classification tasks.

    Extends BaseLitModule with classification-specific functionality:
    - Multiclass accuracy tracking
    - Cross-entropy loss computation
    - W&B logging of predictions vs ground truth
    """

    def __init__(
        self: "BaseClassificationLitModule",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config: BaseClassificationLitModuleConfig
        self.accuracy = MulticlassAccuracy(
            num_classes=self.config.num_classes,
        )
        self.to_wandb_media: Callable[..., Any] = lambda x: x

    @property
    @abstractmethod
    def wandb_media_x(self):
        """Convert input tensor to W&B media format for visualization.

        Subclasses should implement this to return a callable that transforms
        input samples (e.g., images) into W&B-compatible media objects like
        wandb.Image. This is used when logging sample predictions to tables.
        """
        ...

    @final
    def step(
        self: "BaseClassificationLitModule",
        data: tuple[
            Float[Tensor, "batch_size *x_dim"],
            Int[Tensor, "batch_size"],
        ],
        stage: An[str, one_of("train", "val", "test")],
    ) -> Float[Tensor, ""]:
        x: Float[Tensor, "batch_size *x_dim"] = data[0]
        y: Int[Tensor, "batch_size"] = data[1]
        logits: Float[Tensor, "batch_size num_classes"] = self.nnmodule(x)
        y_hat: Int[Tensor, "batch_size"] = torch.argmax(input=logits, dim=1)
        accuracy: Float[Tensor, ""] = self.accuracy(preds=y_hat, target=y)
        self.log(name=f"{stage}/acc", value=accuracy)
        self.save_wandb_data(stage, x, y, y_hat, logits)
        return f.cross_entropy(input=logits, target=y)

    @final
    def save_wandb_data(
        self: "BaseClassificationLitModule",
        stage: An[str, one_of("train", "val", "test")],
        x: Float[Tensor, "batch_size *x_dim"],
        y: Int[Tensor, "batch_size"],
        y_hat: Int[Tensor, "batch_size"],
        logits: Float[Tensor, "batch_size num_classes"],
    ) -> None:
        data = (
            self.wandb_train_data if stage == "train" else self.wandb_val_data
        )
        if data or self.global_rank != 0:
            return
        x, y, y_hat, logits = x.cpu(), y.cpu(), y_hat.cpu(), logits.cpu()
        for i in range(self.config.wandb_num_samples):
            index = (
                self.curr_val_epoch * self.config.wandb_num_samples + i
            ) % len(x)
            data.append(
                {
                    "x": self.wandb_media_x(x[index]),
                    "y": y[index],
                    "y_hat": y_hat[index],
                    "logits": logits[index].tolist(),
                },
            )
