from dataclasses import dataclass
from functools import partial
from typing import Annotated as An

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from common.dl.datamodule.base import BaseDataModule, BaseDataModuleConfig
from common.utils.beartype import ge, lt, one_of


@dataclass
class MNISTDataModuleConfig(BaseDataModuleConfig):
    val_percentage: An[float, ge(0), lt(1)] = 0.005


class MNISTDataModule(BaseDataModule):

    def __init__(
        self: "MNISTDataModule",
        config: MNISTDataModuleConfig,
        dataloader: partial[DataLoader],
    ) -> None:
        super().__init__(config=config, dataloader=dataloader)
        self.train_val_split = (
            1 - config.val_percentage,
            config.val_percentage,
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # Pre-computed mean and std for the MNIST dataset.
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            ],
        )

    def prepare_data(self: "MNISTDataModule") -> None:
        MNIST(root=self.config.data_dir, download=True)

    def setup(
        self: "MNISTDataModule",
        stage: An[str, one_of("fit", "validate", "test")],
    ) -> None:
        mnist_full = MNIST(
            root=self.config.data_dir,
            train=True,
            transform=self.transform,
        )
        self.datasets.train, self.datasets.val = random_split(
            dataset=mnist_full,
            lengths=self.train_val_split,
        )
