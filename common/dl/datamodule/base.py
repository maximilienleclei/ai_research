from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import Annotated as An
from typing import final

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from common.utils.beartype import ge, not_empty, one_of


@dataclass
class Datasets:
    train: (
        Dataset[Tensor | dict[str, Tensor]] | HFDataset | HFDatasetDict | None
    ) = None
    val: (
        Dataset[Tensor | dict[str, Tensor]] | HFDataset | HFDatasetDict | None
    ) = None


@dataclass
class BaseDataModuleConfig:
    data_dir: An[str, not_empty()] = "${config.data_dir}"
    device: An[str, one_of("cpu", "gpu")] = "${config.device}"
    max_per_device_batch_size: An[int, ge(1)] | None = None
    max_per_device_num_workers: An[int, ge(0)] | None = None
    fixed_per_device_batch_size: An[int, ge(1)] | None = None
    fixed_per_device_num_workers: An[int, ge(0)] | None = None
    shuffle_train_dataset: bool = True
    shuffle_val_dataset: bool = True


class BaseDataModule(LightningDataModule, ABC):
    def __init__(
        self: "BaseDataModule",
        config: BaseDataModuleConfig,
        dataloader: partial[DataLoader],
    ) -> None:
        super().__init__()
        self.config = config
        self.dataloader_partial = dataloader
        self.datasets = Datasets()
        # Both will be overriden if `None`
        self.per_device_batch_size = config.fixed_per_device_batch_size
        self.per_device_num_workers = config.fixed_per_device_num_workers

    @final
    def load_state_dict(
        self: "BaseDataModule",
        state_dict: dict[str, int],
    ) -> None:
        self.per_device_batch_size = state_dict["per_device_batch_size"]
        self.per_device_num_workers = state_dict["per_device_num_workers"]

    @final
    def state_dict(self: "BaseDataModule") -> dict[str, int]:
        return {
            "per_device_batch_size": self.per_device_batch_size,
            "per_device_num_workers": self.per_device_num_workers,
        }

    @final
    def x_dataloader(
        self: "BaseDataModule",
        dataset: Dataset[Tensor] | HFDataset | None,
        *,
        shuffle: bool = True,
    ) -> DataLoader[Tensor]:
        if dataset is None:
            raise AttributeError
        return DataLoader(
            dataset=dataset,
            batch_size=self.per_device_batch_size,
            shuffle=shuffle,
            num_workers=self.per_device_num_workers,
        )

    @final
    def train_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        return self.x_dataloader(
            dataset=self.datasets.train,
            shuffle=self.config.shuffle_train_dataset,
        )

    def val_dataloader(self: "BaseDataModule") -> DataLoader[Tensor]:
        return self.x_dataloader(
            dataset=self.datasets.val,
            shuffle=self.config.shuffle_val_dataset,
        )
