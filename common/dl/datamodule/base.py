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

from common.dl.types import DatasetLike
from common.utils.beartype import ge, not_empty, one_of


@dataclass
class Datasets:
    """Container for train and validation datasets.

    Attributes:
        train: Training dataset. Can be a PyTorch Dataset, HuggingFace Dataset,
            or None if not used.
        val: Validation dataset. Can be a PyTorch Dataset, HuggingFace Dataset,
            or None if not used.
    """

    train: DatasetLike = None
    val: DatasetLike = None


@dataclass
class BaseDataModuleConfig:
    """Configuration for the base data module.

    Attributes:
        data_dir: Directory containing the dataset files.
        device: Computing device ("cpu" or "gpu").
        max_per_device_batch_size: Upper bound for automatic batch size search.
            If None, the search uses dataset-based heuristics.
        max_per_device_num_workers: Upper bound for automatic worker count search.
            If None, uses system CPU count.
        fixed_per_device_batch_size: Fixed batch size per device, bypassing
            automatic search. If None, automatic search is performed.
        fixed_per_device_num_workers: Fixed number of dataloader workers per device,
            bypassing automatic search. If None, automatic search is performed.
        shuffle_train_dataset: Whether to shuffle the training dataset each epoch.
        shuffle_val_dataset: Whether to shuffle the validation dataset.
    """

    data_dir: An[str, not_empty()] = "${config.data_dir}"
    device: An[str, one_of("cpu", "gpu")] = "${config.device}"
    max_per_device_batch_size: An[int, ge(1)] | None = None
    max_per_device_num_workers: An[int, ge(0)] | None = None
    fixed_per_device_batch_size: An[int, ge(1)] | None = None
    fixed_per_device_num_workers: An[int, ge(0)] | None = None
    shuffle_train_dataset: bool = True
    shuffle_val_dataset: bool = True


class BaseDataModule(LightningDataModule, ABC):
    """Base Lightning data module for dataset handling.

    Provides infrastructure for loading and serving datasets during training.
    Subclasses should implement `prepare_data()` and `setup()` to define
    how data is downloaded/processed and split into train/val sets.

    The module supports automatic batch size and worker count tuning when
    `fixed_per_device_batch_size` or `fixed_per_device_num_workers` are not
    set in the config.
    """

    def __init__(
        self: "BaseDataModule",
        config: BaseDataModuleConfig,
        dataloader: partial[DataLoader],
    ) -> None:
        """Initialize the data module.

        Args:
            config: Data module configuration.
            dataloader: Partial function for DataLoader instantiation.
                Typically pre-configured with pin_memory, persistent_workers, etc.
        """
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
        """Create a DataLoader for the given dataset.

        Args:
            dataset: The dataset to create a DataLoader for.
            shuffle: Whether to shuffle the data each epoch.

        Returns:
            Configured DataLoader instance.

        Raises:
            AttributeError: If dataset is None, indicating the dataset
                was not properly initialized in setup().
        """
        if dataset is None:
            raise AttributeError("Dataset is None. Ensure setup() initializes the dataset.")
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
