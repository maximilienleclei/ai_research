"""DataModule for loading collected states for autoencoder training."""

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated as An

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split

from common.dl.datamodule.base import BaseDataModule, BaseDataModuleConfig
from common.utils.beartype import ge, lt, not_empty


@dataclass
class StateDataModuleConfig(BaseDataModuleConfig):
    states_file: An[str, not_empty()] = "???"  # Path to .npy file with collected states
    val_percentage: An[float, ge(0), lt(1)] = 0.1
    normalize: bool = True


class StateDataModule(BaseDataModule):
    """DataModule for loading pre-collected states."""

    def __init__(
        self: "StateDataModule",
        config: StateDataModuleConfig,
        dataloader: partial[DataLoader],
    ) -> None:
        super().__init__(config=config, dataloader=dataloader)
        self.config: StateDataModuleConfig = config
        self.train_val_split = (
            1 - config.val_percentage,
            config.val_percentage,
        )
        self.state_mean: Tensor | None = None
        self.state_std: Tensor | None = None

    def setup(
        self: "StateDataModule",
        stage: str,
    ) -> None:
        # Load states from file
        states_path = Path(self.config.states_file)
        if not states_path.is_absolute():
            states_path = Path(self.config.data_dir) / states_path

        states = np.load(states_path)
        states_tensor = torch.from_numpy(states).float()

        # Normalize if requested
        if self.config.normalize:
            self.state_mean = states_tensor.mean(dim=0)
            self.state_std = states_tensor.std(dim=0) + 1e-8
            states_tensor = (states_tensor - self.state_mean) / self.state_std

            # Save normalization stats for later use by evaluator
            self._save_normalization_stats()

        # Create dataset (input = output for autoencoder)
        dataset = TensorDataset(states_tensor)

        # Split into train/val
        self.datasets.train, self.datasets.val = random_split(
            dataset=dataset,
            lengths=self.train_val_split,
        )

    def _save_normalization_stats(self: "StateDataModule") -> None:
        """Save normalization statistics to a JSON file."""
        if self.state_mean is None:
            return

        # Get the results directory from the states file path
        states_path = Path(self.config.states_file)
        stats_dir = states_path.parent
        stats_path = stats_dir / "normalization_stats.json"

        stats = {
            "mean": self.state_mean.tolist(),
            "std": self.state_std.tolist(),
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
