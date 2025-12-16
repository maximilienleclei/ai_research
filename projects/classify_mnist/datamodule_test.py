from pathlib import Path

import pytest
from torch.utils.data import Subset
from torchvision.datasets import MNIST

from projects.classify_mnist.datamodule import (
    MNISTDataModule,
    MNISTDataModuleConfig,
)


@pytest.fixture
def datamodule(tmp_path: Path) -> MNISTDataModule:
    return MNISTDataModule(
        MNISTDataModuleConfig(
            data_dir=str(tmp_path) + "/",
            device="cpu",
            val_percentage=0.1,
        ),
    )


def test_setup_fit(datamodule: MNISTDataModule) -> None:
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    assert isinstance(datamodule.datasets.train, Subset)
    assert isinstance(datamodule.datasets.val, Subset)

    assert len(datamodule.datasets.train) == 54000
    assert len(datamodule.datasets.val) == 6000


def test_setup_test(datamodule: MNISTDataModule) -> None:
    datamodule.prepare_data()
    datamodule.setup(stage="test")

    assert isinstance(datamodule.datasets.test, MNIST)
    assert len(datamodule.datasets.test) == 10000
