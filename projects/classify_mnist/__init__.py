"""MNIST classification."""

from beartype import BeartypeConf
from beartype.claw import beartype_this_package

from .datamodule import MNISTDataModule, MNISTDataModuleConfig
from .litmodule import MNISTClassificationLitModule

beartype_this_package(conf=BeartypeConf(is_pep484_tower=True))

__all__ = [
    "MNISTDataModuleConfig",
    "MNISTDataModule",
    "MNISTClassificationLitModule",
]
