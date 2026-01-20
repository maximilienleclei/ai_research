"""Type aliases for the deep learning framework.

This module provides type aliases to improve code readability and
enable better static type checking.
"""

from functools import partial
from typing import TypeAlias

from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# A dataset can be a PyTorch Dataset, HuggingFace Dataset/DatasetDict, or None
DatasetLike: TypeAlias = (
    Dataset[Tensor | dict[str, Tensor]] | HFDataset | HFDatasetDict | None
)

# A partial DataLoader factory
DataLoaderPartial: TypeAlias = partial[DataLoader]
