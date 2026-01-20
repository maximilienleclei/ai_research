"""Miscellaneous utility functions."""

import random
from collections.abc import Callable
from typing import Any

import numpy as np
import requests
import torch


def get_path(clb: Callable[..., Any]) -> str:
    """Get the fully qualified path of a callable (module.name)."""
    return f"{clb.__module__}.{clb.__name__}"


def seed_all(seed: int | np.uint32) -> None:
    """Seed all random number generators for reproducibility.

    Seeds: random, numpy, torch CPU, and torch CUDA.
    """
    random.seed(a=int(seed))
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=int(seed))
    torch.cuda.manual_seed_all(seed=int(seed))


def can_connect_to_internet() -> bool:
    """Check if internet connection is available."""
    try:
        response = requests.get(url="https://www.google.com", timeout=5)
        response.raise_for_status()
    except Exception:
        return False
    return True
