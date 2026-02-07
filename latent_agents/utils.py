"""Minimal utilities for the latent_agents library."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(preference: Optional[str] = None) -> torch.device:
    """Return a ``torch.device``, defaulting to CUDA if available."""
    if preference is not None:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
