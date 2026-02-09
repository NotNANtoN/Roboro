"""Reproducibility utilities — deterministic seeding for all RNG sources."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seeds for all RNG sources used during training.

    Covers:
      * Python ``random`` module (used by some Gymnasium envs).
      * NumPy global RNG (legacy code, wrappers).
      * PyTorch CPU and CUDA RNGs.

    Note: for full CUDA determinism, callers may also want::

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    but this can significantly slow down training and is left to the user.

    Args:
        seed: Integer seed.  Use 0 for a sensible default.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # covers CPU + CUDA (calls cuda.manual_seed_all)
