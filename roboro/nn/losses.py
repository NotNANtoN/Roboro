"""TD loss functions for value-based learning.

Usage::

    from roboro.nn.losses import get_td_loss

    loss_fn = get_td_loss("huber")
    loss = loss_fn(predicted_q, target_q)
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

TDLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def get_td_loss(name: str) -> TDLossFn:
    """Resolve a TD loss function by name.

    Supported:
        - ``"huber"`` — Smooth L1 loss, robust to outlier TD-errors.
          The original DQN paper (Mnih et al., 2015) clips the TD error to
          [-1, 1] before squaring, which is equivalent to Huber loss.
        - ``"mse"`` — Mean squared error. Common in many implementations;
          simpler but more sensitive to large TD-errors.
    """
    losses: dict[str, TDLossFn] = {
        "mse": nn.functional.mse_loss,
        "huber": nn.functional.smooth_l1_loss,
    }
    key = name.lower()
    if key not in losses:
        raise ValueError(f"Unknown TD loss '{name}'. Choose from {list(losses)}")
    return losses[key]
