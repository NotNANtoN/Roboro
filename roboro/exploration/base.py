"""Base exploration module interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from roboro.core.types import Batch


class BaseExploration(nn.Module, ABC):
    """Computes an intrinsic reward bonus for a batch of transitions."""

    @abstractmethod
    def intrinsic_reward(self, batch: Batch) -> torch.Tensor:
        """Compute intrinsic reward for the batch.

        Args:
            batch: a ``Batch`` of transitions.

        Returns:
            ``(B,)`` intrinsic reward values to add to extrinsic reward.
        """
