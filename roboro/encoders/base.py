"""Base encoder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseEncoder(nn.Module, ABC):
    """Maps raw observations to a fixed-size latent feature vector.

    Every encoder exposes *feature_dim* so downstream modules (critics,
    actors) can wire themselves without magic numbers.
    """

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensionality of the output feature vector."""

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of observations.

        Args:
            obs: ``(B, *obs_shape)``

        Returns:
            features: ``(B, feature_dim)``
        """
