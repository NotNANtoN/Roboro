"""Base dynamics model interface."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseDynamics(nn.Module, ABC):
    """Predicts next latent state given current latent state and action.

    Optionally predicts reward and/or termination.
    """

    @abstractmethod
    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the next latent state.

        Args:
            latent: ``(B, latent_dim)``
            action: ``(B, action_dim)``

        Returns:
            next_latent: ``(B, latent_dim)``
        """

    def predict_reward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor | None:
        """Optional: predict reward from latent + action.  Default: ``None``."""
        return None
