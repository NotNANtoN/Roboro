"""Base actor interface."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseActor(nn.Module, ABC):
    """Selects actions given encoded observations.

    The actor is the *policy head* — it may be a learned network (SAC) or a
    simple wrapper around a critic (ε-greedy DQN).

    ``act()`` is the main entry point used during environment interaction.
    ``forward()`` is used during training to get differentiable outputs
    (e.g. log-probs for policy-gradient methods).
    """

    @abstractmethod
    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Select actions for environment interaction (no graph needed).

        Args:
            obs: ``(B, feature_dim)`` encoded observations.
            deterministic: if True, suppress exploration noise / sampling.

        Returns:
            ``(B, *action_shape)`` actions ready to send to the environment.
        """

    @abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute actions **and** log-probs (for training).

        Args:
            obs: ``(B, feature_dim)``

        Returns:
            actions: ``(B, *action_shape)``
            log_probs: ``(B,)`` or ``None`` if not applicable.
        """
