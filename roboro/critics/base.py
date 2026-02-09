"""Base critic interfaces."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseQCritic(nn.Module, ABC):
    """Estimates Q(s, a) — the expected return for taking *a* in state *s*.

    Subclasses decide how actions are represented:
    - *Discrete*: ``Q(s)`` returns values for **all** actions → ``(B, n_actions)``
    - *Continuous*: ``Q(s, a)`` takes an explicit action tensor → ``(B, 1)``
    """

    @abstractmethod
    def forward(self, obs: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        """Compute Q-values.

        Args:
            obs: ``(B, feature_dim)`` — encoder output.
            actions: ``(B, action_dim)`` for continuous critics, ``None`` for discrete.

        Returns:
            Discrete: ``(B, n_actions)`` Q-values for every action.
            Continuous: ``(B, 1)`` Q-value for the given action.
        """


class BaseVCritic(nn.Module, ABC):
    """Estimates V(s) — the expected return from state *s*."""

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute state value.

        Args:
            obs: ``(B, feature_dim)``

        Returns:
            ``(B, 1)`` state values.
        """
