"""Concrete Q-critic implementations: discrete and continuous."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from roboro.critics.base import BaseQCritic
from roboro.nn.blocks import MLPBlock


class DiscreteQCritic(BaseQCritic):
    """Q(s) → R^{n_actions}  for discrete action spaces.

    The network takes only *obs* and outputs a Q-value for every action.

    Args:
        feature_dim: size of the input feature vector (encoder output).
        n_actions: number of discrete actions.
        trunk: optional pre-built ``nn.Module``.  When ``None`` (default),
            an ``MLPBlock(feature_dim → n_actions)`` is created from the
            remaining keyword arguments.
        **kwargs: forwarded to ``MLPBlock`` (``hidden_dim``, ``n_layers``,
            ``activation``, ``use_layer_norm``).
    """

    def __init__(
        self,
        feature_dim: int,
        n_actions: int,
        trunk: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        if trunk is not None:
            self.trunk = trunk
        else:
            self.trunk = MLPBlock(in_dim=feature_dim, out_dim=n_actions, **kwargs)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        q_values = cast(torch.Tensor, self.trunk(obs))  # (B, n_actions)
        if actions is not None:
            # Gather Q-values for the specified actions
            return cast(
                torch.Tensor, q_values.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
            )  # (B,)
        return q_values


class ContinuousQCritic(BaseQCritic):
    """Q(s, a) → R  for continuous action spaces.

    The network takes concatenated *[obs, action]* and outputs a scalar
    Q-value.  This is the standard formulation used by DDPG, TD3, SAC.

    Args:
        feature_dim: size of the observation feature vector.
        action_dim: size of the action vector.
        trunk: optional pre-built ``nn.Module``.  When ``None``,
            an ``MLPBlock(feature_dim + action_dim → 1)`` is created.
        **kwargs: forwarded to ``MLPBlock``.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        trunk: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if trunk is not None:
            self.trunk = trunk
        else:
            self.trunk = MLPBlock(in_dim=feature_dim + action_dim, out_dim=1, **kwargs)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        if actions is None:
            raise ValueError("ContinuousQCritic requires explicit actions.")
        x = torch.cat([obs, actions], dim=-1)
        return cast(torch.Tensor, self.trunk(x).squeeze(-1))  # (B,)


class TwinQCritic(nn.Module):
    """Clipped double-Q: maintains two Q-networks, returns the minimum.

    Used by SAC, TD3 to combat overestimation bias.
    Works with both discrete and continuous Q-critics.
    """

    def __init__(self, q1: BaseQCritic, q2: BaseQCritic) -> None:
        super().__init__()
        self.q1 = q1
        self.q2 = q2

    def forward(self, obs: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        """Return the element-wise minimum of the two Q-networks."""
        q1_val = cast(torch.Tensor, self.q1(obs, actions))
        q2_val = cast(torch.Tensor, self.q2(obs, actions))
        return torch.min(q1_val, q2_val)

    def both(
        self, obs: torch.Tensor, actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both Q-values (needed for the loss computation)."""
        return self.q1(obs, actions), self.q2(obs, actions)
