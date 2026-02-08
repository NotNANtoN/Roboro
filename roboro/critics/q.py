"""Concrete Q-critic implementations: discrete and continuous."""

from __future__ import annotations

import torch
from torch import nn

from roboro.critics.base import BaseQCritic


class DiscreteQCritic(BaseQCritic):
    """Q(s) → R^{n_actions}  for discrete action spaces.

    The network takes only *obs* and outputs a Q-value for every action.
    """

    def __init__(
        self,
        feature_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        layers: list[nn.Module] = []
        in_dim = feature_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        q_values = self.net(obs)  # (B, n_actions)
        if actions is not None:
            # Gather Q-values for the specified actions
            return q_values.gather(1, actions.long().unsqueeze(-1)).squeeze(-1)  # (B,)
        return q_values


class ContinuousQCritic(BaseQCritic):
    """Q(s, a) → R  for continuous action spaces.

    The network takes concatenated *[obs, action]* and outputs a scalar
    Q-value.  This is the standard formulation used by DDPG, TD3, SAC.
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = feature_dim + action_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor | None = None) -> torch.Tensor:
        if actions is None:
            raise ValueError("ContinuousQCritic requires explicit actions.")
        x = torch.cat([obs, actions], dim=-1)
        return self.net(x).squeeze(-1)  # (B,)


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
        q1_val = self.q1(obs, actions)
        q2_val = self.q2(obs, actions)
        return torch.min(q1_val, q2_val)

    def both(
        self, obs: torch.Tensor, actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both Q-values (needed for the loss computation)."""
        return self.q1(obs, actions), self.q2(obs, actions)
