"""Deterministic actor for DDPG / TD3."""

from typing import Any

import torch
from torch import nn

from roboro.actors.base import BaseActor
from roboro.nn.blocks import MLPBlock


class DeterministicActor(BaseActor):
    """Learned deterministic policy: obs → tanh-squashed continuous action.

    Output is scaled to ``[action_low, action_high]``.  During exploration,
    Gaussian noise is added (controlled by ``noise_std``).

    Args:
        obs_dim: dimensionality of the observation / feature vector.
        action_dim: dimensionality of the action vector.
        action_low: lower bound of each action dimension.
        action_high: upper bound of each action dimension.
        noise_std: standard deviation of exploration noise.
        trunk: optional custom ``nn.Module``.  Defaults to
            ``MLPBlock(obs_dim → action_dim, output_activation="tanh")``.
        **kwargs: forwarded to ``MLPBlock``.
    """

    action_low: torch.Tensor
    action_high: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: float = -1.0,
        action_high: float = 1.0,
        noise_std: float = 0.1,
        trunk: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.noise_std = noise_std

        # Register action bounds as buffers (move with .to(device))
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

        if trunk is not None:
            self.trunk = trunk
        else:
            self.trunk = MLPBlock(
                in_dim=obs_dim,
                out_dim=action_dim,
                output_activation="tanh",
                **kwargs,
            )

    def _scale_action(self, raw: torch.Tensor) -> torch.Tensor:
        """Map tanh output [-1, 1] → [action_low, action_high]."""
        return self.action_low + (raw + 1.0) * 0.5 * (self.action_high - self.action_low)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Deterministic action + optional Gaussian noise.

        Args:
            obs: ``(B, obs_dim)``
            deterministic: if ``True``, suppress exploration noise.

        Returns:
            ``(B, action_dim)`` actions clipped to valid range.
        """
        raw = self.trunk(obs)  # (B, action_dim) in [-1, 1]
        action = self._scale_action(raw)
        if not deterministic and self.noise_std > 0.0:
            noise = torch.randn_like(action) * self.noise_std
            action = (action + noise).clamp(self.action_low, self.action_high)
        return action

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Differentiable forward — returns scaled action (no log-prob)."""
        raw = self.trunk(obs)
        action = self._scale_action(raw)
        return action, None
