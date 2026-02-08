"""Target network utilities — composable, not baked into critics."""

from __future__ import annotations

import copy

import torch
from torch import nn


class TargetNetwork(nn.Module):
    """Wraps any network with a frozen, slowly-updated copy.

    Supports both hard (periodic copy) and soft (Polyak averaging) updates.
    CrossQ-style algorithms simply don't create a TargetNetwork at all.

    Example::

        critic = ContinuousQCritic(...)
        target_critic = TargetNetwork(critic, mode="polyak", tau=0.005)

        # In the training loop:
        target_q = target_critic(obs, actions)  # forward through frozen copy
        target_critic.update()                  # soft or hard update
    """

    def __init__(
        self,
        source: nn.Module,
        mode: str = "polyak",
        tau: float = 0.005,
        hard_update_period: int = 1000,
    ) -> None:
        super().__init__()
        self.source = source
        self.target = copy.deepcopy(source)
        # Freeze target parameters
        for p in self.target.parameters():
            p.requires_grad = False

        assert mode in ("polyak", "hard"), f"Unknown mode: {mode}"
        self.mode = mode
        self.tau = tau
        self.hard_update_period = hard_update_period
        self._step_count = 0

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the **target** network (no grad)."""
        with torch.no_grad():
            return self.target(*args, **kwargs)

    def update(self) -> None:
        """Update the target toward the source."""
        self._step_count += 1
        if self.mode == "polyak":
            self._polyak_update()
        elif self.mode == "hard" and self._step_count % self.hard_update_period == 0:
            self._hard_update()

    def _polyak_update(self) -> None:
        for tp, sp in zip(self.target.parameters(), self.source.parameters(), strict=True):
            tp.data.lerp_(sp.data, 1.0 - self.tau)

    def _hard_update(self) -> None:
        self.target.load_state_dict(self.source.state_dict())
