"""Base planner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from roboro.dynamics.base import BaseDynamics


class BasePlanner(ABC):
    """Selects actions by optimizing over a learned dynamics model.

    The planner does **not** learn parameters itself — it uses a frozen
    dynamics model + reward/value head to search for good action sequences.
    """

    @abstractmethod
    def plan(
        self,
        latent: torch.Tensor,
        dynamics: BaseDynamics,
        horizon: int,
    ) -> torch.Tensor:
        """Return the best first action for the given latent state.

        Args:
            latent: ``(B, latent_dim)``
            dynamics: the world model to unroll.
            horizon: planning horizon (number of steps to look ahead).

        Returns:
            ``(B, action_dim)`` the chosen action.
        """
