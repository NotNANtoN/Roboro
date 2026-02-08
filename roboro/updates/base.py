"""Base update rule interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roboro.core.types import Batch


@dataclass
class UpdateResult:
    """Return value of an update step — carries loss(es) and metrics to log."""

    loss: float
    metrics: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


class BaseUpdate(ABC):
    """Defines *how* to learn from a batch of transitions.

    An update rule is the composition of critic loss, actor loss, entropy
    tuning, etc.  It owns references to the modules it updates (critic,
    actor, …) but does **not** own them — the algorithm recipe wires things.
    """

    @abstractmethod
    def update(self, batch: Batch, step: int) -> UpdateResult:
        """Run one gradient step on *batch*.

        Args:
            batch: a ``Batch`` from the replay buffer / dataset.
            step: global training step (useful for scheduling).

        Returns:
            ``UpdateResult`` with the scalar loss and logging metrics.
        """
