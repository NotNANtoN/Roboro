"""Canonical data types that flow through the entire library.

Every component speaks the same language: observations, actions, rewards
are always represented using these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Batch:
    """A batch of transitions sampled from a replay buffer or dataset.

    Required fields are always present.  Optional fields are populated only
    when a specific buffer wrapper or algorithm needs them (e.g. PER weights,
    HER substituted goals, n-step returns).

    All tensor fields have shape ``(batch_size, *feature_shape)`` unless
    otherwise noted.
    """

    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor  # (B,) or (B, 1)
    next_obs: torch.Tensor
    dones: torch.Tensor  # (B,) bool

    # ── optional / populated by wrappers ────────────────────────────────────
    indices: torch.Tensor | None = None  # buffer indices (for PER updates)
    weights: torch.Tensor | None = None  # importance-sampling weights (PER)
    log_probs: torch.Tensor | None = None  # log π(a|s) — needed by on-policy / SAC
    returns: torch.Tensor | None = None  # precomputed returns (n-step, GAE, …)

    extras: dict[str, Any] = field(default_factory=dict)

    # ── helpers ─────────────────────────────────────────────────────────────
    @property
    def batch_size(self) -> int:
        return self.obs.shape[0]

    def to(self, device: torch.device | str) -> Batch:
        """Move every tensor field to *device*, return self for chaining."""
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if isinstance(val, torch.Tensor):
                setattr(self, f, val.to(device))
        return self
