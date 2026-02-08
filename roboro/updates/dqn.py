"""DQN update rule: classic Q-learning with target network."""

from __future__ import annotations

import torch
from torch import nn, optim

from roboro.core.types import Batch
from roboro.critics.base import BaseQCritic
from roboro.critics.target import TargetNetwork
from roboro.updates.base import BaseUpdate, UpdateResult


class DQNUpdate(BaseUpdate):
    """One-step Q-learning (DQN).

    Loss: ``MSE( Q(s, a),  r + gamma * max_a' Q_target(s', a') * (1 - done) )``

    The target network is updated every ``update()`` call (Polyak or hard
    depending on how ``TargetNetwork`` was configured).
    """

    def __init__(
        self,
        q_critic: BaseQCritic,
        target: TargetNetwork,
        lr: float = 1e-3,
        gamma: float = 0.99,
    ) -> None:
        self.q_critic = q_critic
        self.target = target
        self.gamma = gamma
        self.optimizer = optim.Adam(q_critic.parameters(), lr=lr)

    def update(self, batch: Batch, step: int) -> UpdateResult:
        """Run one DQN gradient step."""
        # ── compute targets ─────────────────────────────────────────────────
        with torch.no_grad():
            next_q = self.target(batch.next_obs)  # (B, n_actions)
            max_next_q = next_q.max(dim=-1).values  # (B,)
            td_target = batch.rewards + self.gamma * max_next_q * (~batch.dones).float()

        # ── compute current Q ───────────────────────────────────────────────
        current_q = self.q_critic(batch.obs, batch.actions)  # (B,)

        loss = nn.functional.mse_loss(current_q, td_target)

        # ── gradient step ───────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ── target update ───────────────────────────────────────────────────
        self.target.update()

        return UpdateResult(
            loss=loss.item(),
            metrics={
                "q_mean": current_q.mean().item(),
                "td_target_mean": td_target.mean().item(),
            },
        )
