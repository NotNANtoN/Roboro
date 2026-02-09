"""DQN update rule: classic Q-learning with target network."""

from __future__ import annotations

import torch
from torch import nn, optim

from roboro.core.types import Batch
from roboro.critics.base import BaseQCritic
from roboro.critics.target import TargetNetwork
from roboro.nn.losses import TDLossFn, get_td_loss
from roboro.updates.base import BaseUpdate, UpdateResult


class DQNUpdate(BaseUpdate):
    """One-step Q-learning (DQN / Double DQN).

    Standard DQN target:
        ``y = r + gamma * max_a' Q_target(s', a') * (1 - done)``

    Double DQN target (``double_q=True``):
        ``a* = argmax_a' Q_online(s', a')``
        ``y = r + gamma * Q_target(s', a*) * (1 - done)``

    The TD loss function is configurable (Huber / MSE).  The original DQN
    paper (Mnih et al., 2015) clips the TD error to [-1, 1], which is
    equivalent to Huber (smooth L1) loss.

    The target network is updated every ``update()`` call (Polyak or hard
    depending on how ``TargetNetwork`` was configured).
    """

    def __init__(
        self,
        q_critic: BaseQCritic,
        target: TargetNetwork,
        lr: float = 1e-3,
        gamma: float = 0.99,
        td_loss: str | TDLossFn = "huber",
        max_grad_norm: float = 10.0,
        double_q: bool = False,
    ) -> None:
        self.q_critic = q_critic
        self.target = target
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.double_q = double_q
        self.optimizer = optim.Adam(q_critic.parameters(), lr=lr)

        # Resolve loss function by name or accept a callable directly
        if isinstance(td_loss, str):
            self._td_loss: TDLossFn = get_td_loss(td_loss)
        else:
            self._td_loss = td_loss

    def update(self, batch: Batch, step: int) -> UpdateResult:
        """Run one DQN gradient step."""
        # ── compute targets ─────────────────────────────────────────────────
        with torch.no_grad():
            if self.double_q:
                # Online net selects best action, target net evaluates it
                next_q_online = self.q_critic(batch.next_obs)  # (B, n_actions)
                best_actions = next_q_online.argmax(dim=-1, keepdim=True)  # (B, 1)
                next_q_target = self.target(batch.next_obs)  # (B, n_actions)
                max_next_q = next_q_target.gather(1, best_actions).squeeze(-1)  # (B,)
            else:
                next_q = self.target(batch.next_obs)  # (B, n_actions)
                max_next_q = next_q.max(dim=-1).values  # (B,)

            td_target = batch.rewards + self.gamma * max_next_q * (~batch.dones).float()

        # ── compute current Q ───────────────────────────────────────────────
        current_q = self.q_critic(batch.obs, batch.actions)  # (B,)

        loss = self._td_loss(current_q, td_target)

        # ── gradient step ───────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.q_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # ── target update ───────────────────────────────────────────────────
        self.target.update()

        return UpdateResult(
            loss=loss.item(),
            metrics={
                "q_mean": current_q.mean().item(),
                "td_target_mean": td_target.mean().item(),
                "grad_norm": float(grad_norm),
            },
        )
