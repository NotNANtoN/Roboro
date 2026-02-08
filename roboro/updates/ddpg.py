"""DDPG update rule: deterministic actor-critic with target networks."""

from __future__ import annotations

import torch
from torch import nn, optim

from roboro.actors.deterministic import DeterministicActor
from roboro.core.types import Batch
from roboro.critics.base import BaseQCritic
from roboro.critics.target import TargetNetwork
from roboro.updates.base import BaseUpdate, UpdateResult


class DDPGUpdate(BaseUpdate):
    """Deep Deterministic Policy Gradient (DDPG).

    Critic loss: ``MSE( Q(s,a), r + gamma * Q_target(s', pi_target(s')) * (1-done) )``
    Actor loss:  ``-mean( Q(s, pi(s)) )``

    Both target networks are Polyak-updated every step.
    """

    def __init__(
        self,
        actor: DeterministicActor,
        actor_target: TargetNetwork,
        critic: BaseQCritic,
        critic_target: TargetNetwork,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
    ) -> None:
        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.gamma = gamma

        self.actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    def update(self, batch: Batch, step: int) -> UpdateResult:
        """Run one DDPG gradient step (critic then actor)."""
        # ═══ Critic update ══════════════════════════════════════════════════
        with torch.no_grad():
            next_actions, _ = self.actor_target(batch.next_obs)  # (B, act_dim)
            target_q = self.critic_target(batch.next_obs, next_actions)  # (B,)
            td_target = batch.rewards + self.gamma * target_q * (~batch.dones).float()

        current_q = self.critic(batch.obs, batch.actions)  # (B,)
        critic_loss = nn.functional.mse_loss(current_q, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ═══ Actor update ═══════════════════════════════════════════════════
        actions, _ = self.actor(batch.obs)  # (B, act_dim)
        actor_loss = -self.critic(batch.obs, actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ═══ Target updates ═════════════════════════════════════════════════
        self.critic_target.update()
        self.actor_target.update()

        return UpdateResult(
            loss=critic_loss.item(),
            metrics={
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item(),
                "q_mean": current_q.mean().item(),
                "td_target_mean": td_target.mean().item(),
            },
        )
