"""DDPG / TD3 update rule: deterministic actor-critic with target networks.

Supports three orthogonal extensions that together constitute TD3:

* **Twin Q** (``twin_q``): clipped double-Q critic to combat overestimation.
* **Delayed actor** (``actor_delay``): update actor & targets every *N* critic steps.
* **Target smoothing** (``target_noise``): regularize targets with clipped action noise.

DDPG is the special case where all three are disabled (the default).
"""

import torch
from torch import nn, optim

from roboro.actors.deterministic import DeterministicActor
from roboro.core.types import Batch
from roboro.critics.base import BaseQCritic
from roboro.critics.q import TwinQCritic
from roboro.critics.target import TargetNetwork
from roboro.updates.base import BaseUpdate, UpdateResult


class DDPGUpdate(BaseUpdate):
    """Deterministic policy gradient with optional TD3 extensions.

    **DDPG** (all defaults)::

        Critic: MSE( Q(s,a),  r + gamma * Q_tgt(s', pi_tgt(s')) * (1-d) )
        Actor:  -mean( Q(s, pi(s)) )
        Targets: Polyak-updated every step.

    **TD3** (``twin_q=True, actor_delay=2, target_noise=0.2``)::

        Critic: MSE(Q1, y) + MSE(Q2, y)
            y = r + gamma * min(Q1_tgt, Q2_tgt)(s', pi_tgt(s') + clip(eps)) * (1-d)
        Actor:  -mean( Q1(s, pi(s)) )         [updated every `actor_delay` steps]
        Targets: Polyak-updated when actor updates.

    Args:
        actor: Deterministic policy network.
        actor_target: Target-wrapped copy of the actor.
        critic: Single ``BaseQCritic`` (DDPG) or ``TwinQCritic`` (TD3).
        critic_target: Target-wrapped copy of the critic.
        actor_lr: Learning rate for the policy.
        critic_lr: Learning rate for the Q-network(s).
        gamma: Discount factor.
        actor_delay: Update actor & targets every *N* critic steps.
            1 → every step (DDPG).  2 → every other step (TD3).
        target_noise: Std-dev of Gaussian noise added to target actions
            (target policy smoothing).  0 → disabled (DDPG).
        target_noise_clip: Clamp range for target noise.
    """

    def __init__(
        self,
        actor: DeterministicActor,
        actor_target: TargetNetwork,
        critic: BaseQCritic | TwinQCritic,
        critic_target: TargetNetwork,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        actor_delay: int = 1,
        target_noise: float = 0.0,
        target_noise_clip: float = 0.5,
    ) -> None:
        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.gamma = gamma

        # TD3 extensions
        self.actor_delay = actor_delay
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self._is_twin = isinstance(critic, TwinQCritic)

        self.actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    def update(self, batch: Batch, step: int) -> UpdateResult:
        """Run one gradient step (critic always, actor on delay schedule)."""
        # ═══ Critic update ══════════════════════════════════════════════════
        with torch.no_grad():
            next_actions, _ = self.actor_target(batch.next_obs)  # (B, act_dim)

            # Target policy smoothing (TD3): add clipped noise to target actions
            if self.target_noise > 0.0:
                noise = (torch.randn_like(next_actions) * self.target_noise).clamp(
                    -self.target_noise_clip, self.target_noise_clip
                )
                next_actions = (next_actions + noise).clamp(
                    self.actor.action_low, self.actor.action_high
                )

            target_q = self.critic_target(batch.next_obs, next_actions)  # (B,)
            td_target = batch.rewards + self.gamma * target_q * (~batch.dones).float()

        # Critic loss — twin Q uses both networks, single Q uses one
        if self._is_twin:
            assert isinstance(self.critic, TwinQCritic)
            q1, q2 = self.critic.both(batch.obs, batch.actions)
            critic_loss = nn.functional.mse_loss(q1, td_target) + nn.functional.mse_loss(
                q2, td_target
            )
            q_mean = q1.mean().item()
        else:
            current_q = self.critic(batch.obs, batch.actions)  # (B,)
            critic_loss = nn.functional.mse_loss(current_q, td_target)
            q_mean = current_q.mean().item()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ═══ Actor update (delayed for TD3) ═════════════════════════════════
        actor_loss_val = 0.0
        if step % self.actor_delay == 0:
            actions, _ = self.actor(batch.obs)  # (B, act_dim)
            # For twin Q, use Q1 only for actor gradient (TD3 convention)
            if self._is_twin:
                assert isinstance(self.critic, TwinQCritic)
                actor_loss = -self.critic.q1(batch.obs, actions).mean()
            else:
                actor_loss = -self.critic(batch.obs, actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            # ═══ Target updates (synced with actor) ═════════════════════════
            self.critic_target.update()
            self.actor_target.update()

        return UpdateResult(
            loss=critic_loss.item(),
            metrics={
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss_val,
                "q_mean": q_mean,
                "td_target_mean": td_target.mean().item(),
            },
        )
