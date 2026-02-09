"""SAC update rule: max-entropy actor-critic with auto alpha-tuning.

Implements the Soft Actor-Critic algorithm (Haarnoja et al., 2018):
  * Critic loss:  MSE on both Q-networks against the soft Bellman target
  * Actor loss:   E[alpha * log pi(a|s) - min Q(s,a)]
  * Alpha loss:   -alpha * (log pi(a|s) + H_target)   (when ``learnable_alpha=True``)
"""

import torch
from torch import nn, optim

from roboro.actors.squashed_gaussian import SquashedGaussianActor
from roboro.core.types import Batch
from roboro.critics.q import TwinQCritic
from roboro.critics.target import TargetNetwork
from roboro.updates.base import BaseUpdate, UpdateResult


class SACUpdate(BaseUpdate):
    """Soft Actor-Critic update with clipped double-Q and entropy tuning.

    Components wired by the algorithm recipe:
      * ``actor``: ``SquashedGaussianActor`` — reparameterized policy.
      * ``critic``: ``TwinQCritic`` — clipped double-Q for anti-overestimation.
      * ``critic_target``: ``TargetNetwork(TwinQCritic)`` — Polyak-averaged target.

    The entropy coefficient ``alpha`` can be:
      * Fixed (``learnable_alpha=False``, uses ``init_alpha``).
      * Auto-tuned (``learnable_alpha=True``, optimizes toward ``target_entropy``).

    Args:
        actor: Squashed Gaussian policy.
        critic: Twin Q-critic (online).
        critic_target: Target-wrapped twin Q-critic.
        actor_lr: Learning rate for the policy network.
        critic_lr: Learning rate for both Q-networks.
        alpha_lr: Learning rate for the entropy coefficient.
        gamma: Discount factor.
        init_alpha: Initial value of the entropy coefficient.
        learnable_alpha: Whether to auto-tune alpha.
        target_entropy: Target entropy for alpha-tuning.
            Defaults to ``-action_dim`` (standard heuristic).
    """

    def __init__(
        self,
        actor: SquashedGaussianActor,
        critic: TwinQCritic,
        critic_target: TargetNetwork,
        *,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        init_alpha: float = 1.0,
        learnable_alpha: bool = True,
        target_entropy: float | None = None,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.gamma = gamma

        # ── entropy coefficient ──────────────────────────────────────────
        self.learnable_alpha = learnable_alpha
        # Store log(alpha) -- optimizing in log-space keeps alpha positive.
        self.log_alpha = torch.tensor(float(init_alpha), dtype=torch.float32).log()
        if learnable_alpha:
            self.log_alpha = nn.Parameter(self.log_alpha)  # type: ignore[assignment]

        self.target_entropy: float = (
            target_entropy if target_entropy is not None else -float(actor.action_dim)
        )

        # ── optimizers ───────────────────────────────────────────────────
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
        self.alpha_optimizer: optim.Adam | None = (
            optim.Adam([self.log_alpha], lr=alpha_lr) if learnable_alpha else None
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy coefficient (always positive)."""
        return self.log_alpha.exp().detach()

    # ── main update ──────────────────────────────────────────────────────────

    def update(self, batch: Batch, step: int) -> UpdateResult:
        """Run one SAC gradient step: critic -> actor -> alpha.

        Args:
            batch: Transition batch from the replay buffer.
            step: Global training step (unused, but part of the interface).

        Returns:
            ``UpdateResult`` with critic loss and detailed metrics.
        """
        alpha = self.log_alpha.exp()

        # ═══ Critic update ══════════════════════════════════════════════════
        with torch.no_grad():
            next_actions, next_log_probs = self.actor(batch.next_obs)
            # Target uses the minimum of both target Q-networks
            target_q = self.critic_target(batch.next_obs, next_actions)  # min(Q1, Q2)
            soft_target = batch.rewards + self.gamma * (~batch.dones).float() * (
                target_q - alpha * next_log_probs
            )

        q1, q2 = self.critic.both(batch.obs, batch.actions)
        critic_loss = nn.functional.mse_loss(q1, soft_target) + nn.functional.mse_loss(
            q2, soft_target
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ═══ Actor update ═══════════════════════════════════════════════════
        actions_pi, log_probs_pi = self.actor(batch.obs)
        q_pi = self.critic(batch.obs, actions_pi)  # min(Q1, Q2)
        actor_loss = (alpha.detach() * log_probs_pi - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ═══ Alpha update ═══════════════════════════════════════════════════
        alpha_loss_val = 0.0
        if self.learnable_alpha and self.alpha_optimizer is not None:
            # Loss: -alpha * (log_pi + H_target)
            alpha_loss = -(
                self.log_alpha.exp() * (log_probs_pi.detach() + self.target_entropy)
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_val = alpha_loss.item()

        # ═══ Target update ══════════════════════════════════════════════════
        self.critic_target.update()

        return UpdateResult(
            loss=critic_loss.item(),
            metrics={
                "critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item(),
                "alpha": alpha.item(),
                "alpha_loss": alpha_loss_val,
                "q_mean": q1.mean().item(),
                "log_prob_mean": log_probs_pi.mean().item(),
                "target_q_mean": soft_target.mean().item(),
            },
        )
