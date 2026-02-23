"""Epsilon-greedy actor wrapping a discrete Q-critic."""

from typing import Any, cast

import torch

from roboro.actors.base import BaseActor
from roboro.critics.base import BaseQCritic


class EpsilonGreedyActor(BaseActor):
    """Selects actions via ε-greedy exploration over a Q-network.

    * With probability ``epsilon`` → random action.
    * Otherwise → argmax Q(s, ·).

    ``epsilon`` is mutable so it can be decayed during training.
    """

    def __init__(
        self,
        q_critic: BaseQCritic,
        n_actions: int,
        epsilon: float = 1.0,
    ) -> None:
        super().__init__()
        self.q_critic = q_critic
        self.n_actions = n_actions
        self.epsilon = epsilon

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """ε-greedy action selection.

        Args:
            obs: ``(B, feature_dim)``
            deterministic: if ``True``, always take argmax (no exploration).

        Returns:
            action: ``(B,)`` integer action indices.
            info: empty dict.
        """
        q_values = cast(torch.Tensor, self.q_critic(obs))  # (B, n_actions)
        if deterministic or self.epsilon <= 0.0:
            return q_values.argmax(dim=-1), {}

        batch_size = obs.shape[0]
        greedy = q_values.argmax(dim=-1)
        random_actions = torch.randint(0, self.n_actions, (batch_size,), device=obs.device)
        mask = torch.rand(batch_size, device=obs.device) < self.epsilon
        action = torch.where(mask, random_actions, greedy)
        return action, {}

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, None]:
        """For DQN, forward just returns the greedy action (no log-probs)."""
        q_values = self.q_critic(obs)
        return q_values.argmax(dim=-1), None
