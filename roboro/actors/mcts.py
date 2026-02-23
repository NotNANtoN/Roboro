"""MCTS Actor wrapper for model-based planning."""

from typing import Any, cast

import torch
from torch import nn

from roboro.actors.base import BaseActor
from roboro.planners.mcts import run_mcts


class MCTSActor(BaseActor):
    """Selects actions using Monte Carlo Tree Search over a learned world model.

    This actor delegates the actual planning to ``run_mcts()``, passing in
    its networks as callables. It returns the chosen action along with the
    MCTS visit counts (improved policy) and root value estimate in the ``info`` dict.

    Args:
        dynamics_net: a module taking ``(state, action)`` and returning
            ``(next_state, reward)``.
        value_net: a module taking ``(state)`` and returning a scalar value.
        policy_net: a module taking ``(state)`` and returning unnormalized action logits.
        num_actions: size of the discrete action space.
        num_simulations: number of MCTS iterations to run per step.
        discount: reward discount factor (gamma).
        c_puct: exploration constant for the PUCT formula.
        temperature: controls exploration in action selection (1.0 = sample, 0.0 = argmax).
    """

    def __init__(
        self,
        dynamics_net: nn.Module,
        value_net: nn.Module,
        policy_net: nn.Module,
        num_actions: int,
        num_simulations: int = 50,
        discount: float = 0.99,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_fraction: float = 0.25,
    ) -> None:
        super().__init__()
        self.dynamics_net = dynamics_net
        self.value_net = value_net
        self.policy_net = policy_net

        self.num_actions = num_actions
        self.num_simulations = num_simulations
        self.discount = discount
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_fraction = dirichlet_fraction

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run MCTS to select an action and calculate improved policy targets.

        Args:
            obs: ``(B, feature_dim)`` encoded observations.
            deterministic: if ``True``, forces temperature to 0.0 (argmax selection).

        Returns:
            actions: ``(B,)`` chosen discrete actions.
            info: dict containing:
                - ``"mcts_policy"``: ``(B, num_actions)`` visit count distribution.
                - ``"mcts_value"``: ``(B,)`` root node value estimate.
        """
        temp = 0.0 if deterministic else self.temperature

        # Create lightweight closures that handle any necessary tensor unpacking
        def dynamics_fn(
            s: torch.Tensor, a: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            next_s, r, done_logits = self.dynamics_net(s, a)
            # Clip rewards to prevent MCTS explosion
            # done is probability > 0.5 (logit > 0)
            done = done_logits > 0
            return next_s, cast(torch.Tensor, r.clamp(-10.0, 10.0)), done

        def value_fn(s: torch.Tensor) -> torch.Tensor:
            # Clip values to prevent MCTS explosion
            return cast(torch.Tensor, self.value_net(s).squeeze(-1).clamp(-100.0, 100.0))

        def policy_fn(s: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, self.policy_net(s))

        action, mcts_policy, mcts_value = run_mcts(
            state=obs,
            dynamics_fn=dynamics_fn,
            value_fn=value_fn,
            policy_fn=policy_fn,
            num_simulations=self.num_simulations,
            num_actions=self.num_actions,
            discount=self.discount,
            c_puct=self.c_puct,
            temperature=temp,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_fraction=self.dirichlet_fraction,
        )

        # MCTS returns tensors; we push them to CPU for storage in the replay buffer.
        # We squeeze the batch dimension (assumed B=1 in standard training loop)
        info = {
            "mcts_policy": mcts_policy.squeeze(0).cpu().numpy(),
            "mcts_value": mcts_value.squeeze(0).cpu().numpy(),
        }

        return action, info

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Not used for MCTS training (targets come from buffer extras)."""
        raise NotImplementedError("MCTSActor does not support differentiable forward.")
