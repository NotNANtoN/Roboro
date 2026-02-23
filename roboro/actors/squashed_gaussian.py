"""Squashed Gaussian actor for SAC.

Outputs a tanh-squashed sample from a learned diagonal Gaussian.
The reparameterization trick enables low-variance policy gradients.
"""

from typing import Any

import torch
from torch import nn
from torch.distributions import Normal

from roboro.actors.base import BaseActor
from roboro.nn.blocks import MLPBlock

# Clamp bounds for numerical stability (matches CleanRL / SB3).
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class SquashedGaussianActor(BaseActor):
    """Stochastic policy: obs -> tanh(mu + sigma * eps), with exact log-prob.

    Architecture: shared MLP trunk → two linear heads (mean, log_std).
    The action is sampled via the reparameterization trick, squashed through
    ``tanh``, and scaled to ``[action_low, action_high]``.

    Used by SAC — the log-probability accounts for the tanh change-of-variables.

    Args:
        obs_dim: dimensionality of the observation / feature vector.
        action_dim: dimensionality of the action vector.
        action_low: lower bound of each action dimension.
        action_high: upper bound of each action dimension.
        hidden_dim: width of hidden layers in the trunk.
        n_layers: total layers in trunk (``n_layers - 1`` hidden + 1 output).
        activation: hidden activation name (resolved by ``get_activation``).
        use_layer_norm: apply ``LayerNorm`` after each hidden layer.
    """

    action_low: torch.Tensor
    action_high: torch.Tensor

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: float = -1.0,
        action_high: float = 1.0,
        *,
        hidden_dim: int = 256,
        n_layers: int = 2,
        activation: str = "relu",
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim

        # Register action bounds as buffers (move with .to(device))
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

        # Shared trunk — output_activation=None so we get raw features.
        # The trunk outputs `hidden_dim` features (last hidden layer).
        self.trunk = MLPBlock(
            in_dim=obs_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            output_activation=activation,  # keep features active
            use_layer_norm=use_layer_norm,
        )

        # Two linear heads on top of the trunk
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        self._init_heads()

    # ── internals ────────────────────────────────────────────────────────────

    def _init_heads(self) -> None:
        """Small-scale initialization for the output heads (stable start)."""
        for head in (self.mean_head, self.log_std_head):
            nn.init.uniform_(head.weight, -1e-3, 1e-3)
            nn.init.constant_(head.bias, 0.0)

    def _scale_action(self, tanh_action: torch.Tensor) -> torch.Tensor:
        """Map tanh output [-1, 1] → [action_low, action_high]."""
        return self.action_low + (tanh_action + 1.0) * 0.5 * (self.action_high - self.action_low)

    def _get_distribution(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) of the pre-squash Gaussian."""
        features = self.trunk(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        return mean, std

    # ── BaseActor interface ──────────────────────────────────────────────────

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample an action for environment interaction.

        Args:
            obs: ``(B, obs_dim)``
            deterministic: if ``True``, return ``tanh(mean)`` (no sampling).

        Returns:
            actions: ``(B, action_dim)`` in ``[action_low, action_high]``.
            info: empty dict.
        """
        mean, std = self._get_distribution(obs)
        if deterministic:
            raw = torch.tanh(mean)
        else:
            dist = Normal(mean, std)
            raw = torch.tanh(dist.sample())
        return self._scale_action(raw), {}

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample + exact log-prob (for training).

        The log-probability accounts for the tanh squashing via the
        change-of-variables formula::

            log pi(a|s) = log N(u|mu,sigma) - sum_i log(1 - tanh^2(u_i))

        where ``u`` is the pre-squash sample and ``a = tanh(u)``.

        Args:
            obs: ``(B, obs_dim)``

        Returns:
            actions: ``(B, action_dim)`` in ``[action_low, action_high]``.
            log_probs: ``(B,)`` log-probability of each action.
        """
        mean, std = self._get_distribution(obs)
        dist = Normal(mean, std)

        # Reparameterized sample: u = mu + sigma * eps
        u = dist.rsample()
        tanh_u = torch.tanh(u)

        # Log-prob with tanh correction (numerically stable)
        log_prob = dist.log_prob(u)  # (B, action_dim)
        # Correction: -log(1 - tanh²(u)) = -log(1 - tanh(u)²)
        log_prob = log_prob - torch.log(1.0 - tanh_u.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)  # (B,)

        action = self._scale_action(tanh_u)
        return action, log_prob
