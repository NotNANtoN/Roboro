"""Standard MLP building block — the *one* place layer construction lives."""

from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


def get_activation(name: str) -> type[nn.Module]:
    """Resolve an activation function by name.

    Returns the **class** (not an instance) so callers can instantiate it.
    """
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "mish": nn.Mish,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "identity": nn.Identity,
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(activations)}")
    return activations[key]


class CategoricalSupport(nn.Module):
    """C51 Categorical Distribution Support.

    Transforms scalars into categorical distributions over a fixed set of bins,
    and vice versa. Used to stabilize learning of large, unbounded values/rewards.
    """

    support: torch.Tensor

    def __init__(self, v_min: float = -300.0, v_max: float = 300.0, num_atoms: int = 601) -> None:
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # Register support as a buffer so it moves to the correct device automatically
        support = torch.linspace(v_min, v_max, num_atoms)
        self.register_buffer("support", support)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to expected scalar value."""
        probs = F.softmax(logits, dim=-1)
        support_tensor = cast(torch.Tensor, self.support)
        return (probs * support_tensor).sum(dim=-1, keepdim=True)

    def to_categorical(self, x: torch.Tensor) -> torch.Tensor:
        """Convert scalar value to categorical target distribution using l2 projection."""
        x = x.squeeze(-1).clamp(self.v_min, self.v_max)
        batch_size = x.shape[0]

        # Compute bin indices and weights
        b = (x - self.v_min) / self.delta_z
        l_idx = b.floor().long()
        u_idx = b.ceil().long()

        # Handle exact matches (l_idx == u_idx) to prevent zero weights
        exact_matches = l_idx == u_idx
        u_idx[exact_matches] += 1

        # When clamping u_idx to self.num_atoms - 1, we must also ensure l_idx is valid
        # If b is exactly at v_max, l_idx and u_idx will both initially be num_atoms - 1
        # Then exact_matches makes u_idx = num_atoms.
        # Clamping u_idx makes it num_atoms - 1 again.
        # But wait, if they are both num_atoms - 1, wl = 0 and wu = 0.
        # Let's fix exact matches more robustly:
        u_idx = u_idx.clamp(max=self.num_atoms - 1)
        l_idx = l_idx.clamp(max=self.num_atoms - 1)

        # We need to handle the case where x is exactly v_max differently,
        # or just use the standard l2 projection formula which inherently handles this if done right.
        # If l_idx == u_idx, then b is an integer.
        # The weight wl = u_idx - b = 0. wu = b - l_idx = 0. We'd get 0 total weight.
        # The standard fix is to force l_idx and u_idx apart by adding 1 to u_idx BEFORE clamping,
        # or setting weights manually for exact matches.

        # Weights
        wl = u_idx.float() - b
        wu = b - l_idx.float()

        # Fix exact match weights: if l_idx == b, then it's exactly on a bin.
        # All weight should go to l_idx.
        exact_matches = l_idx.float() == b
        wl[exact_matches] = 1.0
        wu[exact_matches] = 0.0
        u_idx[exact_matches] = l_idx[exact_matches]  # Keep u_idx within bounds

        # Build target distribution
        target = torch.zeros(batch_size, self.num_atoms, device=x.device)
        # Using scatter_add_ to safely accumulate weights
        target.scatter_add_(1, l_idx.unsqueeze(1), wl.unsqueeze(1))
        target.scatter_add_(1, u_idx.unsqueeze(1), wu.unsqueeze(1))
        return target

    def c51_project(
        self, next_dist: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        """Project the next state distribution onto the current support.

        Args:
            next_dist: Probabilities of the next state/action (B, num_atoms).
            rewards: Rewards tensor (B,).
            dones: Boolean termination tensor (B,).
            gamma: Discount factor.

        Returns:
            Target categorical distribution (B, num_atoms).
        """
        batch_size = next_dist.shape[0]

        # Ensure rewards and dones are (B, 1) to broadcast with support (num_atoms,)
        rewards = rewards.view(batch_size, 1)
        dones = dones.view(batch_size, 1).float()

        # Compute t_z = R + gamma * z * (1 - done)
        support_tensor = cast(torch.Tensor, self.support)
        t_z = rewards + gamma * support_tensor.unsqueeze(0) * (1.0 - dones)
        t_z = t_z.clamp(self.v_min, self.v_max)

        # Compute bin indices and weights
        b = (t_z - self.v_min) / self.delta_z
        l_idx = b.floor().long()
        u_idx = b.ceil().long()

        l_idx = l_idx.clamp(max=self.num_atoms - 1)
        u_idx = u_idx.clamp(max=self.num_atoms - 1)

        wl = u_idx.float() - b
        wu = b - l_idx.float()

        # Handle exact matches robustly
        exact_matches = l_idx.float() == b
        wl[exact_matches] = 1.0
        wu[exact_matches] = 0.0
        u_idx[exact_matches] = l_idx[exact_matches]

        # Distribute probabilities from next_dist into the current support bins
        target = torch.zeros(batch_size, self.num_atoms, device=next_dist.device)
        offset = (
            torch.linspace(
                0, (batch_size - 1) * self.num_atoms, batch_size, device=next_dist.device
            )
            .long()
            .unsqueeze(1)
        )

        # Flatten for fast scatter
        target.view(-1).index_add_(0, (l_idx + offset).view(-1), (next_dist * wl).view(-1))
        target.view(-1).index_add_(0, (u_idx + offset).view(-1), (next_dist * wu).view(-1))

        return target


class MLPBlock(nn.Module):
    """General-purpose MLP: the standard block architecture.

    Architecture (``n_layers=3``, ``hidden_dim=256``)::

        in_dim → 256 → act → 256 → act → out_dim [→ output_act]

    * **Hidden layers**: ``n_layers - 1`` layers of size ``hidden_dim`` with
      ``activation`` (and optional ``LayerNorm``).
    * **Output layer**: one ``Linear(hidden_dim, out_dim)`` with optional
      ``output_activation``.

    This is the single block that encoders, critics, and actors compose.
    To switch the backbone (e.g. SambaV2), replace or subclass ``MLPBlock``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int = 256,
        n_layers: int = 2,
        activation: str = "relu",
        output_activation: str | None = None,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        assert n_layers >= 1, "Need at least 1 layer (the output layer)."

        act_cls = get_activation(activation)
        layers: list[nn.Module] = []

        # ── hidden layers ───────────────────────────────────────────────────
        prev_dim = in_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_cls())
            prev_dim = hidden_dim

        # ── output layer ────────────────────────────────────────────────────
        layers.append(nn.Linear(prev_dim, out_dim))
        if output_activation is not None:
            out_act_cls = get_activation(output_activation)
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(out_act_cls())

        self.net = nn.Sequential(*layers)
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._init_weights()

    # ── properties ──────────────────────────────────────────────────────────
    @property
    def in_dim(self) -> int:
        return self._in_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    # ── forward ─────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))

    # ── init ────────────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
