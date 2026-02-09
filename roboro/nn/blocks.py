"""Standard MLP building block — the *one* place layer construction lives."""

from typing import cast

import torch
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
