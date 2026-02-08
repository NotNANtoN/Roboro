"""MLP encoder for low-dimensional observations."""

from __future__ import annotations

import torch
from torch import nn

from roboro.encoders.base import BaseEncoder


class MLPEncoder(BaseEncoder):
    """Two-layer MLP that maps flat observations to a latent feature vector.

    Architecture::

        obs → Linear → LayerNorm (opt) → act → Linear → LayerNorm (opt) → act → features

    This is the standard *block* architecture that can be swapped for e.g.
    SambaV2 or other drop-in replacements by subclassing or composition.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        feature_dim: int = 256,
        activation: str = "relu",
        use_layer_norm: bool = False,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self._feature_dim = feature_dim

        act_fn = _get_activation(activation)

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _i in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, feature_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(feature_dim))
        layers.append(act_fn())

        self.net = nn.Sequential(*layers)
        self._init_weights()

    # ── BaseEncoder interface ───────────────────────────────────────────────
    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    # ── internals ───────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


def _get_activation(name: str) -> type[nn.Module]:
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "mish": nn.Mish,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(activations)}")
    return activations[name.lower()]
