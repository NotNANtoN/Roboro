"""MLP encoder for low-dimensional observations."""

from __future__ import annotations

import torch
from torch import nn

from roboro.encoders.base import BaseEncoder
from roboro.nn.blocks import MLPBlock


class MLPEncoder(BaseEncoder):
    """Maps flat observations to a latent feature vector via an ``MLPBlock``.

    By default builds a standard MLP; pass a custom ``trunk`` to swap
    the architecture (e.g. SambaV2).

    Args:
        obs_dim: dimensionality of the raw observation.
        feature_dim: dimensionality of the output feature vector.
        trunk: optional pre-built ``nn.Module``.  If ``None``, an
            ``MLPBlock`` is created from the remaining kwargs.
        **kwargs: forwarded to ``MLPBlock`` (``hidden_dim``, ``n_layers``,
            ``activation``, ``use_layer_norm``).
    """

    def __init__(
        self,
        obs_dim: int,
        feature_dim: int = 256,
        trunk: nn.Module | None = None,
        **kwargs: int | str | bool,
    ) -> None:
        super().__init__()
        self._feature_dim = feature_dim

        if trunk is not None:
            self.trunk = trunk
        else:
            # default: output activation = relu (encoder produces active features)
            kwargs.setdefault("output_activation", "relu")
            self.trunk = MLPBlock(in_dim=obs_dim, out_dim=feature_dim, **kwargs)

    # ── BaseEncoder interface ───────────────────────────────────────────────
    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)
