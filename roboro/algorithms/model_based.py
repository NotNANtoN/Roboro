"""1-Step Model-Based (Simplified MuZero) recipe.

This recipe wires together the 1-step MCTS components based on `ModelBasedCfg`.
It uses the `MUZERO_1STEP` preset.
"""

from typing import Any, cast

import gymnasium as gym
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from roboro.actors.mcts import MCTSActor
from roboro.core.config import ModelBasedCfg
from roboro.core.device import maybe_compile
from roboro.data.replay_buffer import ReplayBuffer
from roboro.nn.blocks import CategoricalSupport, MLPBlock
from roboro.training.trainer import TrainResult, train_off_policy
from roboro.updates.world_model import WorldModelUpdate


class CategoricalValueNet(nn.Module):
    """Categorical Value Network using C51 Support."""

    support: CategoricalSupport

    def __init__(self, obs_dim: int, cfg: Any = None) -> None:
        super().__init__()
        hidden_dim = getattr(cfg, "hidden_dim", 256)
        n_layers = getattr(cfg, "n_layers", 2)
        activation = getattr(cfg, "activation", "relu")
        use_layer_norm = getattr(cfg, "use_layer_norm", False)

        # Determine number of atoms based on cfg
        self.num_atoms = getattr(cfg, "num_atoms", 601)
        self.v_min = getattr(cfg, "v_min", -300.0)
        self.v_max = getattr(cfg, "v_max", 300.0)

        self.support = CategoricalSupport(
            v_min=self.v_min, v_max=self.v_max, num_atoms=self.num_atoms
        )

        self.mlp = MLPBlock(
            in_dim=obs_dim,
            out_dim=self.num_atoms,  # predict logits for each bin
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, obs: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        logits = cast(torch.Tensor, self.mlp(obs))
        if return_logits:
            return logits
        # Return expected scalar value by default for compatibility with MCTS
        return cast(torch.Tensor, self.support(logits))


class CategoricalDynamicsNet(nn.Module):
    """Predicts next obs, reward (categorical), and done given (obs, discrete_action)."""

    def __init__(self, obs_dim: int, n_actions: int, cfg: Any = None) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.obs_dim = obs_dim

        hidden_dim = getattr(cfg, "hidden_dim", 256)
        n_layers = getattr(cfg, "n_layers", 2)
        activation = getattr(cfg, "activation", "relu")
        use_layer_norm = getattr(cfg, "use_layer_norm", False)

        # Determine number of atoms for reward (often smaller range than value)
        self.num_atoms = getattr(cfg, "num_atoms", 601)
        self.v_min = getattr(cfg, "v_min", -300.0)
        self.v_max = getattr(cfg, "v_max", 300.0)

        self.support = CategoricalSupport(
            v_min=self.v_min, v_max=self.v_max, num_atoms=self.num_atoms
        )

        # Concatenate obs + one-hot action
        self.mlp = MLPBlock(
            in_dim=obs_dim + n_actions,
            out_dim=obs_dim + self.num_atoms + 1,  # next_obs + reward_logits + done_logits
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor, return_logits: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # action is (B,) integer. Convert to one-hot
        action_onehot = F.one_hot(action.long(), num_classes=self.n_actions).float()
        x = torch.cat([obs, action_onehot], dim=-1)
        out = self.mlp(x)

        next_obs = out[..., : self.obs_dim]
        reward_logits = out[..., self.obs_dim : self.obs_dim + self.num_atoms]
        done_logits = out[..., -1:]

        reward_out = (
            reward_logits if return_logits else cast(torch.Tensor, self.support(reward_logits))
        )

        return next_obs, reward_out, done_logits


def train_model_based(
    env_id: str,
    cfg: ModelBasedCfg | None = None,
) -> TrainResult:
    """Build and train a 1-step model-based agent (MCTS + AlphaZero update).

    Args:
        env_id: Gymnasium environment id (must have discrete actions).
        cfg: Full configuration. Uses ``ModelBasedCfg()`` defaults when ``None``.
            Device is read from ``cfg.train.device``.
    """
    if cfg is None:
        cfg = ModelBasedCfg()

    # Seed all RNGs before any model creation (weight init must be deterministic)
    from roboro.core.seed import set_seed

    set_seed(cfg.train.seed)

    device = torch.device(cfg.train.device)

    env = gym.make(env_id)
    assert isinstance(
        env.observation_space, gym.spaces.Box
    ), "Only Box observation space supported."
    assert isinstance(
        env.action_space, gym.spaces.Discrete
    ), "Only Discrete action space supported."

    obs_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    # ── build components ────────────────────────────────────────────────────

    dynamics_net = CategoricalDynamicsNet(obs_dim, n_actions, cfg.dynamics_network).to(device)

    value_net = CategoricalValueNet(obs_dim, cfg.value_network).to(device)

    policy_net = MLPBlock(
        in_dim=obs_dim,
        out_dim=n_actions,
        hidden_dim=cfg.policy_network.hidden_dim,
        n_layers=cfg.policy_network.n_layers,
        activation=cfg.policy_network.activation,
        use_layer_norm=cfg.policy_network.use_layer_norm,
    ).to(device)

    # torch.compile
    if cfg.train.compile:
        dynamics_net = maybe_compile(dynamics_net)  # type: ignore[assignment]
        value_net = maybe_compile(value_net)  # type: ignore[assignment]
        policy_net = maybe_compile(policy_net)  # type: ignore[assignment]

    actor = MCTSActor(
        dynamics_net=dynamics_net,
        value_net=value_net,
        policy_net=policy_net,
        num_actions=n_actions,
        num_simulations=cfg.num_simulations,
        discount=cfg.gamma,
        c_puct=cfg.c_puct,
        temperature=cfg.temperature,
        dirichlet_alpha=cfg.dirichlet_alpha,
        dirichlet_fraction=cfg.dirichlet_fraction,
    ).to(device)

    buffer = ReplayBuffer(
        capacity=cfg.buffer.capacity,
        obs_shape=(obs_dim,),
        action_shape=(),  # discrete: scalar actions
        seed=cfg.train.seed,
    )

    update = WorldModelUpdate(
        dynamics_net=dynamics_net,
        value_net=value_net,
        policy_net=policy_net,
        gamma=cfg.gamma,
        lr=cfg.lr,
        max_grad_norm=cfg.max_grad_norm,
    )

    # ── train ───────────────────────────────────────────────────────────────
    result = train_off_policy(
        env=env,
        actor=actor,
        update=update,
        buffer=buffer,
        cfg=cfg.train,
        device=device,
    )
    env.close()
    return result
