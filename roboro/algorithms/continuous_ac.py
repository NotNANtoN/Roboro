"""Continuous actor-critic recipe — DDPG, TD3, SAC, etc.

This is a *recipe*: it wires together orthogonal components based on
``ContinuousActorCriticCfg``.  The specific variant (DDPG, TD3, SAC, …) is
fully determined by the config.  Use presets from ``roboro.presets`` for named
algorithms::

    from roboro.presets import DDPG, SAC
    from roboro.algorithms.continuous_ac import train_continuous_ac

    result = train_continuous_ac("Pendulum-v1", cfg=SAC)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from roboro.actors.base import BaseActor
from roboro.core.config import ContinuousActorCriticCfg
from roboro.core.device import maybe_compile
from roboro.critics.base import BaseQCritic
from roboro.critics.q import ContinuousQCritic, TwinQCritic
from roboro.critics.target import TargetNetwork
from roboro.data.replay_buffer import ReplayBuffer
from roboro.training.trainer import TrainResult, train_off_policy
from roboro.updates.base import BaseUpdate


def _build_actor(
    cfg: ContinuousActorCriticCfg,
    obs_dim: int,
    action_dim: int,
    action_low: float,
    action_high: float,
) -> BaseActor:
    """Build the actor based on ``cfg.actor_type``."""
    a_net = cfg.actor_network

    if cfg.actor_type == "deterministic":
        from roboro.actors.deterministic import DeterministicActor

        return DeterministicActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            noise_std=cfg.noise_std,
            hidden_dim=a_net.hidden_dim,
            n_layers=a_net.n_layers,
            activation=a_net.activation,
            use_layer_norm=a_net.use_layer_norm,
        )

    if cfg.actor_type == "squashed_gaussian":
        from roboro.actors.squashed_gaussian import SquashedGaussianActor

        return SquashedGaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            hidden_dim=a_net.hidden_dim,
            n_layers=a_net.n_layers,
            activation=a_net.activation,
            use_layer_norm=a_net.use_layer_norm,
        )

    raise ValueError(
        f"Unknown actor_type '{cfg.actor_type}'. Choose from 'deterministic', 'squashed_gaussian'."
    )


def _build_critic(
    cfg: ContinuousActorCriticCfg,
    obs_dim: int,
    action_dim: int,
) -> ContinuousQCritic | TwinQCritic:
    """Build a single or twin Q-critic based on ``cfg.twin_q``."""
    c_net = cfg.critic_network
    kwargs = {
        "feature_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": c_net.hidden_dim,
        "n_layers": c_net.n_layers,
        "activation": c_net.activation,
        "use_layer_norm": c_net.use_layer_norm,
    }

    if cfg.twin_q:
        q1 = ContinuousQCritic(**kwargs)  # type: ignore[arg-type]
        q2 = ContinuousQCritic(**kwargs)  # type: ignore[arg-type]
        return TwinQCritic(q1, q2)

    return ContinuousQCritic(**kwargs)  # type: ignore[arg-type]


def _build_update(
    cfg: ContinuousActorCriticCfg,
    actor: BaseActor,
    critic: ContinuousQCritic | TwinQCritic,
    critic_target: TargetNetwork,
    actor_target: TargetNetwork | None = None,
) -> BaseUpdate:
    """Select the update rule based on the config."""

    if cfg.actor_type == "squashed_gaussian":
        # SAC update — requires TwinQCritic
        from roboro.actors.squashed_gaussian import SquashedGaussianActor
        from roboro.updates.sac import SACUpdate

        assert isinstance(actor, SquashedGaussianActor)
        assert isinstance(critic, TwinQCritic), "SAC requires twin_q=True (clipped double-Q)."
        return SACUpdate(
            actor=actor,
            critic=critic,
            critic_target=critic_target,
            actor_lr=cfg.actor_lr,
            critic_lr=cfg.critic_lr,
            alpha_lr=cfg.alpha_lr,
            gamma=cfg.gamma,
            init_alpha=cfg.init_alpha,
            learnable_alpha=cfg.learnable_alpha,
            target_entropy=cfg.target_entropy,
        )

    # DDPG update — requires DeterministicActor + actor target
    from roboro.actors.deterministic import DeterministicActor
    from roboro.updates.ddpg import DDPGUpdate

    assert isinstance(actor, DeterministicActor)
    assert actor_target is not None, "DDPG requires an actor target network."
    assert isinstance(critic, BaseQCritic), "DDPG requires a single Q-critic (not TwinQCritic)."
    return DDPGUpdate(
        actor=actor,
        actor_target=actor_target,
        critic=critic,
        critic_target=critic_target,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
    )


def train_continuous_ac(
    env_id: str,
    cfg: ContinuousActorCriticCfg | None = None,
) -> TrainResult:
    """Build and train a continuous actor-critic agent.

    The config determines the specific variant:
      * ``actor_type="deterministic"`` → DDPG / TD3
      * ``actor_type="squashed_gaussian"`` + ``twin_q=True`` → SAC

    Args:
        env_id: Gymnasium environment id (must have continuous Box actions).
        cfg: Full configuration.  Uses ``ContinuousActorCriticCfg()`` defaults
            when ``None``.  Device is read from ``cfg.train.device``.
    """
    if cfg is None:
        cfg = ContinuousActorCriticCfg()
    device = torch.device(cfg.train.device)

    env = gym.make(env_id)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = float(np.min(env.action_space.low))
    action_high = float(np.max(env.action_space.high))

    # ── build components ────────────────────────────────────────────────────
    actor = _build_actor(cfg, obs_dim, action_dim, action_low, action_high).to(device)
    critic = _build_critic(cfg, obs_dim, action_dim).to(device)

    # Create targets BEFORE compile (deepcopy of uncompiled module)
    critic_target = TargetNetwork(critic, mode=cfg.target.mode, tau=cfg.target.tau).to(device)

    # DDPG/TD3 need an actor target; SAC does not
    actor_target: TargetNetwork | None = None
    if cfg.actor_type == "deterministic":
        actor_target = TargetNetwork(actor, mode=cfg.target.mode, tau=cfg.target.tau).to(device)

    # torch.compile — apply after creating targets, before update
    if cfg.train.compile:
        actor = maybe_compile(actor)  # type: ignore[assignment]
        critic = maybe_compile(critic)  # type: ignore[assignment]

    buffer = ReplayBuffer(
        capacity=cfg.buffer.capacity,
        obs_shape=(obs_dim,),
        action_shape=(action_dim,),
    )

    update = _build_update(cfg, actor, critic, critic_target, actor_target)

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
