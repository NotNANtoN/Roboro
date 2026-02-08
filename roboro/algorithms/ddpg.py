"""DDPG algorithm recipe — wires components for continuous control."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from roboro.actors.deterministic import DeterministicActor
from roboro.critics.q import ContinuousQCritic
from roboro.critics.target import TargetNetwork
from roboro.data.replay_buffer import ReplayBuffer
from roboro.training.trainer import TrainResult, train_off_policy
from roboro.updates.ddpg import DDPGUpdate


def make_ddpg(
    env_id: str,
    *,
    hidden_dim: int = 256,
    n_layers: int = 2,
    actor_lr: float = 1e-3,
    critic_lr: float = 1e-3,
    gamma: float = 0.99,
    tau: float = 0.005,
    noise_std: float = 0.1,
    buffer_size: int = 100_000,
    batch_size: int = 256,
    warmup_steps: int = 1000,
    total_steps: int = 50_000,
    eval_interval: int = 2000,
    eval_episodes: int = 5,
    seed: int = 0,
) -> TrainResult:
    """Construct and train a DDPG agent.

    Wires together a ``DeterministicActor``, a ``ContinuousQCritic``,
    ``TargetNetwork`` wrappers for both, and the ``DDPGUpdate`` rule.
    """
    env = gym.make(env_id)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = float(np.min(env.action_space.low))
    action_high = float(np.max(env.action_space.high))

    # ── build components ────────────────────────────────────────────────────
    actor = DeterministicActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        noise_std=noise_std,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    )
    actor_target = TargetNetwork(actor, mode="polyak", tau=tau)

    critic = ContinuousQCritic(
        feature_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, n_layers=n_layers
    )
    critic_target = TargetNetwork(critic, mode="polyak", tau=tau)

    buffer = ReplayBuffer(capacity=buffer_size)

    update = DDPGUpdate(
        actor=actor,
        actor_target=actor_target,
        critic=critic,
        critic_target=critic_target,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
    )

    # ── train ───────────────────────────────────────────────────────────────
    result = train_off_policy(
        env=env,
        actor=actor,
        update=update,
        buffer=buffer,
        total_steps=total_steps,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
    )
    env.close()
    return result
