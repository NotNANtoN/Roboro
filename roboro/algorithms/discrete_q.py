"""Discrete Q-learning recipe — DQN, Double DQN, etc.

This is a *recipe*: it wires together orthogonal components based on
``DiscreteQCfg``.  The specific variant (DQN, Double DQN, …) is fully
determined by the config.  Use presets from ``roboro.presets`` for named
algorithms::

    from roboro.presets import DQN, DOUBLE_DQN
    from roboro.algorithms.discrete_q import train_discrete_q

    result = train_discrete_q("CartPole-v1", cfg=DQN)
"""

from __future__ import annotations

import gymnasium as gym
import torch

from roboro.actors.epsilon_greedy import EpsilonGreedyActor
from roboro.core.config import DiscreteQCfg
from roboro.core.device import maybe_compile
from roboro.critics.q import DiscreteQCritic
from roboro.critics.target import TargetNetwork
from roboro.data.replay_buffer import ReplayBuffer
from roboro.training.trainer import TrainResult, train_off_policy
from roboro.updates.dqn import DQNUpdate


def train_discrete_q(
    env_id: str,
    cfg: DiscreteQCfg | None = None,
) -> TrainResult:
    """Build and train a discrete Q-learning agent.

    Args:
        env_id: Gymnasium environment id (must have discrete actions).
        cfg: Full configuration.  Uses ``DiscreteQCfg()`` defaults when ``None``.
            Device is read from ``cfg.train.device``.
    """
    if cfg is None:
        cfg = DiscreteQCfg()
    device = torch.device(cfg.train.device)

    env = gym.make(env_id)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    obs_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    # ── build components ────────────────────────────────────────────────────
    net = cfg.network
    q_critic = DiscreteQCritic(
        feature_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=net.hidden_dim,
        n_layers=net.n_layers,
        activation=net.activation,
        use_layer_norm=net.use_layer_norm,
    ).to(device)

    target = TargetNetwork(
        q_critic,
        mode=cfg.target.mode,
        tau=cfg.target.tau,
        hard_update_period=cfg.target.hard_update_period,
    ).to(device)

    # torch.compile — apply after creating target (deepcopy), before actor
    if cfg.train.compile:
        q_critic = maybe_compile(q_critic)  # type: ignore[assignment]

    actor = EpsilonGreedyActor(q_critic, n_actions=n_actions, epsilon=cfg.epsilon_start).to(device)

    buffer = ReplayBuffer(
        capacity=cfg.buffer.capacity,
        obs_shape=(obs_dim,),
        action_shape=(),  # discrete: scalar actions
    )

    update = DQNUpdate(
        q_critic,
        target,
        lr=cfg.lr,
        gamma=cfg.gamma,
        td_loss=cfg.td_loss,
        max_grad_norm=cfg.max_grad_norm,
        double_q=cfg.double_q,
    )

    # ── epsilon schedule (linear decay) ─────────────────────────────────────
    _decay_steps = cfg.epsilon_decay_steps
    _eps_start = cfg.epsilon_start
    _eps_end = cfg.epsilon_end

    class _EpsilonScheduler:
        """Simple linear epsilon decay via the training loop's step counter."""

        def __init__(self) -> None:
            self._original_update = update.update

        def __call__(self, batch: object, step: int) -> object:
            frac = min(1.0, step / _decay_steps)
            actor.epsilon = _eps_start + frac * (_eps_end - _eps_start)
            return self._original_update(batch, step)  # type: ignore[arg-type]

    update.update = _EpsilonScheduler()  # type: ignore[assignment]

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
