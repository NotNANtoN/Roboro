"""DQN algorithm recipe — wires components for discrete Q-learning."""

from __future__ import annotations

import gymnasium as gym

from roboro.actors.epsilon_greedy import EpsilonGreedyActor
from roboro.critics.q import DiscreteQCritic
from roboro.critics.target import TargetNetwork
from roboro.data.replay_buffer import ReplayBuffer
from roboro.training.trainer import TrainResult, train_off_policy
from roboro.updates.dqn import DQNUpdate


def make_dqn(
    env_id: str,
    *,
    hidden_dim: int = 128,
    n_layers: int = 2,
    lr: float = 1e-3,
    gamma: float = 0.99,
    buffer_size: int = 50_000,
    batch_size: int = 128,
    warmup_steps: int = 1000,
    total_steps: int = 50_000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 20_000,
    target_update_mode: str = "hard",
    target_update_period: int = 500,
    tau: float = 0.005,
    eval_interval: int = 2000,
    eval_episodes: int = 5,
    seed: int = 0,
) -> TrainResult:
    """Construct and train a DQN agent.

    This is a *recipe*: it wires together a ``DiscreteQCritic``, an
    ``EpsilonGreedyActor``, a ``TargetNetwork``, a ``ReplayBuffer``, and the
    ``DQNUpdate`` rule, then runs ``train_off_policy``.
    """
    env = gym.make(env_id)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    obs_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    # ── build components ────────────────────────────────────────────────────
    q_critic = DiscreteQCritic(
        feature_dim=obs_dim, n_actions=n_actions, hidden_dim=hidden_dim, n_layers=n_layers
    )
    target = TargetNetwork(
        q_critic, mode=target_update_mode, tau=tau, hard_update_period=target_update_period
    )
    actor = EpsilonGreedyActor(q_critic, n_actions=n_actions, epsilon=epsilon_start)
    buffer = ReplayBuffer(capacity=buffer_size)
    update = DQNUpdate(q_critic, target, lr=lr, gamma=gamma)

    # ── epsilon schedule (linear decay) ─────────────────────────────────────
    class _EpsilonScheduler:
        """Simple linear epsilon decay via the training loop's step counter."""

        def __init__(self) -> None:
            self._original_update = update.update

        def __call__(self, batch: object, step: int) -> object:
            # Decay epsilon
            frac = min(1.0, step / epsilon_decay_steps)
            actor.epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
            return self._original_update(batch, step)  # type: ignore[arg-type]

    update.update = _EpsilonScheduler()  # type: ignore[assignment]

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
