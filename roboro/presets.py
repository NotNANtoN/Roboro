"""Named algorithm presets.

Each "algorithm" is just a configuration of orthogonal techniques.
``DQN``, ``DOUBLE_DQN``, ``DDPG`` etc. are convenient aliases — every
parameter is spelled out explicitly so you can *see* the decomposition.

Usage::

    from roboro.presets import DQN
    from roboro.algorithms.discrete_q import train_discrete_q

    result = train_discrete_q("CartPole-v1", cfg=DQN)

To tweak a preset, use ``dataclasses.replace``::

    from dataclasses import replace
    my_dqn = replace(DQN, lr=3e-4, double_q=True)
"""

from roboro.core.config import (
    BufferCfg,
    ContinuousActorCriticCfg,
    DiscreteQCfg,
    NetworkCfg,
    TargetNetCfg,
    TrainCfg,
)

# ── Discrete Q-learning family ───────────────────────────────────────────────
#
# Hyperparameters match CleanRL dqn.py (classic control variant).
# Reference results (CleanRL, 3 seeds, 500k steps):
#   CartPole-v1:    488.69 +/- 16.11
#   Acrobot-v1:     -91.54 +/-  7.20
#   MountainCar-v0: -194.95 +/- 8.48

DQN = DiscreteQCfg(
    # Architecture (CleanRL uses 120→84; we use uniform 120 — close enough)
    network=NetworkCfg(hidden_dim=120, n_layers=2, activation="relu", use_layer_norm=False),
    # Target: periodic hard copy every 500 gradient steps
    target=TargetNetCfg(mode="hard", hard_update_period=500),
    # Replay — small buffer keeps data fresh
    buffer=BufferCfg(capacity=10_000),
    # Training loop
    train=TrainCfg(
        total_steps=500_000,
        warmup_steps=10_000,
        batch_size=128,
        train_freq=10,
        eval_interval=50_000,
        eval_episodes=10,
    ),
    # Learning
    lr=2.5e-4,
    gamma=0.99,
    td_loss="mse",
    max_grad_norm=0.5,  # tight clipping — critical for stability
    # Epsilon: linear decay over 50% of training
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=250_000,
    # Standard DQN (no double-Q)
    double_q=False,
)
"""Mnih et al., 2015 — tuned for classic control (matches CleanRL dqn.py).
Discrete Q + MSE loss + hard target copy + epsilon-greedy + uniform replay.
"""

DOUBLE_DQN = DiscreteQCfg(
    network=NetworkCfg(hidden_dim=120, n_layers=2, activation="relu", use_layer_norm=False),
    target=TargetNetCfg(mode="hard", hard_update_period=500),
    buffer=BufferCfg(capacity=10_000),
    train=TrainCfg(
        total_steps=500_000,
        warmup_steps=10_000,
        batch_size=128,
        train_freq=10,
        eval_interval=50_000,
        eval_episodes=10,
    ),
    lr=2.5e-4,
    gamma=0.99,
    td_loss="mse",
    max_grad_norm=0.5,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=250_000,
    # Online net selects best action, target net evaluates it
    double_q=True,
)
"""van Hasselt et al., 2016.
DQN + online-net action selection with target-net evaluation (reduces overestimation).
"""


# ── Continuous actor-critic family ────────────────────────────────────────────

DDPG = ContinuousActorCriticCfg(
    # Separate network configs for actor and critic
    actor_network=NetworkCfg(hidden_dim=256, n_layers=2, activation="relu", use_layer_norm=False),
    critic_network=NetworkCfg(hidden_dim=256, n_layers=2, activation="relu", use_layer_norm=False),
    # Target: Polyak averaging (soft update)
    target=TargetNetCfg(mode="polyak", tau=0.005),
    # Replay
    buffer=BufferCfg(capacity=100_000),
    # Training loop
    train=TrainCfg(
        total_steps=30_000,
        warmup_steps=1_000,
        batch_size=256,
        eval_interval=5_000,
        eval_episodes=10,
    ),
    # Learning
    actor_lr=1e-3,
    critic_lr=1e-3,
    gamma=0.99,
    noise_std=0.1,
    # No TD3/SAC extensions
    twin_q=False,
    actor_delay=1,
    target_noise=0.0,
    target_noise_clip=0.5,
)
"""Lillicrap et al., 2016.
Deterministic policy + single Q-critic + Polyak target + Gaussian exploration noise.
"""

SAC = ContinuousActorCriticCfg(
    # Separate network configs for actor and critic
    actor_network=NetworkCfg(hidden_dim=256, n_layers=2, activation="relu", use_layer_norm=False),
    critic_network=NetworkCfg(hidden_dim=256, n_layers=2, activation="relu", use_layer_norm=False),
    # Target: Polyak averaging (soft update) — critic only, SAC has no actor target
    target=TargetNetCfg(mode="polyak", tau=0.005),
    # Replay
    buffer=BufferCfg(capacity=100_000),
    # Training loop
    train=TrainCfg(
        total_steps=50_000,
        warmup_steps=1_000,
        batch_size=256,
        eval_interval=5_000,
        eval_episodes=10,
    ),
    # Learning
    actor_lr=3e-4,
    critic_lr=3e-4,
    gamma=0.99,
    noise_std=0.0,  # unused — SAC explores via entropy
    # SAC-specific: squashed Gaussian + twin Q + entropy tuning
    actor_type="squashed_gaussian",
    twin_q=True,
    actor_delay=1,
    target_noise=0.0,
    target_noise_clip=0.5,
    # Entropy
    init_alpha=1.0,
    learnable_alpha=True,
    target_entropy=None,  # auto: -action_dim
    alpha_lr=3e-4,
)
"""Haarnoja et al., 2018.
Squashed Gaussian policy + clipped double Q + max-entropy Bellman backup + auto alpha-tuning.
"""

TD3 = ContinuousActorCriticCfg(
    # Separate network configs for actor and critic
    actor_network=NetworkCfg(hidden_dim=256, n_layers=2, activation="relu", use_layer_norm=False),
    critic_network=NetworkCfg(hidden_dim=256, n_layers=2, activation="relu", use_layer_norm=False),
    # Target: Polyak averaging (soft update)
    target=TargetNetCfg(mode="polyak", tau=0.005),
    # Replay
    buffer=BufferCfg(capacity=100_000),
    # Training loop
    train=TrainCfg(
        total_steps=50_000,
        warmup_steps=1_000,
        batch_size=256,
        eval_interval=5_000,
        eval_episodes=10,
    ),
    # Learning
    actor_lr=1e-3,
    critic_lr=1e-3,
    gamma=0.99,
    noise_std=0.1,
    # TD3-specific: deterministic policy + three extensions over DDPG
    actor_type="deterministic",
    twin_q=True,  # clipped double Q
    actor_delay=2,  # delayed actor update
    target_noise=0.2,  # target policy smoothing
    target_noise_clip=0.5,
)
"""Fujimoto et al., 2018.
DDPG + clipped double Q + delayed actor update + target policy smoothing.
"""
