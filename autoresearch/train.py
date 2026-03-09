"""Unified RL training: shared hyperparameters across CartPole and Pendulum.

THIS FILE IS MODIFIED BY THE AUTORESEARCH AGENT.
Everything is fair game: architecture, optimizer, hyperparameters, training
loop, loss function, exploration strategy, replay buffer, etc.

Goal: maximize `score` — the mean of normalized returns across BOTH envs.
The SAME core hyperparameters (hidden_dim, n_layers, lr, gamma, batch_size,
etc.) must work for both discrete (CartPole) and continuous (Pendulum).

You may import anything from the roboro library (../roboro/).
You may NOT modify prepare.py.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import gymnasium as gym

from roboro.core.seed import set_seed
from roboro.core.config import TrainCfg
from roboro.critics.q import DiscreteQCritic, ContinuousQCritic, TwinQCritic
from roboro.critics.target import TargetNetwork
from roboro.actors.epsilon_greedy import EpsilonGreedyActor
from roboro.actors.squashed_gaussian import SquashedGaussianActor
from roboro.data.replay_buffer import ReplayBuffer
from roboro.updates.dqn import DQNUpdate
from roboro.updates.sac import SACUpdate
from roboro.training.trainer import train_off_policy

from prepare import TASKS, evaluate, print_summary

# ═════════════════════════════════════════════════════════════════════════════
# SHARED HYPERPARAMETERS — these must work for BOTH envs
# ═════════════════════════════════════════════════════════════════════════════
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN_DIM = 128
N_LAYERS = 2
ACTIVATION = "relu"
LR = 3e-4
GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_CAPACITY = 50_000
WARMUP_STEPS = 1_000
TRAIN_FREQ = 1
TAU = 0.005

# DQN-specific (CartPole)
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 25_000
DOUBLE_Q = True
TD_LOSS = "mse"
MAX_GRAD_NORM = 10.0

# SAC-specific (Pendulum)
INIT_ALPHA = 1.0
LEARNABLE_ALPHA = True


# ═════════════════════════════════════════════════════════════════════════════
# TASK 1: CartPole-v1 (discrete Q-learning)
# ═════════════════════════════════════════════════════════════════════════════
def train_cartpole() -> tuple[float, int]:
    """Train DQN on CartPole with shared hyperparams. Returns (eval_return, num_params)."""
    task = TASKS["cartpole"]
    set_seed(SEED)

    env = gym.make(task.env_id)
    obs_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.n)

    q_critic = DiscreteQCritic(
        feature_dim=obs_dim,
        n_actions=n_actions,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        activation=ACTIVATION,
    )
    target_net = TargetNetwork(q_critic, mode="polyak", tau=TAU)
    actor = EpsilonGreedyActor(q_critic, n_actions=n_actions, epsilon=EPSILON_START)

    buffer = ReplayBuffer(
        capacity=BUFFER_CAPACITY, obs_shape=(obs_dim,), action_shape=(), seed=SEED
    )

    update = DQNUpdate(
        q_critic,
        target_net,
        lr=LR,
        gamma=GAMMA,
        td_loss=TD_LOSS,
        max_grad_norm=MAX_GRAD_NORM,
        double_q=DOUBLE_Q,
    )

    _orig = update.update

    def _scheduled(batch, step):
        frac = min(1.0, step / EPSILON_DECAY_STEPS)
        actor.epsilon = EPSILON_START + frac * (EPSILON_END - EPSILON_START)
        return _orig(batch, step)

    update.update = _scheduled  # type: ignore[assignment]

    cfg = TrainCfg(
        total_steps=task.step_budget,
        warmup_steps=WARMUP_STEPS,
        batch_size=BATCH_SIZE,
        train_freq=TRAIN_FREQ,
        eval_interval=task.step_budget + 1,
        device="cpu",
        show_progress=False,
    )
    train_off_policy(env, actor, update, buffer, cfg, device="cpu")

    def policy_fn(obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, _ = actor.act(obs_t, deterministic=True)
            return int(action.squeeze(0).item())

    eval_return = evaluate(task.env_id, policy_fn, task.eval_episodes, task.eval_seed)
    num_params = sum(p.numel() for p in q_critic.parameters())
    env.close()
    return eval_return, num_params


# ═════════════════════════════════════════════════════════════════════════════
# TASK 2: Pendulum-v1 (continuous actor-critic / SAC)
# ═════════════════════════════════════════════════════════════════════════════
def train_pendulum() -> tuple[float, int]:
    """Train SAC on Pendulum with shared hyperparams. Returns (eval_return, num_params)."""
    task = TASKS["pendulum"]
    set_seed(SEED)
    device = torch.device(DEVICE)

    env = gym.make(task.env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = float(np.min(env.action_space.low))
    action_high = float(np.max(env.action_space.high))

    actor = SquashedGaussianActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        activation=ACTIVATION,
    ).to(device)

    q1 = ContinuousQCritic(
        feature_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        activation=ACTIVATION,
    ).to(device)
    q2 = ContinuousQCritic(
        feature_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        activation=ACTIVATION,
    ).to(device)
    critic = TwinQCritic(q1, q2).to(device)

    critic_target = TargetNetwork(critic, mode="polyak", tau=TAU).to(device)

    buffer = ReplayBuffer(
        capacity=BUFFER_CAPACITY,
        obs_shape=(obs_dim,),
        action_shape=(action_dim,),
        seed=SEED,
    )

    update = SACUpdate(
        actor=actor,
        critic=critic,
        critic_target=critic_target,
        actor_lr=LR,
        critic_lr=LR,
        alpha_lr=LR,
        gamma=GAMMA,
        init_alpha=INIT_ALPHA,
        learnable_alpha=LEARNABLE_ALPHA,
    )

    cfg = TrainCfg(
        total_steps=task.step_budget,
        warmup_steps=WARMUP_STEPS,
        batch_size=BATCH_SIZE,
        train_freq=TRAIN_FREQ,
        eval_interval=task.step_budget + 1,
        device=DEVICE,
        show_progress=False,
    )
    train_off_policy(env, actor, update, buffer, cfg, device=device)

    def policy_fn(obs):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _ = actor.act(obs_t, deterministic=True)
            return action.squeeze(0).cpu().numpy()

    eval_return = evaluate(task.env_id, policy_fn, task.eval_episodes, task.eval_seed)
    num_params = sum(p.numel() for p in actor.parameters()) + sum(
        p.numel() for p in critic.parameters()
    )
    env.close()
    return eval_return, num_params


# ═════════════════════════════════════════════════════════════════════════════
# RUN BOTH TASKS
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    start = time.time()

    cartpole_return, num_params_q = train_cartpole()
    pendulum_return, num_params_ac = train_pendulum()

    total_time = time.time() - start

    print_summary(
        cartpole_return=cartpole_return,
        pendulum_return=pendulum_return,
        training_seconds=total_time,
        total_seconds=total_time,
        num_params_q=num_params_q,
        num_params_ac=num_params_ac,
        device=DEVICE,
    )
