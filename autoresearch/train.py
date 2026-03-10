"""Unified RL training: shared SAC hyperparameters across Hopper and Walker2d.

THIS FILE IS MODIFIED BY THE AUTORESEARCH AGENT.
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
from roboro.critics.q import ContinuousQCritic, TwinQCritic
from roboro.critics.target import TargetNetwork
from roboro.actors.squashed_gaussian import SquashedGaussianActor
from roboro.data.replay_buffer import ReplayBuffer
from roboro.updates.sac import SACUpdate
from roboro.training.trainer import train_off_policy

from prepare import TASKS, evaluate, print_summary, start_timer, check_time

# ═════════════════════════════════════════════════════════════════════════════
# SHARED HYPERPARAMETERS
# ═════════════════════════════════════════════════════════════════════════════
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN_DIM = 128
N_LAYERS = 2
ACTIVATION = "relu"
USE_LAYER_NORM = True
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 256
BUFFER_CAPACITY = 100_000
WARMUP_STEPS = 1000
TRAIN_FREQ = 2
TAU = 0.005

INIT_ALPHA = 0.1
LEARNABLE_ALPHA = True


def train_sac(task_name: str) -> tuple[float, int]:
    task = TASKS[task_name]
    set_seed(SEED)
    device = torch.device(DEVICE)

    env = gym.make(task.env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = float(np.min(env.action_space.low))
    action_high = float(np.max(env.action_space.high))

    actor = SquashedGaussianActor(
        obs_dim=obs_dim, action_dim=action_dim,
        action_low=action_low, action_high=action_high,
        hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
        activation=ACTIVATION, use_layer_norm=USE_LAYER_NORM,
    ).to(device)

    q1 = ContinuousQCritic(
        feature_dim=obs_dim, action_dim=action_dim,
        hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
        activation=ACTIVATION, use_layer_norm=USE_LAYER_NORM,
    ).to(device)
    q2 = ContinuousQCritic(
        feature_dim=obs_dim, action_dim=action_dim,
        hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
        activation=ACTIVATION, use_layer_norm=USE_LAYER_NORM,
    ).to(device)
    critic = TwinQCritic(q1, q2).to(device)
    critic_target = TargetNetwork(critic, mode="polyak", tau=TAU).to(device)

    buffer = ReplayBuffer(
        capacity=BUFFER_CAPACITY, obs_shape=(obs_dim,),
        action_shape=(action_dim,), seed=SEED,
    )

    update = SACUpdate(
        actor=actor, critic=critic, critic_target=critic_target,
        actor_lr=LR, critic_lr=LR, alpha_lr=LR,
        gamma=GAMMA, init_alpha=INIT_ALPHA, learnable_alpha=LEARNABLE_ALPHA,
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

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _ = actor.act(obs_t, deterministic=True)
            return action.squeeze(0).cpu().numpy()

    eval_return = evaluate(task.env_id, policy_fn, task.eval_episodes, task.eval_seed)
    num_params = sum(p.numel() for p in actor.parameters()) + \
                 sum(p.numel() for p in critic.parameters())
    env.close()
    return eval_return, num_params


if __name__ == "__main__":
    start_timer()

    hopper_return, np_h = train_sac("hopper")
    check_time()
    walker_return, np_w = train_sac("walker")
    total_time = check_time()

    print_summary(
        hopper_return=hopper_return, walker_return=walker_return,
        training_seconds=total_time, total_seconds=total_time,
        num_params_hopper=np_h, num_params_walker=np_w, device=DEVICE,
    )
