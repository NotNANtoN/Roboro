"""Unified RL training: shared SAC hyperparameters across Hopper and Walker2d.

THIS FILE IS MODIFIED BY THE AUTORESEARCH AGENT.
Custom training loop with delayed actor updates for more critic training.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from roboro.core.seed import set_seed
from roboro.critics.q import ContinuousQCritic, TwinQCritic
from roboro.critics.target import TargetNetwork
from roboro.actors.squashed_gaussian import SquashedGaussianActor
from roboro.data.replay_buffer import ReplayBuffer

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
ACTOR_DELAY = 2  # update actor every N critic updates
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

    actor_opt = torch.optim.Adam(actor.parameters(), lr=LR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR)

    log_alpha = torch.tensor(float(INIT_ALPHA)).log()
    if LEARNABLE_ALPHA:
        log_alpha = nn.Parameter(log_alpha)
    alpha_opt = torch.optim.Adam([log_alpha], lr=LR) if LEARNABLE_ALPHA else None
    target_entropy = -float(action_dim)

    obs, _ = env.reset(seed=SEED)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    episode_reward = 0.0
    grad_steps = 0

    for step in range(1, task.step_budget + 1):
        # Act
        action, _ = actor.act(obs_t)
        act_np = action.squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(act_np)

        buffer.add(obs=obs, action=act_np, reward=float(reward),
                   next_obs=next_obs, done=terminated)

        obs = next_obs
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward += float(reward)

        if terminated or truncated:
            episode_reward = 0.0
            obs, _ = env.reset()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        # Learn every step after warmup
        if step >= WARMUP_STEPS and len(buffer) >= BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE).to(device)
            alpha = log_alpha.exp()
            grad_steps += 1

            # Critic update (every step)
            with torch.no_grad():
                na, nlp = actor(batch.next_obs)
                tq = critic_target(batch.next_obs, na)
                soft_t = batch.rewards + GAMMA * (~batch.dones).float() * (tq - alpha * nlp)
            q1v, q2v = critic.both(batch.obs, batch.actions)
            c_loss = F.mse_loss(q1v, soft_t) + F.mse_loss(q2v, soft_t)
            critic_opt.zero_grad()
            c_loss.backward()
            critic_opt.step()

            # Actor + alpha update (delayed)
            if grad_steps % ACTOR_DELAY == 0:
                ap, lp = actor(batch.obs)
                qp = critic(batch.obs, ap)
                a_loss = (alpha.detach() * lp - qp).mean()
                actor_opt.zero_grad()
                a_loss.backward()
                actor_opt.step()

                if LEARNABLE_ALPHA and alpha_opt is not None:
                    al = -(log_alpha.exp() * (lp.detach() + target_entropy)).mean()
                    alpha_opt.zero_grad()
                    al.backward()
                    alpha_opt.step()

            # Target update
            critic_target.update()

    def policy_fn(o):
        with torch.no_grad():
            o_t = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
            a, _ = actor.act(o_t, deterministic=True)
            return a.squeeze(0).cpu().numpy()

    eval_return = evaluate(task.env_id, policy_fn, task.eval_episodes, task.eval_seed)
    n_params = sum(p.numel() for p in actor.parameters()) + \
               sum(p.numel() for p in critic.parameters())
    env.close()
    return eval_return, n_params


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
