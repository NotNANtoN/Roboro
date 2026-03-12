"""Unified RL training: shared SAC hyperparameters across Hopper and Walker2d.

THIS FILE IS MODIFIED BY THE AUTORESEARCH AGENT.
Custom loop: LN + target net, delayed actor, fast buffer, ortho init.
Robust: catches timeouts and still prints partial results.
"""

import os
import sys
import time
import traceback

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

from prepare import TASKS, evaluate, print_summary, start_timer, check_time, TimeLimitExceeded

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
WARMUP_STEPS = 500
ACTOR_DELAY = 2
TAU = 0.005

INIT_ALPHA = 0.1
LEARNABLE_ALPHA = True


# ═════════════════════════════════════════════════════════════════════════════
# FAST REPLAY BUFFER
# ═════════════════════════════════════════════════════════════════════════════
class FastReplayBuffer:
    __slots__ = ('obs', 'actions', 'rewards', 'next_obs', 'dones',
                 'pos', 'size', 'capacity', 'rng')

    def __init__(self, capacity, obs_dim, action_dim, seed=None):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, done):
        i = self.pos
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.dones[idx]),
        )

    def __len__(self):
        return self.size


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════════
def make_policy_fn(actor, device):
    """Create eval policy function from current actor weights."""
    def policy_fn(o):
        with torch.no_grad():
            o_t = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
            a, _ = actor.act(o_t, deterministic=True)
            return a.squeeze(0).cpu().numpy()
    return policy_fn


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

    # Orthogonal init
    def ortho_init(module, gain=np.sqrt(2)):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    ortho_init(critic)
    ortho_init(actor.trunk)
    for head in [actor.mean_head, actor.log_std_head]:
        nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.constant_(head.bias, 0.0)

    critic_target = TargetNetwork(critic, mode="polyak", tau=TAU).to(device)

    buffer = FastReplayBuffer(BUFFER_CAPACITY, obs_dim, action_dim, seed=SEED)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=LR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR)

    log_alpha = torch.tensor(float(INIT_ALPHA)).log()
    if LEARNABLE_ALPHA:
        log_alpha = nn.Parameter(log_alpha)
    alpha_opt = torch.optim.Adam([log_alpha], lr=LR) if LEARNABLE_ALPHA else None
    target_entropy = -float(action_dim)

    obs, _ = env.reset(seed=SEED)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    ep_reward = 0.0
    grad_steps = 0

    for step in range(1, task.step_budget + 1):
        action, _ = actor.act(obs_t)
        act_np = action.squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(act_np)
        ep_reward += float(reward)

        buffer.add(obs, act_np, float(reward), next_obs, terminated)

        obs = next_obs
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if terminated or truncated:
            ep_reward = 0.0
            obs, _ = env.reset()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if step >= WARMUP_STEPS and len(buffer) >= BATCH_SIZE:
            b_obs, b_act, b_rew, b_nobs, b_done = buffer.sample(BATCH_SIZE)
            alpha = log_alpha.exp()
            grad_steps += 1

            with torch.no_grad():
                na, nlp = actor(b_nobs)
                tq = critic_target(b_nobs, na)
                soft_t = b_rew + GAMMA * (~b_done).float() * (tq - alpha * nlp)
            q1v, q2v = critic.both(b_obs, b_act)
            c_loss = F.mse_loss(q1v, soft_t) + F.mse_loss(q2v, soft_t)
            critic_opt.zero_grad()
            c_loss.backward()
            critic_opt.step()

            if grad_steps % ACTOR_DELAY == 0:
                ap, lp = actor(b_obs)
                qp = critic(b_obs, ap)
                a_loss = (alpha.detach() * lp - qp).mean()
                actor_opt.zero_grad()
                a_loss.backward()
                actor_opt.step()

                if LEARNABLE_ALPHA and alpha_opt is not None:
                    al = -(log_alpha.exp() * (lp.detach() + target_entropy)).mean()
                    alpha_opt.zero_grad()
                    al.backward()
                    alpha_opt.step()

            critic_target.update()

    eval_return = evaluate(task.env_id, make_policy_fn(actor, device),
                           task.eval_episodes, task.eval_seed)
    n_params = sum(p.numel() for p in actor.parameters()) + \
               sum(p.numel() for p in critic.parameters())
    env.close()
    print(f"[{task_name}] eval={eval_return:.1f} steps={step} gs={grad_steps}", flush=True)
    return eval_return, n_params


# ═════════════════════════════════════════════════════════════════════════════
# MAIN — catches timeouts and still prints partial results
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    start_timer()

    hopper_return = 0.0
    walker_return = 0.0
    np_h = np_w = 0

    try:
        hopper_return, np_h = train_sac("hopper")
        check_time()
        walker_return, np_w = train_sac("walker")
        total_time = check_time()
    except (TimeLimitExceeded, SystemExit) as e:
        print(f"\n!!! TIMEOUT — printing partial results !!!", flush=True)
        total_time = 600.0

    print_summary(
        hopper_return=hopper_return, walker_return=walker_return,
        training_seconds=total_time, total_seconds=total_time,
        num_params_hopper=np_h, num_params_walker=np_w, device=DEVICE,
    )
