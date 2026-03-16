"""Unified RL training: shared SAC hyperparameters across cheetah-run and humanoid-walk.

THIS FILE IS MODIFIED BY THE AUTORESEARCH AGENT.
Starting config: best recipe from the MuJoCo campaign (LN + ortho init + fast
buffer + delayed actor). HIDDEN_DIM=256 to handle humanoid's 67-dim obs / 21-dim act.

v2 improvement: per-step metric logging to runs/ for post-run diagnostics.
"""

import csv
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

from prepare_dmc import TASKS, evaluate, print_summary, start_timer, check_time, TimeLimitExceeded, make_dmc_env

# ═════════════════════════════════════════════════════════════════════════════
# SHARED HYPERPARAMETERS — must work for BOTH cheetah-run and humanoid-walk
# ═════════════════════════════════════════════════════════════════════════════
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN_DIM = 256
N_LAYERS = 2
ACTIVATION = "relu"
USE_LAYER_NORM = True
LR = 5e-3
GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_CAPACITY = 100_000
WARMUP_STEPS = 512
ACTOR_DELAY = 2
TAU = 0.005
TRAIN_FREQ = 1
UTD = 1

INIT_ALPHA = 0.1
LEARNABLE_ALPHA = True
TARGET_ENTROPY_SCALE = 0.5
ALPHA_MIN = 0.01
ACTION_REPEAT = 2


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
# METRICS LOGGER — v2: per-step diagnostics dumped to CSV
# ═════════════════════════════════════════════════════════════════════════════
class MetricsLogger:
    """Collects per-step metrics and writes them to CSV after training."""

    def __init__(self):
        self.rows: list[dict] = []

    def log(self, step, **kwargs):
        self.rows.append({"step": step, **kwargs})

    def dump(self, path):
        if not self.rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        keys = self.rows[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════════
def make_policy_fn(actor, device):
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

    env = make_dmc_env(task.env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = float(np.min(env.action_space.low))
    action_high = float(np.max(env.action_space.high))

    print(f"[{task_name}] obs_dim={obs_dim} action_dim={action_dim} "
          f"action_range=[{action_low}, {action_high}]", flush=True)

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

    # Orthogonal init — biggest single improvement from MuJoCo campaign
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
    target_entropy = -float(action_dim) * TARGET_ENTROPY_SCALE

    metrics = MetricsLogger()

    obs, _ = env.reset(seed=SEED)
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    ep_reward = 0.0
    ep_count = 0
    grad_steps = 0
    decisions = 0
    env_steps = 0
    last_metrics = {}

    while env_steps < task.step_budget:
        action, _ = actor.act(obs_t)
        act_np = action.squeeze(0).cpu().numpy()
        decision_obs = obs

        repeat_reward = 0.0
        done = False
        for _ in range(ACTION_REPEAT):
            next_obs, reward, terminated, truncated, _ = env.step(act_np)
            repeat_reward += float(reward)
            ep_reward += float(reward)
            env_steps += 1
            done = terminated or truncated
            if done or env_steps >= task.step_budget:
                break

        buffer.add(decision_obs, act_np, repeat_reward, next_obs, terminated)
        decisions += 1

        obs = next_obs
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if done:
            ep_count += 1
            ep_reward = 0.0
            obs, _ = env.reset()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        warmup_decisions = WARMUP_STEPS // ACTION_REPEAT
        if decisions >= warmup_decisions and len(buffer) >= BATCH_SIZE and decisions % TRAIN_FREQ == 0:
          for _utd in range(UTD):
            b_obs, b_act, b_rew, b_nobs, b_done = buffer.sample(BATCH_SIZE)
            alpha = log_alpha.exp()
            grad_steps += 1

            gamma_eff = GAMMA ** ACTION_REPEAT
            with torch.no_grad():
                na, nlp = actor(b_nobs)
                tq = critic_target(b_nobs, na)
                soft_t = b_rew + gamma_eff * (~b_done).float() * (tq - alpha * nlp)
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
                    with torch.no_grad():
                        log_alpha.clamp_(min=np.log(ALPHA_MIN))

                last_metrics = {
                    "critic_loss": c_loss.item(),
                    "actor_loss": a_loss.item(),
                    "alpha": alpha.item(),
                    "q_mean": q1v.mean().item(),
                    "target_q_mean": soft_t.mean().item(),
                    "log_prob": lp.mean().item(),
                }

            critic_target.update()

            if grad_steps % 1000 == 0 and last_metrics:
                metrics.log(env_steps, grad_steps=grad_steps, ep_count=ep_count, **last_metrics)

        if env_steps % 20000 < ACTION_REPEAT and env_steps >= 20000:
            alpha_val = log_alpha.exp().item()
            q_str = f"q={last_metrics.get('q_mean', 0):.1f}" if last_metrics else "q=n/a"
            cl_str = f"cl={last_metrics.get('critic_loss', 0):.3f}" if last_metrics else "cl=n/a"
            print(f"[{task_name}] step={env_steps} gs={grad_steps} eps={ep_count} "
                  f"alpha={alpha_val:.4f} {q_str} {cl_str} "
                  f"elapsed={check_time():.0f}s", flush=True)
    step = env_steps

    # Dump metrics CSV
    metrics.dump(f"runs/{task_name}_metrics.csv")

    eval_return = evaluate(task.env_id, make_policy_fn(actor, device),
                           task.eval_episodes, task.eval_seed)
    n_params = sum(p.numel() for p in actor.parameters()) + \
               sum(p.numel() for p in critic.parameters())
    env.close()
    print(f"[{task_name}] eval={eval_return:.1f} steps={step} gs={grad_steps} "
          f"params={n_params}", flush=True)
    return eval_return, n_params


# ═════════════════════════════════════════════════════════════════════════════
# MAIN — catches timeouts and still prints partial results
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    start_timer()

    cheetah_return = 0.0
    humanoid_return = 0.0
    np_c = np_h = 0

    try:
        cheetah_return, np_c = train_sac("cheetah")
        check_time()
        humanoid_return, np_h = train_sac("humanoid")
        total_time = check_time()
    except (TimeLimitExceeded, SystemExit) as e:
        print(f"\n!!! TIMEOUT — printing partial results !!!", flush=True)
        total_time = 600.0

    print_summary(
        cheetah_return=cheetah_return, humanoid_return=humanoid_return,
        training_seconds=total_time, total_seconds=total_time,
        num_params_cheetah=np_c, num_params_humanoid=np_h, device=DEVICE,
    )
