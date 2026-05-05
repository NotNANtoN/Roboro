"""Unified RL training: shared SAC hyperparameters across cheetah-run and humanoid-walk.

THIS FILE IS MODIFIED BY THE AUTORESEARCH AGENT.
Starting config: best recipe from the MuJoCo campaign (LN + ortho init + fast
buffer + delayed actor). HIDDEN_DIM=256 to handle humanoid's 67-dim obs / 21-dim act.

v2 improvement: per-step metric logging to runs/ for post-run diagnostics.
"""

import copy
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
from roboro.nn.blocks import MLPBlock

from prepare_dmc import TASKS, TaskSpec, evaluate, print_summary, start_timer, check_time, TimeLimitExceeded, make_dmc_env

# ═════════════════════════════════════════════════════════════════════════════
# SHARED HYPERPARAMETERS — must work for BOTH cheetah-run and humanoid-walk
# ═════════════════════════════════════════════════════════════════════════════
SEED = int(os.environ.get("SEED", 42))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN_DIM = 256
N_LAYERS = 2
ACTIVATION = "relu"
USE_LAYER_NORM = True
LR = 5e-3
GAMMA = 0.995
MAX_GRAD_NORM = 1.0
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

# Delightful Policy Gradient (DG) — gates actor loss by σ(A × surprisal).
# Two-sample advantage: A = Q(s, a1) − Q(s, a2) with a1,a2 ~ π(·|s).
# Whitening is essential here: raw Q scales are O(100), so without it the
# sigmoid saturates and the gate collapses to 1 everywhere (DG becomes a no-op).
# DG disabled by default: 3-seed cheetah sweep showed −20% return vs baseline.
# Kept as env-var toggle (DG=1) for future experimentation.
DG_ENABLED = bool(int(os.environ.get("DG", "0")))
DG_ETA = float(os.environ.get("DG_ETA", "1.0"))
DG_CLIP_SURPRISAL = 10.0
DG_WHITEN = bool(int(os.environ.get("DG_WHITEN", "1")))
DG_GATE_Q_ONLY = bool(int(os.environ.get("DG_GATE_Q_ONLY", "0")))  # Only gate -Q, not entropy
CHEETAH_ONLY = bool(int(os.environ.get("CHEETAH_ONLY", "0")))

# DQV-SAC: add a soft V network with Q bootstrapping on V.
# Two variants controlled by DQV_BC:
#
#   DQV=1, DQV_BC=0 (pure DQV, off-policy-biased — collapsed in 3-seed test):
#     V target: r - α·log π(a_replay|s) + γ·V_tgt(s')   [TD on V]
#     Q target: r + γ·V_tgt(s')
#
#   DQV=1, DQV_BC=1 (bias-corrected, per Daley et al. BC-QVMAX):
#     V target: min(Q1,Q2)(s, a~π) - α·log π(a~π|s)     [regression, no TD]
#     Q target: r + γ·V_tgt(s')
#
# BC restores twin-Q pessimism (via min on Q for V's target) and removes
# off-policy bias (V uses fresh policy samples, not replay actions).
#
# DQV_TWIN=1 adds a second V network: Q bootstraps on min(V1_tgt, V2_tgt).
# This restores pessimism in the Q bootstrap, which BC-DQV was missing.
DQV_ENABLED = bool(int(os.environ.get("DQV", "0")))
DQV_BC = bool(int(os.environ.get("DQV_BC", "0")))
DQV_TWIN = bool(int(os.environ.get("DQV_TWIN", "0")))

# SAC-v1 style: add V as a side network for advantage estimates, but keep
# standard SAC Q-learning (Q does NOT bootstrap on V). V just regresses to
# soft-V = min(Q1,Q2)(s, a~π) - α·log π(a~π|s). Useful for DG experiments
# where we want Q - V as a clean advantage signal.
SACV1_ENABLED = bool(int(os.environ.get("SACV1", "0")))

# Dueling SAC: Q(s,a) = V(s) + A(s,a) with shared trunk (RDQ-style).
# Unlike DQV, there's no bootstrap chain — V and A are learned jointly.
# L2 regularization on V and A for identifiability (can't shift constants).
# Twin architecture: two (V,A) pairs, take min(Q1,Q2) for pessimism.
DUELING_ENABLED = bool(int(os.environ.get("DUELING", "0")))
DUELING_BETA = float(os.environ.get("DUELING_BETA", "0.01"))  # L2 reg weight

# SAT-SAC: State-Adaptive Temperature. Learn α(s) instead of scalar α.
# Idea: explore more in hard states (low V), exploit in easy states (high V).
# α_net: small MLP(obs_dim → 1) with softplus output to ensure α > 0.
SAT_ENABLED = bool(int(os.environ.get("SAT", "0")))
SAT_LR = float(os.environ.get("SAT_LR", "0.0001"))  # Lower LR for stability (default 1e-4 vs 3e-4)

# RSAT-SAC: Residual State-Adaptive Temperature.
# α(s) = α_scalar · (1 + ε·tanh(f(s)))  — bounded deviation from scalar base.
# Fixes SAT's high variance by anchoring to proven scalar α.
RSAT_ENABLED = bool(int(os.environ.get("RSAT", "0")))
RSAT_EPS = float(os.environ.get("RSAT_EPS", "0.5"))  # Max ±50% deviation from scalar
RSAT_LR = float(os.environ.get("RSAT_LR", "0.0001"))  # LR for residual network

# UCB-SAC: Uncertainty-Based Curiosity temperature (Q-disagreement variant).
# α(s,a) = α_base · (1 + β · |Q1-Q2| / (|Q_mean| + ε))
UCB_ENABLED = bool(int(os.environ.get("UCB", "0")))
UCB_BETA = float(os.environ.get("UCB_BETA", "1.0"))

# TDE-SAC: TD-Error driven adaptive temperature.
# α(s) = α_base · (1 + β · |TD_error(s)| / (|TD_mean| + ε))
# Uses replay TD error magnitude as a per-state uncertainty signal.
TDE_ENABLED = bool(int(os.environ.get("TDE", "0")))
TDE_BETA = float(os.environ.get("TDE_BETA", "2.0"))  # TD errors are small relative to Q, so higher β

# SPG-SAC: Sampled Policy Gradient actor update (Wiehe et al., 2018).
# Replaces reparameterization-based actor gradient with sample-and-evaluate:
#   1. Start with best = actor_mean(s)
#   2. If Q(s, a_replay) > Q(s, best): best = a_replay
#   3. Sample S actions around best with Gaussian noise, keep highest Q
#   4. If Q(s, best) > Q(s, actor_mean(s)): regress actor toward best
# Searches action space more globally than DPG-style backprop through Q.
SPG_ENABLED = bool(int(os.environ.get("SPG", "0")))
SPG_SAMPLES = int(os.environ.get("SPG_SAMPLES", "8"))   # Offline Gaussian samples per state
SPG_NOISE = float(os.environ.get("SPG_NOISE", "0.3"))   # Std of Gaussian noise for sampling

# Ensemble critics: N>2 Q-networks, use min over all for targets and actor.
# First 2 networks always get same init as standard twin-Q (same seed).
N_CRITICS = int(os.environ.get("N_CRITICS", "2"))


# ═════════════════════════════════════════════════════════════════════════════
# FAST REPLAY BUFFER
# ═════════════════════════════════════════════════════════════════════════════
class FastReplayBuffer:
    __slots__ = ('obs', 'actions', 'rewards', 'next_obs', 'dones',
                 'best_actions', 'pos', 'size', 'capacity', 'rng')

    def __init__(self, capacity, obs_dim, action_dim, seed=None):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.best_actions = np.zeros((capacity, action_dim), dtype=np.float32)
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
        self.best_actions[i] = action  # SBA: init best to original action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, return_indices=False):
        idx = self.rng.integers(0, self.size, size=batch_size)
        batch = (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.dones[idx]),
        )
        if return_indices:
            return (*batch, torch.from_numpy(self.best_actions[idx]), idx)
        return batch

    def update_best_actions(self, indices, new_best):
        """SBA: update best-known actions (separate from original actions)."""
        self.best_actions[indices] = new_best

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
# DUELING CRITIC (RDQ-style for continuous actions)
# ═════════════════════════════════════════════════════════════════════════════
class DuelingCritic(nn.Module):
    """Q(s,a) = V(s) + A(s,a) with shared trunk. RDQ-style L2 reg for identifiability."""

    def __init__(self, obs_dim, action_dim, hidden_dim, n_layers, activation, use_layer_norm):
        super().__init__()
        self.trunk = MLPBlock(
            in_dim=obs_dim, out_dim=hidden_dim, hidden_dim=hidden_dim,
            n_layers=n_layers - 1, activation=activation,
            output_activation=activation, use_layer_norm=use_layer_norm,
        )
        self.v_head = nn.Linear(hidden_dim, 1)
        self.a_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        feat = self.trunk(obs)
        v = self.v_head(feat)
        a = self.a_head(torch.cat([feat, action], dim=-1))
        return (v + a).squeeze(-1)

    def v_and_a(self, obs, action):
        """Return V(s), A(s,a), Q(s,a) for L2 regularization."""
        feat = self.trunk(obs)
        v = self.v_head(feat).squeeze(-1)
        a = self.a_head(torch.cat([feat, action], dim=-1)).squeeze(-1)
        return v, a, v + a


class TwinDuelingCritic(nn.Module):
    """Twin dueling critics: min(Q1, Q2) for pessimism."""

    def __init__(self, obs_dim, action_dim, hidden_dim, n_layers, activation, use_layer_norm):
        super().__init__()
        self.d1 = DuelingCritic(obs_dim, action_dim, hidden_dim, n_layers, activation, use_layer_norm)
        self.d2 = DuelingCritic(obs_dim, action_dim, hidden_dim, n_layers, activation, use_layer_norm)

    def forward(self, obs, action):
        return torch.min(self.d1(obs, action), self.d2(obs, action))

    def both(self, obs, action):
        return self.d1(obs, action), self.d2(obs, action)

    def both_with_va(self, obs, action):
        """Return (v1, a1, q1), (v2, a2, q2) for L2 reg."""
        return self.d1.v_and_a(obs, action), self.d2.v_and_a(obs, action)


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
    steps_override = int(os.environ.get("STEPS", "0"))
    if steps_override > 0:
        task = TaskSpec(
            env_id=task.env_id, step_budget=steps_override,
            eval_episodes=task.eval_episodes, eval_seed=task.eval_seed,
            max_return=task.max_return, min_return=task.min_return,
            description=task.description,
        )
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

    if DUELING_ENABLED:
        critic = TwinDuelingCritic(
            obs_dim=obs_dim, action_dim=action_dim,
            hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
            activation=ACTIVATION, use_layer_norm=USE_LAYER_NORM,
        ).to(device)
    else:
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
        if N_CRITICS > 2:
            # Save RNG state so extra critics don't affect q1/q2 downstream init
            rng_state = torch.get_rng_state()
            extra_qs = []
            for i in range(N_CRITICS - 2):
                torch.manual_seed(SEED + 1000 + i)
                eq = ContinuousQCritic(
                    feature_dim=obs_dim, action_dim=action_dim,
                    hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
                    activation=ACTIVATION, use_layer_norm=USE_LAYER_NORM,
                ).to(device)
                extra_qs.append(eq)
            torch.set_rng_state(rng_state)  # Restore so ortho_init matches N=2

            class EnsembleCritic(nn.Module):
                def __init__(self, q_list):
                    super().__init__()
                    self.qs = nn.ModuleList(q_list)
                def forward(self, obs, action):
                    return torch.stack([q(obs, action) for q in self.qs]).min(dim=0).values
                def both(self, obs, action):
                    vals = [q(obs, action) for q in self.qs]
                    return vals[0], vals[1]  # For compatibility
                def all_q(self, obs, action):
                    return torch.stack([q(obs, action) for q in self.qs])  # (N, B)

            critic = EnsembleCritic([q1, q2] + extra_qs).to(device)
            print(f"[ensemble] Using {N_CRITICS} critics")
        else:
            critic = TwinQCritic(q1, q2).to(device)

    # V network(s) for DQV-SAC or SAC-v1 (unused when both disabled).
    # DQV_TWIN adds a second V for pessimism: Q bootstraps on min(V1_tgt, V2_tgt).
    v_critic = MLPBlock(
        in_dim=obs_dim, out_dim=1, hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS, activation=ACTIVATION, use_layer_norm=USE_LAYER_NORM,
    ).to(device) if (DQV_ENABLED or SACV1_ENABLED) else None
    v_critic2 = MLPBlock(
        in_dim=obs_dim, out_dim=1, hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS, activation=ACTIVATION, use_layer_norm=USE_LAYER_NORM,
    ).to(device) if (DQV_ENABLED and DQV_TWIN) else None

    # Orthogonal init — biggest single improvement from MuJoCo campaign
    def ortho_init(module, gain=np.sqrt(2)):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    if N_CRITICS > 2 and not DUELING_ENABLED:
        # Init q1/q2 first (same as twin-Q), then extras separately
        ortho_init(critic.qs[0])
        ortho_init(critic.qs[1])
        rng_state2 = torch.get_rng_state()
        for i, eq in enumerate(critic.qs[2:]):
            torch.manual_seed(SEED + 2000 + i)
            ortho_init(eq)
        torch.set_rng_state(rng_state2)
    else:
        ortho_init(critic)
    ortho_init(actor.trunk)
    for head in [actor.mean_head, actor.log_std_head]:
        nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.constant_(head.bias, 0.0)
    if v_critic is not None:
        ortho_init(v_critic)
    if v_critic2 is not None:
        ortho_init(v_critic2)

    critic_target = TargetNetwork(critic, mode="polyak", tau=TAU).to(device)
    v_target = TargetNetwork(v_critic, mode="polyak", tau=TAU).to(device) if v_critic is not None else None
    v_target2 = TargetNetwork(v_critic2, mode="polyak", tau=TAU).to(device) if v_critic2 is not None else None

    buffer = FastReplayBuffer(BUFFER_CAPACITY, obs_dim, action_dim, seed=SEED)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=LR)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR)
    v_opt = torch.optim.Adam(v_critic.parameters(), lr=LR) if v_critic is not None else None
    v_opt2 = torch.optim.Adam(v_critic2.parameters(), lr=LR) if v_critic2 is not None else None

    # Alpha (temperature) — scalar, SAT (unbounded), or RSAT (residual-bounded)
    if RSAT_ENABLED:
        # Residual SAT: α(s) = α_scalar · (1 + ε·tanh(f(s)))
        # α_scalar is the standard learnable dual variable (stable anchor)
        # f(s) learns bounded state-dependent corrections
        log_alpha = nn.Parameter(torch.tensor(float(INIT_ALPHA)).log())
        alpha_opt = torch.optim.Adam([log_alpha], lr=LR)
        rsat_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)
        nn.init.orthogonal_(rsat_net[0].weight, gain=0.1)
        nn.init.zeros_(rsat_net[0].bias)
        nn.init.orthogonal_(rsat_net[2].weight, gain=0.01)
        nn.init.zeros_(rsat_net[2].bias)  # Start at tanh(0)=0, so α(s)=α_scalar initially
        rsat_opt = torch.optim.Adam(rsat_net.parameters(), lr=RSAT_LR)
        alpha_net = None  # unused
    elif SAT_ENABLED:
        # Original SAT: α(s) = softplus(alpha_net(s)) — unbounded
        alpha_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)
        nn.init.constant_(alpha_net[-1].bias, float(np.log(INIT_ALPHA)))
        alpha_opt = torch.optim.Adam(alpha_net.parameters(), lr=SAT_LR)
        log_alpha = None
        rsat_net = None
        rsat_opt = None
    else:
        alpha_net = None
        rsat_net = None
        log_alpha = torch.tensor(float(INIT_ALPHA)).log()
        if LEARNABLE_ALPHA:
            log_alpha = nn.Parameter(log_alpha)
        alpha_opt = torch.optim.Adam([log_alpha], lr=LR) if LEARNABLE_ALPHA else None
        rsat_opt = None
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
            if SPG_ENABLED:
                b_obs, b_act, b_rew, b_nobs, b_done, b_best, b_idx = buffer.sample(BATCH_SIZE, return_indices=True)
            else:
                b_obs, b_act, b_rew, b_nobs, b_done = buffer.sample(BATCH_SIZE)
            if RSAT_ENABLED:
                # Residual SAT: α(s) = α_scalar · (1 + ε·tanh(f(s)))
                alpha_scalar = log_alpha.exp()
                alpha = alpha_scalar * (1.0 + RSAT_EPS * torch.tanh(rsat_net(b_obs)).squeeze(-1))
                with torch.no_grad():
                    alpha_next = alpha_scalar * (1.0 + RSAT_EPS * torch.tanh(rsat_net(b_nobs)).squeeze(-1))
            elif SAT_ENABLED:
                # Original SAT: α(s) = softplus(alpha_net(s)) — no clamp, let it learn freely
                alpha = F.softplus(alpha_net(b_obs)).squeeze(-1)
                with torch.no_grad():
                    alpha_next = F.softplus(alpha_net(b_nobs)).squeeze(-1)
            elif UCB_ENABLED:
                # UCB: scalar base alpha for critic targets, modulated for actor later
                alpha = log_alpha.exp()
                alpha_next = alpha
            else:
                alpha = log_alpha.exp()
                alpha_next = alpha
            grad_steps += 1

            gamma_eff = GAMMA ** ACTION_REPEAT
            not_done = (~b_done).float()

            if DUELING_ENABLED:
                # Dueling SAC: Q(s,a) = V(s) + A(s,a) with L2 reg on V and A
                # Standard SAC target, but critic is TwinDuelingCritic
                with torch.no_grad():
                    na, nlp = actor(b_nobs)
                    tq = critic_target(b_nobs, na)
                    # Use alpha_next for next-state entropy (SAT uses state-dependent α)
                    soft_t = b_rew + gamma_eff * not_done * (tq - alpha_next * nlp)

                # Get V, A, Q from dueling critics
                (v1, a1, q1v), (v2, a2, q2v) = critic.both_with_va(b_obs, b_act)

                # TD loss + L2 regularization (RDQ-style)
                td_loss = F.mse_loss(q1v, soft_t) + F.mse_loss(q2v, soft_t)
                l2_reg = (DUELING_BETA / 2) * (v1.pow(2).mean() + a1.pow(2).mean() +
                                                v2.pow(2).mean() + a2.pow(2).mean())
                c_loss = td_loss + l2_reg

                critic_opt.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
                critic_opt.step()

                with torch.no_grad():
                    td_err = ((q1v - soft_t).abs() + (q2v - soft_t).abs()) / 2.0

                v_loss = torch.tensor(0.0)  # no separate V network

            elif DQV_ENABLED:
                # DQV: Q bootstraps on V
                #   Q target: r + γ·V_tgt(s')  [or min(V1_tgt, V2_tgt) if DQV_TWIN]
                with torch.no_grad():
                    v1_next = v_target(b_nobs).squeeze(-1)
                    if DQV_TWIN:
                        v2_next = v_target2(b_nobs).squeeze(-1)
                        v_next = torch.min(v1_next, v2_next)
                    else:
                        v_next = v1_next
                    q_t = b_rew + gamma_eff * not_done * v_next

                q1v, q2v = critic.both(b_obs, b_act)
                c_loss = F.mse_loss(q1v, q_t) + F.mse_loss(q2v, q_t)
                critic_opt.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
                critic_opt.step()

                with torch.no_grad():
                    td_err = ((q1v - q_t).abs() + (q2v - q_t).abs()) / 2.0

                # V target depends on BC mode
                if DQV_BC:
                    # BC-DQV: V regresses to soft-V implied by current Q (no TD, no off-policy bias)
                    #   V_target = min(Q1,Q2)(s, a~π) - α·log π(a~π|s)
                    with torch.no_grad():
                        a_fresh, lp_fresh = actor(b_obs)
                        q1_fresh, q2_fresh = critic.both(b_obs, a_fresh)
                        q_min_fresh = torch.min(q1_fresh, q2_fresh)
                        v_t = q_min_fresh - alpha * lp_fresh
                else:
                    # Pure DQV (off-policy-biased): V bootstraps via TD
                    #   V_target = r - α·log π(a_replay|s) + γ·V_tgt(s')
                    with torch.no_grad():
                        logp_replay = actor.evaluate_log_prob(b_obs, b_act)
                        v_t = b_rew - alpha * logp_replay + gamma_eff * not_done * v_next

                # Update V1 (and V2 if twin)
                v_pred = v_critic(b_obs).squeeze(-1)
                v_loss = F.mse_loss(v_pred, v_t)
                v_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(v_critic.parameters(), MAX_GRAD_NORM)
                v_opt.step()

                if DQV_TWIN:
                    v_pred2 = v_critic2(b_obs).squeeze(-1)
                    v_loss2 = F.mse_loss(v_pred2, v_t)
                    v_opt2.zero_grad()
                    v_loss2.backward()
                    nn.utils.clip_grad_norm_(v_critic2.parameters(), MAX_GRAD_NORM)
                    v_opt2.step()

                soft_t = q_t  # alias for logging

            elif SACV1_ENABLED:
                # SAC-v1: standard SAC Q-learning + V as side network for advantage
                #   Q target: r + γ·(min(Q1_tgt, Q2_tgt)(s', a'~π) - α·log π(a'|s'))
                #   V target: min(Q1,Q2)(s, a~π) - α·log π(a~π|s)  [regression, no TD]
                with torch.no_grad():
                    na, nlp = actor(b_nobs)
                    tq = critic_target(b_nobs, na)
                    soft_t = b_rew + gamma_eff * not_done * (tq - alpha_next * nlp)

                q1v, q2v = critic.both(b_obs, b_act)
                c_loss = F.mse_loss(q1v, soft_t) + F.mse_loss(q2v, soft_t)
                critic_opt.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
                critic_opt.step()

                with torch.no_grad():
                    td_err = ((q1v - soft_t).abs() + (q2v - soft_t).abs()) / 2.0

                # V regresses to soft-V (same as BC-DQV's V target)
                with torch.no_grad():
                    a_fresh, lp_fresh = actor(b_obs)
                    q1_fresh, q2_fresh = critic.both(b_obs, a_fresh)
                    q_min_fresh = torch.min(q1_fresh, q2_fresh)
                    v_t = q_min_fresh - alpha * lp_fresh

                v_pred = v_critic(b_obs).squeeze(-1)
                v_loss = F.mse_loss(v_pred, v_t)
                v_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(v_critic.parameters(), MAX_GRAD_NORM)
                v_opt.step()

            else:
                # Standard SAC (no V network)
                with torch.no_grad():
                    na, nlp = actor(b_nobs)
                    tq = critic_target(b_nobs, na)
                    soft_t = b_rew + gamma_eff * not_done * (tq - alpha_next * nlp)
                if N_CRITICS > 2:
                    all_qv = critic.all_q(b_obs, b_act)  # (N, B)
                    c_loss = sum(F.mse_loss(all_qv[i], soft_t) for i in range(N_CRITICS))
                    q1v, q2v = all_qv[0], all_qv[1]
                else:
                    q1v, q2v = critic.both(b_obs, b_act)
                    c_loss = F.mse_loss(q1v, soft_t) + F.mse_loss(q2v, soft_t)
                critic_opt.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
                critic_opt.step()
                with torch.no_grad():
                    td_err = ((q1v - soft_t).abs() + (q2v - soft_t).abs()) / 2.0
                v_loss = torch.tensor(0.0)

            spg_diag = {}
            if grad_steps % ACTOR_DELAY == 0:
                ap, lp = actor(b_obs)
                qp = critic(b_obs, ap)

                if SPG_ENABLED:
                    # SPG: sample-and-evaluate actor update (Wiehe et al., 2018)
                    # Batched: all S samples evaluated in one critic forward pass
                    B = b_obs.shape[0]
                    S = SPG_SAMPLES
                    with torch.no_grad():
                        _, pre_std = actor._get_distribution(b_obs)
                        # Use pessimistic Q (min of twin) throughout to avoid exploitation
                        q1_a, q2_a = critic.both(b_obs, ap)
                        q_actor = torch.min(q1_a, q2_a)

                        # SBA: start from stored best action
                        best_actions = b_best.clone()
                        q1_b, q2_b = critic.both(b_obs, best_actions)
                        best_q = torch.min(q1_b, q2_b)

                        # Check if current actor is better than stored best
                        actor_better = q_actor > best_q
                        best_actions[actor_better] = ap.detach()[actor_better]
                        best_q[actor_better] = q_actor[actor_better]

                        # Batched Gaussian exploration: S samples per state in one critic pass
                        A = ap.shape[-1]
                        noise = torch.randn(B, S, A, device=b_obs.device) * pre_std.unsqueeze(1).clamp(min=0.01)
                        all_samples = (best_actions.unsqueeze(1) + noise).clamp(-1.0, 1.0)  # (B, S, A)

                        # Single batched critic forward pass
                        obs_exp = b_obs.unsqueeze(1).expand(-1, S, -1).reshape(B * S, -1)
                        act_flat = all_samples.reshape(B * S, -1)
                        q_all = critic(obs_exp, act_flat).reshape(B, S)  # (B, S)

                        # Best sample per state
                        best_sample_idx = q_all.argmax(dim=1)
                        q_best_sample = q_all.gather(1, best_sample_idx.unsqueeze(1)).squeeze(1)
                        best_sample_actions = all_samples[torch.arange(B), best_sample_idx]

                        # Use pessimistic Q (min of twin critics) to avoid Q-exploitation
                        q1_best, q2_best = critic.both(b_obs, best_sample_actions)
                        q_best_pessimistic = torch.min(q1_best, q2_best)

                        sample_better = q_best_pessimistic > best_q
                        best_actions[sample_better] = best_sample_actions[sample_better]
                        best_q[sample_better] = q_best_pessimistic[sample_better]

                        # SBA: store best found action back
                        buffer.update_best_actions(b_idx, best_actions.cpu().numpy())

                        improved = best_q > q_actor
                        spg_frac = improved.float().mean().item()

                    # --- SPG diagnostics: compare to reparameterized gradient ---
                    with torch.no_grad():
                        spg_dir = best_actions - ap.detach()  # (B, A) SPG direction
                        # Source tracking: where did the winning action come from?
                        from_sba = (~actor_better & ~sample_better).float().mean().item()
                        from_actor = (actor_better & ~improved).float().mean().item()
                        from_sample = sample_better.float().mean().item()

                    # Compute reparameterized gradient direction for comparison
                    ap_for_grad = ap.detach().requires_grad_(True)
                    q_for_grad = critic(b_obs, ap_for_grad)
                    q_for_grad.sum().backward()
                    reparam_dir = ap_for_grad.grad.detach()  # (B, A) direction reparam would go

                    # Cosine similarity between SPG and reparam directions
                    with torch.no_grad():
                        cos_sim = F.cosine_similarity(spg_dir, reparam_dir, dim=-1)  # (B,)
                        # Q at reparam step (normalized step in gradient direction)
                        reparam_step = ap.detach() + reparam_dir / (reparam_dir.norm(dim=-1, keepdim=True) + 1e-8) * spg_dir.norm(dim=-1, keepdim=True)
                        reparam_step = reparam_step.clamp(-1.0, 1.0)
                        q1_rep, q2_rep = critic.both(b_obs, reparam_step)
                        q_reparam = torch.min(q1_rep, q2_rep)

                    spg_diag = {
                        "cos_sim_mean": cos_sim.mean().item(),
                        "cos_sim_improved": cos_sim[improved].mean().item() if improved.any() else 0.0,
                        "spg_q_gain": (best_q - q_actor).mean().item(),
                        "reparam_q_gain": (q_reparam - q_actor).mean().item(),
                        "from_sba": from_sba,
                        "from_actor": from_actor,
                        "from_sample": from_sample,
                        "spg_frac": spg_frac,
                        "spg_step_norm": spg_dir.norm(dim=-1).mean().item(),
                    }

                    # Regress ap (reparameterized, with grad) toward best_actions
                    if improved.any():
                        a_loss_spg = F.mse_loss(ap[improved], best_actions[improved])
                    else:
                        a_loss_spg = torch.tensor(0.0, device=b_obs.device)
                    a_loss = a_loss_spg + (alpha.detach() * lp).mean()
                    dg_gate_mean = spg_frac
                    dg_gate_std = 0.0
                    dg_delight_mean = (best_q - q_actor).mean().item()
                    alpha_actor = alpha.detach()

                else:
                    # Standard reparameterized actor update (DDPG-style)
                    # State-dependent alpha modulation for actor
                    if TDE_ENABLED:
                        with torch.no_grad():
                            td_scale = td_err.mean() + 1e-6
                            tde_multiplier = 1.0 + TDE_BETA * td_err / td_scale
                        alpha_actor = alpha.detach() * tde_multiplier
                    elif UCB_ENABLED:
                        with torch.no_grad():
                            q1_fresh, q2_fresh = critic.both(b_obs, ap)
                            q_disagree = (q1_fresh - q2_fresh).abs()
                            q_scale = (q1_fresh + q2_fresh).abs().mean() / 2.0 + 1e-6
                            ucb_multiplier = 1.0 + UCB_BETA * q_disagree / q_scale
                        alpha_actor = alpha.detach() * ucb_multiplier
                    else:
                        alpha_actor = alpha.detach()

                    if DG_ENABLED:
                        with torch.no_grad():
                            if DUELING_ENABLED:
                                (v1, a1, _), (v2, a2, _) = critic.both_with_va(b_obs, ap)
                                advantage = torch.min(a1, a2)
                            elif SACV1_ENABLED or DQV_ENABLED:
                                v_baseline = v_critic(b_obs).squeeze(-1)
                                advantage = qp.detach() - v_baseline
                            else:
                                ap2, _ = actor(b_obs)
                                q_baseline = critic(b_obs, ap2)
                                advantage = qp.detach() - q_baseline
                            surprisal = (-lp.detach()).clamp(-DG_CLIP_SURPRISAL, DG_CLIP_SURPRISAL)
                            delight = advantage * surprisal
                            if DG_WHITEN:
                                delight = (delight - delight.mean()) / (delight.std() + 1e-6)
                            gate = torch.sigmoid(delight / DG_ETA)
                        if DG_GATE_Q_ONLY:
                            a_loss = (alpha_actor * lp - gate * qp).mean()
                        else:
                            a_loss = (gate * (alpha_actor * lp - qp)).mean()
                        dg_gate_mean = gate.mean().item()
                        dg_gate_std = gate.std().item()
                        dg_delight_mean = delight.mean().item()
                    else:
                        a_loss = (alpha_actor * lp - qp).mean()
                        dg_gate_mean = 1.0
                        dg_gate_std = 0.0
                        dg_delight_mean = 0.0

                actor_opt.zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
                actor_opt.step()

                if RSAT_ENABLED:
                    # RSAT: joint update of scalar α and residual net
                    # Recompute α(s) with grad for both log_alpha and rsat_net
                    alpha_scalar_u = log_alpha.exp()
                    alpha_for_update = alpha_scalar_u * (1.0 + RSAT_EPS * torch.tanh(rsat_net(b_obs)).squeeze(-1))
                    al = -(alpha_for_update * (lp.detach() + target_entropy)).mean()
                    alpha_opt.zero_grad()
                    rsat_opt.zero_grad()
                    al.backward()
                    alpha_opt.step()
                    rsat_opt.step()
                    with torch.no_grad():
                        log_alpha.clamp_(min=np.log(ALPHA_MIN))
                    alpha_mean = alpha_for_update.mean().item()
                elif SAT_ENABLED:
                    # Original SAT: train alpha_net to match target entropy per state
                    alpha_for_update = F.softplus(alpha_net(b_obs)).squeeze(-1)
                    al = -(alpha_for_update * (lp.detach() + target_entropy)).mean()
                    alpha_opt.zero_grad()
                    al.backward()
                    alpha_opt.step()
                    alpha_mean = alpha_for_update.mean().item()
                elif LEARNABLE_ALPHA and alpha_opt is not None:
                    al = -(log_alpha.exp() * (lp.detach() + target_entropy)).mean()
                    alpha_opt.zero_grad()
                    al.backward()
                    alpha_opt.step()
                    with torch.no_grad():
                        log_alpha.clamp_(min=np.log(ALPHA_MIN))
                    alpha_mean = log_alpha.exp().item()
                else:
                    alpha_mean = alpha.mean().item() if (SAT_ENABLED or RSAT_ENABLED) else alpha.item()

                last_metrics = {
                    "critic_loss": c_loss.item(),
                    "actor_loss": a_loss.item(),
                    "alpha": alpha_mean,
                    "q_mean": q1v.mean().item(),
                    "target_q_mean": soft_t.mean().item(),
                    "log_prob": lp.mean().item(),
                    "dg_gate_mean": dg_gate_mean,
                    "dg_gate_std": dg_gate_std,
                    "dg_delight_mean": dg_delight_mean,
                    "v_loss": v_loss.item(),
                    "v_mean": v_pred.mean().item() if (DQV_ENABLED or SACV1_ENABLED) else 0.0,
                    "alpha_std": alpha_actor.std().item() if (SAT_ENABLED or RSAT_ENABLED or UCB_ENABLED or TDE_ENABLED) else 0.0,
                    "alpha_min": alpha_actor.min().item() if (SAT_ENABLED or RSAT_ENABLED or UCB_ENABLED or TDE_ENABLED) else alpha_mean,
                    "alpha_max": alpha_actor.max().item() if (SAT_ENABLED or RSAT_ENABLED or UCB_ENABLED or TDE_ENABLED) else alpha_mean,
                    "ucb_mult_mean": ucb_multiplier.mean().item() if UCB_ENABLED else (tde_multiplier.mean().item() if TDE_ENABLED else 1.0),
                    "ucb_mult_max": ucb_multiplier.max().item() if UCB_ENABLED else (tde_multiplier.max().item() if TDE_ENABLED else 1.0),
                    "q_disagree_mean": q_disagree.mean().item() if UCB_ENABLED else (td_err.mean().item() if TDE_ENABLED else 0.0),
                }
                if SPG_ENABLED:
                    last_metrics.update(spg_diag)

            critic_target.update()
            if v_target is not None:
                v_target.update()
            if v_target2 is not None:
                v_target2.update()

            if grad_steps % 1000 == 0 and last_metrics:
                metrics.log(env_steps, grad_steps=grad_steps, ep_count=ep_count, **last_metrics)

        if env_steps % 20000 < ACTION_REPEAT and env_steps >= 20000:
            alpha_val = last_metrics.get("alpha", 0.1) if last_metrics else (log_alpha.exp().item() if log_alpha is not None else 0.1)
            q_str = f"q={last_metrics.get('q_mean', 0):.1f}" if last_metrics else "q=n/a"
            cl_str = f"cl={last_metrics.get('critic_loss', 0):.3f}" if last_metrics else "cl=n/a"
            if (SAT_ENABLED or RSAT_ENABLED) and last_metrics:
                alpha_str = f"α={alpha_val:.3f}[{last_metrics.get('alpha_min', 0):.3f}-{last_metrics.get('alpha_max', 0):.3f}]"
            elif (UCB_ENABLED or TDE_ENABLED) and last_metrics:
                tag = "tde" if TDE_ENABLED else "ucb"
                m = last_metrics.get('ucb_mult_mean', 1.0)
                mx = last_metrics.get('ucb_mult_max', 1.0)
                alpha_str = f"α={alpha_val:.4f} {tag}={m:.2f}x(max={mx:.2f}x)"
            else:
                alpha_str = f"alpha={alpha_val:.4f}"
            print(f"[{task_name}] step={env_steps} gs={grad_steps} eps={ep_count} "
                  f"{alpha_str} {q_str} {cl_str} "
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

    # TASK env var: run a single specific task (e.g., TASK=quadruped)
    single_task = os.environ.get("TASK", "")

    try:
        if single_task:
            ret, np_t = train_sac(single_task)
            total_time = check_time()
            print(f"\n{single_task}_return: {ret:.2f}")
            print(f"training_seconds: {total_time:.1f}")
        else:
            cheetah_return, np_c = train_sac("cheetah")
            check_time()
            if not CHEETAH_ONLY:
                humanoid_return, np_h = train_sac("humanoid")
            total_time = check_time()
            print_summary(
                cheetah_return=cheetah_return, humanoid_return=humanoid_return,
                training_seconds=total_time, total_seconds=total_time,
                num_params_cheetah=np_c, num_params_humanoid=np_h, device=DEVICE,
            )
    except (TimeLimitExceeded, SystemExit) as e:
        print(f"\n!!! TIMEOUT — printing partial results !!!", flush=True)
