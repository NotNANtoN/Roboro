"""Quick screening: 20k steps each env, ~80s total. For rapid idea testing."""

import os, sys, time
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
from prepare import TASKS, evaluate, normalize_return

# ═════════════════════════════════════════════════════════════════════════════
# OVERRIDE THESE VIA ENVIRONMENT VARIABLES OR MODIFY DIRECTLY
# ═════════════════════════════════════════════════════════════════════════════
QUICK_STEPS = 20_000

SEED = 42
DEVICE = "cpu"
HIDDEN_DIM = int(os.environ.get("HD", "128"))
N_LAYERS = int(os.environ.get("NL", "2"))
ACTIVATION = os.environ.get("ACT", "relu")
USE_LAYER_NORM = os.environ.get("LN", "1") == "1"
LR = float(os.environ.get("LR", "1e-3"))
CRITIC_LR = float(os.environ.get("CLR", os.environ.get("LR", "1e-3")))
GAMMA = float(os.environ.get("GAMMA", "0.99"))
BATCH_SIZE = int(os.environ.get("BS", "256"))
WARMUP_STEPS = int(os.environ.get("WU", "500"))
ACTOR_DELAY = int(os.environ.get("AD", "2"))
TAU = float(os.environ.get("TAU", "0.005"))
INIT_ALPHA = float(os.environ.get("ALPHA", "0.1"))
LEARNABLE_ALPHA = True
GRAD_CLIP = float(os.environ.get("GC", "0"))  # 0 = no clip


class FastBuf:
    def __init__(self, cap, od, ad, seed=None):
        self.obs = np.zeros((cap, od), dtype=np.float32)
        self.act = np.zeros((cap, ad), dtype=np.float32)
        self.rew = np.zeros(cap, dtype=np.float32)
        self.nobs = np.zeros((cap, od), dtype=np.float32)
        self.done = np.zeros(cap, dtype=np.bool_)
        self.pos = self.size = 0
        self.cap = cap
        self.rng = np.random.default_rng(seed)

    def add(self, o, a, r, no, d):
        i = self.pos
        self.obs[i]=o; self.act[i]=a; self.rew[i]=r; self.nobs[i]=no; self.done[i]=d
        self.pos = (self.pos+1) % self.cap
        self.size = min(self.size+1, self.cap)

    def sample(self, n):
        idx = self.rng.integers(0, self.size, size=n)
        return (torch.from_numpy(self.obs[idx]), torch.from_numpy(self.act[idx]),
                torch.from_numpy(self.rew[idx]), torch.from_numpy(self.nobs[idx]),
                torch.from_numpy(self.done[idx]))

    def __len__(self): return self.size


def run_quick(task_name):
    task = TASKS[task_name]
    set_seed(SEED)
    env = gym.make(task.env_id)
    od = env.observation_space.shape[0]
    ad = env.action_space.shape[0]
    al, ah = float(np.min(env.action_space.low)), float(np.max(env.action_space.high))

    actor = SquashedGaussianActor(obs_dim=od, action_dim=ad, action_low=al, action_high=ah,
                                   hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS,
                                   activation=ACTIVATION, use_layer_norm=USE_LAYER_NORM)
    q1 = ContinuousQCritic(feature_dim=od, action_dim=ad, hidden_dim=HIDDEN_DIM,
                            n_layers=N_LAYERS, activation=ACTIVATION, use_layer_norm=USE_LAYER_NORM)
    q2 = ContinuousQCritic(feature_dim=od, action_dim=ad, hidden_dim=HIDDEN_DIM,
                            n_layers=N_LAYERS, activation=ACTIVATION, use_layer_norm=USE_LAYER_NORM)
    critic = TwinQCritic(q1, q2)
    ct = TargetNetwork(critic, mode="polyak", tau=TAU)

    buf = FastBuf(100_000, od, ad, seed=SEED)
    aopt = torch.optim.Adam(actor.parameters(), lr=LR)
    copt = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
    log_alpha = nn.Parameter(torch.tensor(float(INIT_ALPHA)).log())
    alopt = torch.optim.Adam([log_alpha], lr=LR)
    te = -float(ad)

    obs, _ = env.reset(seed=SEED)
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    gs = 0

    for step in range(1, QUICK_STEPS + 1):
        a, _ = actor.act(obs_t)
        anp = a.squeeze(0).cpu().numpy()
        no, r, term, trunc, _ = env.step(anp)
        buf.add(obs, anp, float(r), no, term)
        obs = no
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if term or trunc:
            obs, _ = env.reset()
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        if step >= WARMUP_STEPS and len(buf) >= BATCH_SIZE:
            bo, ba, br, bn, bd = buf.sample(BATCH_SIZE)
            alpha = log_alpha.exp()
            gs += 1

            with torch.no_grad():
                na, nlp = actor(bn)
                tq = ct(bn, na)
                st = br + GAMMA * (~bd).float() * (tq - alpha * nlp)
            q1v, q2v = critic.both(bo, ba)
            cl = F.mse_loss(q1v, st) + F.mse_loss(q2v, st)
            copt.zero_grad()
            cl.backward()
            if GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
            copt.step()

            if gs % ACTOR_DELAY == 0:
                ap, lp = actor(bo)
                qp = critic(bo, ap)
                al_ = (alpha.detach() * lp - qp).mean()
                aopt.zero_grad()
                al_.backward()
                aopt.step()
                all_ = -(log_alpha.exp() * (lp.detach() + te)).mean()
                alopt.zero_grad()
                all_.backward()
                alopt.step()

            ct.update()

    def pfn(o):
        with torch.no_grad():
            return actor.act(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0), deterministic=True)[0].squeeze(0).numpy()

    er = evaluate(task.env_id, pfn, 5, task.eval_seed)
    env.close()
    return er


if __name__ == "__main__":
    t0 = time.time()
    hr = run_quick("hopper")
    wr = run_quick("walker")
    elapsed = time.time() - t0

    hn = max(0, hr / 3500)
    wn = max(0, wr / 5000)
    sc = (hn + wn) / 2

    tag = os.environ.get("TAG", "default")
    print(f"[{tag}] score={sc:.4f} hopper={hr:.1f}({hn:.3f}) walker={wr:.1f}({wn:.3f}) time={elapsed:.1f}s")
