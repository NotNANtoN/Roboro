"""Micro-benchmark: measure exact per-step costs for different hidden dims."""
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

SEED = 42
BS = 256
STEPS = 5000

def bench(hd, actor_ln, env_id="Walker2d-v5"):
    set_seed(SEED)
    env = gym.make(env_id)
    od = env.observation_space.shape[0]
    ad = env.action_space.shape[0]
    al, ah = float(np.min(env.action_space.low)), float(np.max(env.action_space.high))

    actor = SquashedGaussianActor(obs_dim=od, action_dim=ad, action_low=al, action_high=ah,
                                   hidden_dim=hd, n_layers=2, activation="relu",
                                   use_layer_norm=actor_ln)
    q1 = ContinuousQCritic(feature_dim=od, action_dim=ad, hidden_dim=hd, n_layers=2,
                            activation="relu", use_layer_norm=True)
    q2 = ContinuousQCritic(feature_dim=od, action_dim=ad, hidden_dim=hd, n_layers=2,
                            activation="relu", use_layer_norm=True)
    critic = TwinQCritic(q1, q2)

    def ortho_init(module, gain=np.sqrt(2)):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
    ortho_init(critic); ortho_init(actor.trunk)

    ct = TargetNetwork(critic, mode="polyak", tau=0.005)
    aopt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    copt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # Fake buffer data
    obs_b = torch.randn(BS, od)
    act_b = torch.randn(BS, ad)
    rew_b = torch.randn(BS)
    nobs_b = torch.randn(BS, od)
    done_b = torch.zeros(BS, dtype=torch.bool)

    obs = torch.randn(1, od)

    # Warmup
    for _ in range(100):
        with torch.no_grad():
            na, nlp = actor(nobs_b)
            tq = ct(nobs_b, na)
            st = rew_b + 0.99 * tq
        q1v, q2v = critic.both(obs_b, act_b)
        cl = F.mse_loss(q1v, st) + F.mse_loss(q2v, st)
        copt.zero_grad(); cl.backward(); copt.step()
        ct.update()

    # Time critic-only step
    t0 = time.perf_counter()
    for _ in range(STEPS):
        alpha = torch.tensor(0.1)
        with torch.no_grad():
            na, nlp = actor(nobs_b)
            tq = ct(nobs_b, na)
            st = rew_b + 0.99 * (tq - alpha * nlp)
        q1v, q2v = critic.both(obs_b, act_b)
        cl = F.mse_loss(q1v, st) + F.mse_loss(q2v, st)
        copt.zero_grad(); cl.backward(); copt.step()
        ct.update()
    t_critic = (time.perf_counter() - t0) / STEPS * 1000

    # Time actor step
    t0 = time.perf_counter()
    for _ in range(STEPS):
        ap, lp = actor(obs_b)
        qp = critic(obs_b, ap)
        al_ = (0.1 * lp - qp).mean()
        aopt.zero_grad(); al_.backward(); aopt.step()
    t_actor = (time.perf_counter() - t0) / STEPS * 1000

    # Time inference (single obs)
    t0 = time.perf_counter()
    for _ in range(STEPS):
        with torch.no_grad():
            actor.act(obs, deterministic=False)
    t_infer = (time.perf_counter() - t0) / STEPS * 1000

    env.close()
    print(f"hd={hd:3d} actor_ln={str(actor_ln):5s} | critic={t_critic:.3f}ms actor={t_actor:.3f}ms infer={t_infer:.3f}ms")
    return t_critic, t_actor, t_infer

if __name__ == "__main__":
    for hd in [64, 96, 128]:
        for aln in [True, False]:
            bench(hd, aln)
