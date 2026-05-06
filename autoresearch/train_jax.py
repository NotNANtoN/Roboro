"""JAX SAC on MuJoCo Playground (MJX). Single-env, fully on-device.

Supports all algorithm variants from train_dmc.py:
  - Standard SAC with twin-Q or N-critic ensemble
  - Dueling SAC (Q = V + A with L2 reg)
  - SPG actor update with SBA, pessimistic Q, batched eval
  - SAT / RSAT state-adaptive temperature
  - SPG diagnostics (cosine sim, Q-gain, source tracking)
  - Per-chunk CSV metrics logging

Usage:
  python train_jax.py                                      # SAC baseline
  DUELING=1 python train_jax.py                            # Dueling SAC
  SPG=1 SPG_SAMPLES=32 DUELING=1 python train_jax.py       # SPG + Dueling
  N_CRITICS=8 python train_jax.py                          # Ensemble SAC
  SAT=1 python train_jax.py                                # State-Adaptive Temp
  RSAT=1 python train_jax.py                               # Residual SAT
"""

import csv
import os
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
import mujoco_playground as mp

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════
SEED = int(os.environ.get("SEED", "42"))
TASK = os.environ.get("TASK", "CheetahRun")
HIDDEN_DIM = 256
LR = 5e-3
GAMMA = 0.995
BATCH_SIZE = 128
BUFFER_CAPACITY = 100_000
WARMUP_STEPS = 512
ACTOR_DELAY = 2
TAU = 0.005
ACTION_REPEAT = 2
INIT_ALPHA = 0.1
TARGET_ENTROPY_SCALE = 0.5
ALPHA_MIN = 0.01
MAX_GRAD_NORM = 1.0
TOTAL_STEPS = int(os.environ.get("STEPS", "100000"))
N_CRITICS = int(os.environ.get("N_CRITICS", "2"))
EVAL_EPISODES = int(os.environ.get("EVAL_EPS", "20"))
LOG_INTERVAL = int(os.environ.get("LOG_INTERVAL", "20000"))
CHUNK_SIZE = int(os.environ.get("CHUNK", "500"))

DUELING_ENABLED = bool(int(os.environ.get("DUELING", "0")))
DUELING_BETA = float(os.environ.get("DUELING_BETA", "0.01"))

SPG_ENABLED = bool(int(os.environ.get("SPG", "0")))
SPG_SAMPLES = int(os.environ.get("SPG_SAMPLES", "32"))

SAT_ENABLED = bool(int(os.environ.get("SAT", "0")))
SAT_LR = float(os.environ.get("SAT_LR", "0.0001"))

RSAT_ENABLED = bool(int(os.environ.get("RSAT", "0")))
RSAT_EPS = float(os.environ.get("RSAT_EPS", "0.5"))
RSAT_LR = float(os.environ.get("RSAT_LR", "0.0001"))

LOG_STD_MIN, LOG_STD_MAX = -10.0, 2.0


# ═════════════════════════════════════════════════════════════════════════════
# NETWORKS
# ═════════════════════════════════════════════════════════════════════════════
def _ortho(scale):
    return nn.initializers.orthogonal(scale)

class QNetwork(nn.Module):
    """Standard Q(s,a) → scalar."""
    hidden_dim: int = 256
    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], -1)
        x = nn.Dense(self.hidden_dim, kernel_init=_ortho(jnp.sqrt(2)))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=_ortho(jnp.sqrt(2)))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        return nn.Dense(1, kernel_init=_ortho(0.01))(x).squeeze(-1)


class DuelingQNetwork(nn.Module):
    """Q(s,a) = V(s) + A(s,a) with shared trunk. Returns (Q, V, A)."""
    hidden_dim: int = 256
    @nn.compact
    def __call__(self, obs, action):
        trunk = nn.Dense(self.hidden_dim, kernel_init=_ortho(jnp.sqrt(2)))(obs)
        trunk = nn.LayerNorm()(trunk)
        trunk = nn.relu(trunk)
        trunk = nn.Dense(self.hidden_dim, kernel_init=_ortho(jnp.sqrt(2)))(trunk)
        trunk = nn.LayerNorm()(trunk)
        trunk = nn.relu(trunk)
        v = nn.Dense(1, kernel_init=_ortho(0.01))(trunk).squeeze(-1)
        a_in = jnp.concatenate([trunk, action], -1)
        a_hidden = nn.relu(nn.Dense(self.hidden_dim, kernel_init=_ortho(jnp.sqrt(2)))(a_in))
        a_val = nn.Dense(1, kernel_init=_ortho(0.01))(a_hidden).squeeze(-1)
        return v + a_val, v, a_val  # (Q, V, A)


class Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    @nn.compact
    def __call__(self, obs, key):
        x = nn.Dense(self.hidden_dim, kernel_init=_ortho(jnp.sqrt(2)))(obs)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=_ortho(jnp.sqrt(2)))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim, kernel_init=_ortho(0.01))(x)
        log_std = jnp.clip(
            nn.Dense(self.action_dim, kernel_init=_ortho(0.01))(x),
            LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        u = mean + std * jax.random.normal(key, mean.shape)
        action = jnp.tanh(u)
        log_prob = (-0.5 * (((u - mean) / (std + 1e-6))**2 + 2*log_std
                   + jnp.log(2*jnp.pi))).sum(-1)
        log_prob -= jnp.log(1 - action**2 + 1e-6).sum(-1)
        return action, log_prob, mean, std


class AlphaNet(nn.Module):
    """SAT: state-dependent alpha. Output: softplus → α > 0."""
    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(64, kernel_init=_ortho(0.1))(obs)
        x = nn.relu(x)
        return nn.softplus(nn.Dense(1, kernel_init=_ortho(0.01),
                           bias_init=nn.initializers.constant(jnp.log(INIT_ALPHA)))(x)).squeeze(-1)


class RSATNet(nn.Module):
    """RSAT: residual α(s) = α_scalar · (1 + ε·tanh(f(s)))."""
    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(64, kernel_init=_ortho(0.1))(obs)
        x = nn.relu(x)
        return nn.Dense(1, kernel_init=_ortho(0.01),
                        bias_init=nn.initializers.zeros)(x).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# REPLAY BUFFER — pure JAX, on-device
# ═════════════════════════════════════════════════════════════════════════════
class Buffer(NamedTuple):
    obs: jnp.ndarray
    act: jnp.ndarray
    rew: jnp.ndarray
    nobs: jnp.ndarray
    done: jnp.ndarray
    best_act: jnp.ndarray    # SBA: stored best action
    pos: jnp.ndarray
    size: jnp.ndarray

def buf_new(obs_dim, act_dim):
    return Buffer(
        obs=jnp.zeros((BUFFER_CAPACITY, obs_dim)),
        act=jnp.zeros((BUFFER_CAPACITY, act_dim)),
        rew=jnp.zeros(BUFFER_CAPACITY),
        nobs=jnp.zeros((BUFFER_CAPACITY, obs_dim)),
        done=jnp.zeros(BUFFER_CAPACITY),
        best_act=jnp.zeros((BUFFER_CAPACITY, act_dim)),
        pos=jnp.int32(0), size=jnp.int32(0))

def buf_add(b, obs, act, rew, nobs, done):
    p = b.pos
    return Buffer(
        obs=b.obs.at[p].set(obs), act=b.act.at[p].set(act),
        rew=b.rew.at[p].set(rew), nobs=b.nobs.at[p].set(nobs),
        done=b.done.at[p].set(done),
        best_act=b.best_act.at[p].set(act),  # SBA: init to original action
        pos=(p + 1) % BUFFER_CAPACITY, size=jnp.minimum(b.size + 1, BUFFER_CAPACITY))

def buf_sample(b, key):
    idx = jax.random.randint(key, (BATCH_SIZE,), 0, b.size)
    return (b.obs[idx], b.act[idx], b.rew[idx], b.nobs[idx], b.done[idx],
            b.best_act[idx], idx)

def buf_update_best(b, idx, best_act):
    return b._replace(best_act=b.best_act.at[idx].set(best_act))


# ═════════════════════════════════════════════════════════════════════════════
# METRICS LOGGER
# ═════════════════════════════════════════════════════════════════════════════
class MetricsLogger:
    def __init__(self):
        self.rows = []
        self.fields = None
    def log(self, **kw):
        if self.fields is None:
            self.fields = list(kw.keys())
        self.rows.append(kw)
    def dump(self, path):
        if not self.rows:
            return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            w.writeheader()
            w.writerows(self.rows)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    rng = jax.random.PRNGKey(SEED)
    print(f"JAX devices: {jax.devices()}", flush=True)

    env = mp.dm_control_suite.load(TASK, config_overrides={'impl': 'jax'})
    obs_dim, action_dim = env.observation_size, env.action_size
    print(f"[{TASK}] obs={obs_dim}, act={action_dim}", flush=True)
    print(f"Config: N_CRITICS={N_CRITICS} DUELING={DUELING_ENABLED} "
          f"SPG={SPG_ENABLED}({SPG_SAMPLES}) SAT={SAT_ENABLED} RSAT={RSAT_ENABLED}", flush=True)

    # ── Network init ─────────────────────────────────────────────────────
    actor = Actor(action_dim=action_dim, hidden_dim=HIDDEN_DIM)
    q_net = DuelingQNetwork(hidden_dim=HIDDEN_DIM) if DUELING_ENABLED else QNetwork(hidden_dim=HIDDEN_DIM)

    rng, akey, *qkeys = jax.random.split(rng, 2 + N_CRITICS)
    d_obs, d_act = jnp.zeros(obs_dim), jnp.zeros(action_dim)
    actor_params = actor.init(akey, d_obs, akey)
    q_params = jax.tree.map(lambda *xs: jnp.stack(xs),
                            *[q_net.init(qkeys[i], d_obs, d_act) for i in range(N_CRITICS)])
    tgt_q = jax.tree.map(jnp.copy, q_params)

    gamma_eff = GAMMA ** ACTION_REPEAT
    target_entropy = -action_dim * TARGET_ENTROPY_SCALE

    # Critic helpers
    if DUELING_ENABLED:
        def q_apply_all(qp, o, a):
            results = jax.vmap(lambda p: q_net.apply(p, o, a))(qp)
            return results[0]  # (N, B) Q-values only
        def q_apply_all_with_va(qp, o, a):
            return jax.vmap(lambda p: q_net.apply(p, o, a))(qp)  # (Q, V, A) each (N, B)
    else:
        q_apply_all = lambda qp, o, a: jax.vmap(lambda p: q_net.apply(p, o, a))(qp)
    q_min_fn = lambda qp, o, a: jnp.min(q_apply_all(qp, o, a), axis=0)

    # SAT / RSAT networks (always init for scan compatibility, only used when enabled)
    sat_net = AlphaNet()
    rsat_net_model = RSATNet()
    rng, sat_key = jax.random.split(rng)
    sat_params = sat_net.init(sat_key, d_obs)
    rsat_params = rsat_net_model.init(sat_key, d_obs)

    # Optimizers
    a_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR))
    c_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR))
    al_opt = optax.adam(LR)
    a_opt_s = a_opt.init(actor_params)
    c_opt_s = c_opt.init(q_params)
    log_alpha = jnp.log(INIT_ALPHA)
    al_opt_s = al_opt.init(log_alpha)

    sat_opt = optax.adam(SAT_LR)
    sat_opt_s = sat_opt.init(sat_params)
    rsat_opt = optax.adam(RSAT_LR)
    rsat_opt_s = rsat_opt.init(rsat_params)

    buf = buf_new(obs_dim, action_dim)
    metrics = MetricsLogger()

    # ── Alpha computation (handles scalar / SAT / RSAT) ──────────────────
    def get_alpha(la, obs, sat_p=None, rsat_p=None):
        """Returns alpha (scalar or per-state) and alpha_next for Bellman."""
        if SAT_ENABLED and sat_p is not None:
            return sat_net.apply(sat_p, obs)
        elif RSAT_ENABLED and rsat_p is not None:
            base = jnp.exp(la)
            return base * (1.0 + RSAT_EPS * jnp.tanh(rsat_net_model.apply(rsat_p, obs)))
        else:
            return jnp.exp(la)

    # ── Core update functions ────────────────────────────────────────────
    def critic_update(qp, tqp, ap, la, obs, act, rew, nobs, dn, key, cs,
                      sat_p=None, rsat_p=None):
        alpha_next = get_alpha(la, nobs, sat_p, rsat_p)
        na, nlp, _, _ = actor.apply(ap, nobs, key)
        tgt = rew + gamma_eff * (1-dn) * (jnp.min(q_apply_all(tqp, nobs, na), 0) - alpha_next*nlp)

        def loss(qp):
            if DUELING_ENABLED:
                qs, vs, a_vals = q_apply_all_with_va(qp, obs, act)
                td_loss = jnp.sum(jnp.mean((qs - tgt[None,:])**2, axis=1))
                # L2 reg: sum per-critic means (match PyTorch: v1²+a1²+v2²+a2²)
                l2_reg = (DUELING_BETA / 2) * jnp.sum(
                    jnp.mean(vs**2, axis=1) + jnp.mean(a_vals**2, axis=1))
                return td_loss + l2_reg
            else:
                qs = q_apply_all(qp, obs, act)
                return jnp.sum(jnp.mean((qs - tgt[None,:])**2, axis=1))

        l, g = jax.value_and_grad(loss)(qp)
        u, cs = c_opt.update(g, cs)
        return optax.apply_updates(qp, u), cs, l

    def actor_update_standard(ap, qp, la, obs, key, aos, sat_p=None, rsat_p=None):
        alpha = get_alpha(la, obs, sat_p, rsat_p)
        def loss(ap):
            a, lp, _, _ = actor.apply(ap, obs, key)
            return jnp.mean(alpha*lp - q_min_fn(qp, obs, a)), lp
        (l, lp), g = jax.value_and_grad(loss, has_aux=True)(ap)
        u, aos = a_opt.update(g, aos)
        return optax.apply_updates(ap, u), aos, lp

    def spg_find_best(ap, qp, obs, act, best_stored, key):
        """SPG: find best action via sampling. Returns (best_act, improved, q_actor, q_best)."""
        _, _, actor_mean, actor_std = actor.apply(ap, obs, key)
        actor_action = jnp.tanh(actor_mean)
        q_actor = q_min_fn(qp, obs, actor_action)

        best = best_stored
        best_q = q_min_fn(qp, obs, best)

        # Check actor
        actor_better = q_actor > best_q
        best = jnp.where(actor_better[:, None], actor_action, best)
        best_q = jnp.where(actor_better, q_actor, best_q)

        # Check replay
        q_replay = q_min_fn(qp, obs, act)
        replay_better = q_replay > best_q
        best = jnp.where(replay_better[:, None], act, best)
        best_q = jnp.where(replay_better, q_replay, best_q)

        # Batched Gaussian samples
        B, S = obs.shape[0], SPG_SAMPLES
        key, sk = jax.random.split(key)
        noise = jax.random.normal(sk, (B, S, action_dim)) * jnp.clip(actor_std[:, None, :], 0.01)
        samples = jnp.clip(best[:, None, :] + noise, -1.0, 1.0)
        obs_exp = jnp.broadcast_to(obs[:, None, :], (B, S, obs_dim)).reshape(B*S, -1)
        q_all = q_min_fn(qp, obs_exp, samples.reshape(B*S, -1)).reshape(B, S)
        best_idx = jnp.argmax(q_all, axis=1)
        best_sample = samples[jnp.arange(B), best_idx]
        q_best_sample = q_all[jnp.arange(B), best_idx]
        sample_better = q_best_sample > best_q
        best = jnp.where(sample_better[:, None], best_sample, best)
        best_q = jnp.where(sample_better, q_best_sample, best_q)

        improved = best_q > q_actor
        return best, improved, q_actor, best_q

    def spg_actor_update(ap, qp, la, obs, best_act, improved, key, aos,
                         sat_p=None, rsat_p=None):
        alpha = get_alpha(la, obs, sat_p, rsat_p)
        def loss(ap):
            a, lp, mean, _ = actor.apply(ap, obs, key)
            actor_sq = jnp.tanh(mean)
            mse = jnp.sum((actor_sq - best_act)**2, axis=-1)
            return jnp.mean(mse * improved) + jnp.mean(alpha * lp), lp
        (l, lp), g = jax.value_and_grad(loss, has_aux=True)(ap)
        u, aos = a_opt.update(g, aos)
        return optax.apply_updates(ap, u), aos, lp

    def spg_diagnostics(ap, qp, obs, best_act, improved, q_actor, best_q, key):
        """Compute cosine sim between SPG direction and reparam gradient."""
        _, _, mean, _ = actor.apply(ap, obs, key)
        actor_sq = jnp.tanh(mean)
        spg_dir = best_act - actor_sq

        # Reparam gradient direction
        def q_of_action(a):
            return q_min_fn(qp, obs, a).sum()
        reparam_dir = jax.grad(q_of_action)(actor_sq)

        cos_sim = jnp.sum(spg_dir * reparam_dir, axis=-1) / (
            jnp.linalg.norm(spg_dir, axis=-1) * jnp.linalg.norm(reparam_dir, axis=-1) + 1e-8)

        # Q at reparam step (normalized)
        step_norm = jnp.linalg.norm(spg_dir, axis=-1, keepdims=True)
        reparam_step = actor_sq + reparam_dir / (jnp.linalg.norm(reparam_dir, axis=-1, keepdims=True) + 1e-8) * step_norm
        reparam_step = jnp.clip(reparam_step, -1.0, 1.0)
        q_reparam = q_min_fn(qp, obs, reparam_step)

        return {
            'cos_sim': cos_sim.mean(),
            'cos_sim_improved': jnp.where(improved, cos_sim, 0.0).sum() / (improved.sum() + 1e-8),
            'spg_q_gain': (best_q - q_actor).mean(),
            'reparam_q_gain': (q_reparam - q_actor).mean(),
            'spg_frac': improved.astype(jnp.float32).mean(),
            'spg_step_norm': jnp.linalg.norm(spg_dir, axis=-1).mean(),
        }

    def alpha_update(la, lp, als):
        def loss(la):
            return -jnp.mean(jnp.exp(la) * (jax.lax.stop_gradient(lp) + target_entropy))
        _, g = jax.value_and_grad(loss)(la)
        u, als = al_opt.update(g, als)
        return jnp.maximum(optax.apply_updates(la, u), jnp.log(ALPHA_MIN)), als

    def sat_update(sat_p, lp, obs, sat_os):
        def loss(sat_p):
            alpha_s = sat_net.apply(sat_p, obs)
            return -jnp.mean(alpha_s * (jax.lax.stop_gradient(lp) + target_entropy))
        _, g = jax.value_and_grad(loss)(sat_p)
        u, sat_os = sat_opt.update(g, sat_os)
        return optax.apply_updates(sat_p, u), sat_os

    # ── Training step ──────────────────────────────────────────────────────
    USE_SAT_CARRY = SAT_ENABLED or RSAT_ENABLED

    def train_step(carry, _):
        if USE_SAT_CARRY:
            es, buf, qp, tqp, ap, cs, aos, la, als, sp, sos, rp, ros, gs, key = carry
        else:
            es, buf, qp, tqp, ap, cs, aos, la, als, gs, key = carry
            sp, sos, rp, ros = None, None, None, None

        key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)

        # Collect
        obs0 = es.obs
        action, _, _, _ = actor.apply(ap, obs0, k1)
        def do_repeat(carry, _):
            s, r = carry
            s = env.step(s, action)
            return (s, r + s.reward), None
        (es, tot_rew), _ = jax.lax.scan(do_repeat, (es, 0.0), None, length=ACTION_REPEAT)
        buf = buf_add(buf, obs0, action, tot_rew, es.obs, es.done.astype(jnp.float32))

        # Sample and train
        bo, ba, br, bn, bd, b_best, b_idx = buf_sample(buf, k2)

        qp, cs, cl = critic_update(qp, tqp, ap, la, bo, ba, br, bn, bd, k3, cs, sp, rp)
        gs = gs + 1

        if SPG_ENABLED:
            best, improved, q_actor, best_q = spg_find_best(ap, qp, bo, ba, b_best, k4)
            ap2, aos2, lp = spg_actor_update(ap, qp, la, bo, best, improved.astype(jnp.float32),
                                             k5, aos, sp, rp)
            buf = buf_update_best(buf, b_idx, best)
        else:
            ap2, aos2, lp = actor_update_standard(ap, qp, la, bo, k4, aos, sp, rp)

        # Alpha / SAT / RSAT update
        if SAT_ENABLED:
            sp2, sos2 = sat_update(sp, lp, bo, sos)
            la2, als2 = la, als
        elif RSAT_ENABLED:
            la2, als2 = alpha_update(la, lp, als)
            def rsat_loss(rp_):
                base = jnp.exp(la2)
                alpha_s = base * (1.0 + RSAT_EPS * jnp.tanh(rsat_net_model.apply(rp_, bo)))
                return -jnp.mean(alpha_s * (jax.lax.stop_gradient(lp) + target_entropy))
            _, rg = jax.value_and_grad(rsat_loss)(rp)
            ru, ros2 = rsat_opt.update(rg, ros)
            rp2 = optax.apply_updates(rp, ru)
        else:
            la2, als2 = alpha_update(la, lp, als)

        do_a = (gs % ACTOR_DELAY == 0)
        ap = jax.tree.map(lambda n, o: jnp.where(do_a, n, o), ap2, ap)
        aos = jax.tree.map(lambda n, o: jnp.where(do_a, n, o), aos2, aos)
        la = jnp.where(do_a, la2, la)
        als = jax.tree.map(lambda n, o: jnp.where(do_a, n, o), als2, als)
        if SAT_ENABLED:
            sp = jax.tree.map(lambda n, o: jnp.where(do_a, n, o), sp2, sp)
            sos = jax.tree.map(lambda n, o: jnp.where(do_a, n, o), sos2, sos)
        if RSAT_ENABLED:
            rp = jax.tree.map(lambda n, o: jnp.where(do_a, n, o), rp2, rp)
            ros = jax.tree.map(lambda n, o: jnp.where(do_a, n, o), ros2, ros)

        tqp = jax.tree.map(lambda t, q: t * (1 - TAU) + q * TAU, tqp, qp)

        if USE_SAT_CARRY:
            return (es, buf, qp, tqp, ap, cs, aos, la, als, sp, sos, rp, ros, gs, key), la
        return (es, buf, qp, tqp, ap, cs, aos, la, als, gs, key), la

    # ── Warmup ───────────────────────────────────────────────────────────
    def warmup_step(carry, _):
        es, buf, key = carry
        key, k1 = jax.random.split(key)
        obs0 = es.obs
        action = jax.random.uniform(k1, (action_dim,), minval=-1.0, maxval=1.0)
        def do_repeat(carry, _):
            s, r = carry
            s = env.step(s, action)
            return (s, r + s.reward), None
        (es, tot_rew), _ = jax.lax.scan(do_repeat, (es, 0.0), None, length=ACTION_REPEAT)
        buf = buf_add(buf, obs0, action, tot_rew, es.obs, es.done.astype(jnp.float32))
        return (es, buf, key), None

    # ── Run ──────────────────────────────────────────────────────────────
    print("JIT compiling...", flush=True)
    rng, ek, wk, tk = jax.random.split(rng, 4)
    es = env.reset(ek)

    warmup_n = WARMUP_STEPS // ACTION_REPEAT
    (es, buf, _), _ = jax.lax.scan(warmup_step, (es, buf, wk), None, length=warmup_n)
    env_steps = WARMUP_STEPS
    print(f"Buffer filled: {int(buf.size)}", flush=True)

    if USE_SAT_CARRY:
        carry = (es, buf, q_params, tgt_q, actor_params, c_opt_s, a_opt_s,
                 log_alpha, al_opt_s, sat_params, sat_opt_s, rsat_params, rsat_opt_s,
                 jnp.int32(0), tk)
        GS_IDX, AP_IDX, QP_IDX, LA_IDX, BUF_IDX = 13, 4, 2, 7, 1
    else:
        carry = (es, buf, q_params, tgt_q, actor_params, c_opt_s, a_opt_s,
                 log_alpha, al_opt_s, jnp.int32(0), tk)
        GS_IDX, AP_IDX, QP_IDX, LA_IDX, BUF_IDX = 9, 4, 2, 7, 1

    # JIT warmup
    carry, _ = jax.lax.scan(train_step, carry, None, length=1)
    env_steps += ACTION_REPEAT
    print("Starting training...", flush=True)
    t0 = time.time()

    decisions_left = (TOTAL_STEPS - env_steps) // ACTION_REPEAT
    n_chunks = decisions_left // CHUNK_SIZE

    for i in range(n_chunks):
        carry, alphas = jax.lax.scan(train_step, carry, None, length=CHUNK_SIZE)
        env_steps += CHUNK_SIZE * ACTION_REPEAT
        gs = int(carry[GS_IDX])
        la_val = carry[LA_IDX]
        elapsed = time.time() - t0

        alpha_val = float(jnp.exp(la_val)) if not SAT_ENABLED else float(alphas[-1])

        if env_steps % LOG_INTERVAL < CHUNK_SIZE * ACTION_REPEAT + 100:
            print(f"[{TASK}] step={env_steps} gs={gs} alpha={alpha_val:.4f} "
                  f"sps={env_steps/elapsed:.0f} elapsed={elapsed:.0f}s", flush=True)

        metrics.log(
            step=env_steps, grad_steps=gs, alpha=alpha_val,
            sps=env_steps/elapsed, elapsed=elapsed)

    elapsed = time.time() - t0
    gs = int(carry[GS_IDX])
    actor_params = carry[AP_IDX]
    q_params = carry[QP_IDX]
    print(f"\nTraining: {elapsed:.1f}s, {env_steps/elapsed:.0f} sps, {gs} gs", flush=True)

    # ── SPG diagnostics (post-training, on last buffer sample) ───────────
    if SPG_ENABLED:
        buf = carry[BUF_IDX]
        rng, dk = jax.random.split(rng)
        bo, ba, br, bn, bd, b_best, b_idx = buf_sample(buf, dk)
        rng, dk2 = jax.random.split(rng)
        best, improved, q_actor, best_q = spg_find_best(actor_params, q_params, bo, ba, b_best, dk2)
        rng, dk3 = jax.random.split(rng)
        diag = spg_diagnostics(actor_params, q_params, bo, best, improved, q_actor, best_q, dk3)
        print(f"SPG diagnostics: cos_sim={float(diag['cos_sim']):.3f} "
              f"q_gain={float(diag['spg_q_gain']):.3f} "
              f"reparam_gain={float(diag['reparam_q_gain']):.3f} "
              f"frac_improved={float(diag['spg_frac']):.2f}", flush=True)

    # ── Eval ─────────────────────────────────────────────────────────────
    @jax.jit
    def eval_episode(key):
        def step(carry, _):
            s, k, ret = carry
            k, ak = jax.random.split(k)
            a, _, _, _ = actor.apply(actor_params, s.obs, ak)
            s = env.step(s, a)
            return (s, k, ret + s.reward), None
        s = env.reset(key)
        (_, _, ret), _ = jax.lax.scan(step, (s, key, 0.0), None, length=1000)
        return ret

    print("Evaluating...", flush=True)
    rng, eval_key = jax.random.split(rng)
    returns = jax.vmap(eval_episode)(jax.random.split(eval_key, EVAL_EPISODES))
    mean_ret, std_ret = float(returns.mean()), float(returns.std())

    total = time.time() - t0
    print(f"Eval ({EVAL_EPISODES} eps): {mean_ret:.1f} +/- {std_ret:.1f}")
    print(f"training_seconds: {elapsed:.1f}")
    print(f"total_seconds: {total:.1f}")
    print(f"{TASK.lower()}_return: {mean_ret:.2f}")

    metrics.dump(f"runs/{TASK.lower()}_jax_metrics.csv")


if __name__ == "__main__":
    main()
