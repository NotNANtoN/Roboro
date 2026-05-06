"""JAX SAC on MuJoCo Playground (MJX). Clean port of train_dmc.py core.

Supports:
  - Standard SAC with twin-Q or N-critic ensemble
  - SPG actor update (sample-and-evaluate)
  - Dueling critics (TODO)
  - Vectorized env stepping via jax.vmap

Usage:
  python train_jax.py                          # CheetahRun default
  TASK=HumanoidWalk python train_jax.py        # Different task
  N_CRITICS=8 python train_jax.py              # Ensemble
  SPG=1 SPG_SAMPLES=32 python train_jax.py     # SPG actor update
"""

import os
import time
import functools
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

NUM_ENVS = int(os.environ.get("NUM_ENVS", "16"))
TOTAL_STEPS = int(os.environ.get("STEPS", "100000"))
N_CRITICS = int(os.environ.get("N_CRITICS", "2"))
EVAL_EPISODES = int(os.environ.get("EVAL_EPS", "20"))

SPG_ENABLED = bool(int(os.environ.get("SPG", "0")))
SPG_SAMPLES = int(os.environ.get("SPG_SAMPLES", "32"))

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


# ═════════════════════════════════════════════════════════════════════════════
# NETWORKS (Flax)
# ═════════════════════════════════════════════════════════════════════════════
class QNetwork(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], -1)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(1, kernel_init=nn.initializers.orthogonal(0.01))(x)
        return x.squeeze(-1)


class Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs, key):
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(obs)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)
        log_std = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)

        noise = jax.random.normal(key, mean.shape)
        u = mean + std * noise  # pre-squash
        action = jnp.tanh(u)

        # Log prob with tanh correction
        log_prob = -0.5 * (((u - mean) / (std + 1e-6)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
        log_prob = log_prob.sum(-1)
        log_prob -= jnp.log(1 - action ** 2 + 1e-6).sum(-1)  # tanh correction

        return action, log_prob, mean, std

    def get_action_deterministic(self, obs):
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(obs)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)
        return jnp.tanh(mean)


# ═════════════════════════════════════════════════════════════════════════════
# REPLAY BUFFER (numpy, off-JAX for simplicity)
# ═════════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(42)

    def add(self, obs, action, reward, next_obs, done):
        # obs can be (N, obs_dim) for vectorized envs
        n = obs.shape[0] if obs.ndim > 1 else 1
        if obs.ndim == 1:
            obs, action, reward, next_obs, done = (
                obs[None], action[None], np.array([reward]), next_obs[None], np.array([done])
            )
        for i in range(n):
            idx = self.pos
            self.obs[idx] = obs[i]
            self.actions[idx] = action[i]
            self.rewards[idx] = reward[i]
            self.next_obs[idx] = next_obs[i]
            self.dones[idx] = done[i]
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            jnp.array(self.obs[idx]),
            jnp.array(self.actions[idx]),
            jnp.array(self.rewards[idx]),
            jnp.array(self.next_obs[idx]),
            jnp.array(self.dones[idx].astype(np.float32)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════════
def main():
    rng = jax.random.PRNGKey(SEED)
    print(f"JAX devices: {jax.devices()}")

    # Environment
    env = mp.dm_control_suite.load(TASK, config_overrides={'impl': 'jax'})
    obs_dim = env.observation_size
    action_dim = env.action_size
    print(f"[{TASK}] obs_dim={obs_dim}, action_dim={action_dim}")

    # Init networks
    actor = Actor(action_dim=action_dim, hidden_dim=HIDDEN_DIM)
    q_net = QNetwork(hidden_dim=HIDDEN_DIM)

    rng, actor_key, *q_keys = jax.random.split(rng, 2 + N_CRITICS)
    dummy_obs = jnp.zeros(obs_dim)
    dummy_act = jnp.zeros(action_dim)

    actor_params = actor.init(actor_key, dummy_obs, actor_key)
    # Stack all critic params into a single pytree for vmap
    q_params_single = q_net.init(q_keys[0], dummy_obs, dummy_act)
    q_params_stacked = jax.tree.map(
        lambda *xs: jnp.stack(xs),
        *[q_net.init(q_keys[i], dummy_obs, dummy_act) for i in range(N_CRITICS)]
    )
    target_q_params = jax.tree.map(jnp.copy, q_params_stacked)

    log_alpha = jnp.log(INIT_ALPHA)
    target_entropy = -action_dim * TARGET_ENTROPY_SCALE

    # Vmapped critic: apply all N critics at once
    def q_apply_all(q_params_stacked, obs, action):
        """Apply all N critics, return (N, B) Q-values."""
        return jax.vmap(lambda qp: q_net.apply(qp, obs, action))(q_params_stacked)

    def q_min(q_params_stacked, obs, action):
        """Min over all critics."""
        return jnp.min(q_apply_all(q_params_stacked, obs, action), axis=0)

    # Optimizers
    actor_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR))
    critic_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR))
    alpha_opt = optax.adam(LR)

    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(q_params_stacked)  # Single opt state for stacked params
    alpha_opt_state = alpha_opt.init(log_alpha)

    buffer = ReplayBuffer(BUFFER_CAPACITY, obs_dim, action_dim)

    # Vectorized env
    v_reset = jax.jit(jax.vmap(env.reset))
    v_step = jax.jit(jax.vmap(env.step))

    rng, env_key = jax.random.split(rng)
    env_keys = jax.random.split(env_key, NUM_ENVS)
    env_states = v_reset(env_keys)

    # ── JIT-compiled update functions ────────────────────────────────────────
    @jax.jit
    def critic_step(q_params_stacked, target_q_params, actor_params, log_alpha,
                    obs, actions, rewards, next_obs, dones, key, opt_state):
        alpha = jnp.exp(log_alpha)
        gamma_eff = GAMMA ** ACTION_REPEAT
        not_done = 1.0 - dones

        # Compute target
        next_action, next_log_prob, _, _ = actor.apply(actor_params, next_obs, key)
        min_target_q = jnp.min(q_apply_all(target_q_params, next_obs, next_action), axis=0)
        target = rewards + gamma_eff * not_done * (min_target_q - alpha * next_log_prob)

        def loss_fn(q_params_stacked):
            all_q = q_apply_all(q_params_stacked, obs, actions)  # (N, B)
            per_critic_loss = jnp.mean((all_q - target[None, :]) ** 2, axis=1)  # (N,)
            return jnp.sum(per_critic_loss), all_q[0]

        (c_loss, q1v), grads = jax.value_and_grad(loss_fn, has_aux=True)(q_params_stacked)
        updates, new_opt_state = critic_opt.update(grads, opt_state)
        new_q_params = optax.apply_updates(q_params_stacked, updates)
        return new_q_params, new_opt_state, c_loss, q1v

    @jax.jit
    def actor_step(actor_params, q_params_stacked, log_alpha, obs, key, opt_state):
        def loss_fn(actor_params):
            alpha = jnp.exp(log_alpha)
            action, log_prob, _, _ = actor.apply(actor_params, obs, key)
            min_q = q_min(q_params_stacked, obs, action)
            return jnp.mean(alpha * log_prob - min_q), log_prob
        (a_loss, log_prob), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_params)
        updates, new_opt_state = actor_opt.update(grads, opt_state)
        new_actor_params = optax.apply_updates(actor_params, updates)
        return new_actor_params, new_opt_state, a_loss, log_prob

    @jax.jit
    def alpha_step(log_alpha, log_prob, opt_state):
        def loss_fn(log_alpha):
            alpha = jnp.exp(log_alpha)
            return -jnp.mean(alpha * (jax.lax.stop_gradient(log_prob) + target_entropy))
        al, grads = jax.value_and_grad(loss_fn)(log_alpha)
        updates, new_opt_state = alpha_opt.update(grads, opt_state)
        new_log_alpha = optax.apply_updates(log_alpha, updates)
        new_log_alpha = jnp.maximum(new_log_alpha, jnp.log(ALPHA_MIN))
        return new_log_alpha, new_opt_state

    @jax.jit
    def update_targets(q_params_stacked, target_q_params):
        return jax.tree.map(lambda t, q: t * (1 - TAU) + q * TAU, target_q_params, q_params_stacked)

    # ── SPG functions ────────────────────────────────────────────────────────
    @jax.jit
    def spg_find_best(actor_params, q_params_stacked, obs, actions, key):
        B = obs.shape[0]
        S = SPG_SAMPLES

        _, _, actor_mean, actor_std = actor.apply(actor_params, obs, key)
        actor_action = jnp.tanh(actor_mean)
        q_actor = q_min(q_params_stacked, obs, actor_action)

        best_actions = actor_action
        best_q = q_actor

        q_replay = q_min(q_params_stacked, obs, actions)
        better = q_replay > best_q
        best_actions = jnp.where(better[:, None], actions, best_actions)
        best_q = jnp.where(better, q_replay, best_q)

        key, sample_key = jax.random.split(key)
        noise = jax.random.normal(sample_key, (B, S, action_dim)) * jnp.clip(actor_std[:, None, :], 0.01)
        all_samples = jnp.clip(best_actions[:, None, :] + noise, -1.0, 1.0)

        obs_exp = jnp.broadcast_to(obs[:, None, :], (B, S, obs_dim)).reshape(B * S, -1)
        act_flat = all_samples.reshape(B * S, -1)
        q_all_min = q_min(q_params_stacked, obs_exp, act_flat).reshape(B, S)

        best_idx = jnp.argmax(q_all_min, axis=1)
        best_sample_actions = all_samples[jnp.arange(B), best_idx]
        q_best_sample = q_all_min[jnp.arange(B), best_idx]

        sample_better = q_best_sample > best_q
        best_actions = jnp.where(sample_better[:, None], best_sample_actions, best_actions)
        best_q = jnp.where(sample_better, q_best_sample, best_q)

        improved = best_q > q_actor
        return best_actions, improved, q_actor, best_q

    @jax.jit
    def spg_actor_step(actor_params, q_params_stacked, log_alpha, obs,
                       best_actions, improved, key, opt_state):
        def loss_fn(actor_params):
            alpha = jnp.exp(log_alpha)
            action, log_prob, mean, _ = actor.apply(actor_params, obs, key)
            actor_squashed = jnp.tanh(mean)
            mse = jnp.sum((actor_squashed - best_actions) ** 2, axis=-1)
            spg_loss = jnp.mean(mse * improved)
            return spg_loss + jnp.mean(alpha * log_prob), log_prob
        (a_loss, log_prob), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_params)
        updates, new_opt_state = actor_opt.update(grads, opt_state)
        new_actor_params = optax.apply_updates(actor_params, updates)
        return new_actor_params, new_opt_state, a_loss, log_prob

    # ── Main loop ────────────────────────────────────────────────────────────
    grad_steps = 0
    env_steps = 0
    t0 = time.time()

    print(f"Training {TASK}, {TOTAL_STEPS} steps, {N_CRITICS} critics, "
          f"SPG={SPG_ENABLED}({SPG_SAMPLES}), {NUM_ENVS} envs", flush=True)
    print("JIT compiling env...", flush=True)
    # Warmup JIT
    _test_state = env.reset(jax.random.PRNGKey(0))
    _test_state = env.step(_test_state, jnp.zeros(action_dim))
    print("Starting training...", flush=True)

    while env_steps < TOTAL_STEPS:
        # Collect experience from vectorized envs
        rng, act_key = jax.random.split(rng)
        obs_batch = env_states.obs

        if env_steps < WARMUP_STEPS:
            rng, noise_key = jax.random.split(rng)
            actions = jax.random.uniform(noise_key, (NUM_ENVS, action_dim), minval=-1.0, maxval=1.0)
        else:
            actions, _, _, _ = actor.apply(actor_params, obs_batch, act_key)

        # Step all envs (with action repeat)
        total_rewards = jnp.zeros(NUM_ENVS)
        next_states = env_states
        for _ in range(ACTION_REPEAT):
            next_states = v_step(next_states, actions)
            total_rewards = total_rewards + next_states.reward
        next_obs = next_states.obs
        dones = next_states.done

        # Add to buffer
        buffer.add(
            np.array(obs_batch), np.array(actions),
            np.array(total_rewards), np.array(next_obs), np.array(dones)
        )
        env_steps += NUM_ENVS * ACTION_REPEAT

        # Auto-reset: playground envs auto-reset on done, so just use next_states
        # (MJX envs reset automatically when done=True via the wrapper)
        env_states = next_states

        # Training: multiple grad steps per env collection to match PyTorch UTD
        # PyTorch: 1 grad step per ACTION_REPEAT env steps (1 env)
        # JAX: NUM_ENVS * ACTION_REPEAT env steps per collection → need NUM_ENVS grad steps
        if buffer.size >= BATCH_SIZE and env_steps >= WARMUP_STEPS:
          for _ in range(NUM_ENVS):
            b_obs, b_act, b_rew, b_nobs, b_done = buffer.sample(BATCH_SIZE)

            # Critic update
            rng, critic_key = jax.random.split(rng)
            q_params_stacked, critic_opt_state, c_loss, q1v = critic_step(
                q_params_stacked, target_q_params, actor_params, log_alpha,
                b_obs, b_act, b_rew, b_nobs, b_done, critic_key, critic_opt_state)

            grad_steps += 1

            # Actor update (delayed)
            if grad_steps % ACTOR_DELAY == 0:
                rng, actor_key = jax.random.split(rng)

                if SPG_ENABLED:
                    best_actions, improved, q_actor, best_q = spg_find_best(
                        actor_params, q_params_stacked, b_obs, b_act, actor_key)
                    rng, spg_key = jax.random.split(rng)
                    actor_params, actor_opt_state, a_loss, log_prob = spg_actor_step(
                        actor_params, q_params_stacked, log_alpha, b_obs,
                        best_actions, improved.astype(jnp.float32), spg_key, actor_opt_state)
                else:
                    actor_params, actor_opt_state, a_loss, log_prob = actor_step(
                        actor_params, q_params_stacked, log_alpha, b_obs,
                        actor_key, actor_opt_state)

                # Alpha update
                log_alpha, alpha_opt_state = alpha_step(log_alpha, log_prob, alpha_opt_state)

            # Target update
            target_q_params = update_targets(q_params_stacked, target_q_params)

        # Logging
        if env_steps % 20000 < NUM_ENVS * ACTION_REPEAT and env_steps >= 20000:
            elapsed = time.time() - t0
            sps = env_steps / elapsed
            alpha_val = float(jnp.exp(log_alpha))
            print(f"[{TASK}] step={env_steps} gs={grad_steps} "
                  f"alpha={alpha_val:.4f} sps={sps:.0f} "
                  f"elapsed={elapsed:.0f}s", flush=True)

    # ── Evaluation ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s ({env_steps/elapsed:.0f} sps), "
          f"{grad_steps} grad steps", flush=True)

    # Eval: vectorized rollout
    @jax.jit
    def eval_step(state, actor_params, key):
        action, _, _, _ = actor.apply(actor_params, state.obs, key)
        return env.step(state, action)

    print("Evaluating...", flush=True)
    eval_returns = []
    for ep in range(EVAL_EPISODES):
        rng, eval_key = jax.random.split(rng)
        state = env.reset(eval_key)
        ep_return = 0.0
        for _ in range(1000):
            state = eval_step(state, actor_params, eval_key)
            ep_return += float(state.reward)
            if state.done:
                break
        eval_returns.append(ep_return)

    mean_ret = np.mean(eval_returns)
    std_ret = np.std(eval_returns)
    total_time = time.time() - t0
    print(f"Eval ({EVAL_EPISODES} eps): {mean_ret:.1f} +/- {std_ret:.1f}")
    print(f"training_seconds: {elapsed:.1f}")
    print(f"total_seconds: {total_time:.1f}")
    print(f"{TASK.lower()}_return: {mean_ret:.2f}")


if __name__ == "__main__":
    main()
