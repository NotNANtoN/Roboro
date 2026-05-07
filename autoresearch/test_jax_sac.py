"""Brax SAC on MJX CheetahRun — matching PyTorch hyperparameters."""
import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

# Patch deprecated JAX APIs for Brax compatibility
if not hasattr(jax, 'device_put_replicated'):
    def _device_put_replicated(x, devices):
        return jax.device_put(jnp.stack([x] * len(devices)),
                              jax.sharding.NamedSharding(
                                  jax.sharding.Mesh(np.array(devices), ('x',)),
                                  jax.sharding.PartitionSpec('x')))
    jax.device_put_replicated = _device_put_replicated

from brax.training.agents.sac import train as sac_train
from brax.training.agents.sac import networks as sac_networks
import mujoco_playground as mp

print(f"JAX devices: {jax.devices()}")

env = mp.dm_control_suite.load('CheetahRun', config_overrides={'impl': 'jax'})
print(f"CheetahRun: obs={env.observation_size}, act={env.action_size}")

# Custom network factory matching our PyTorch config:
# HIDDEN_DIM=256, N_LAYERS=2, LayerNorm=True
def custom_network_factory(observation_size, action_size, preprocess_observations_fn):
    return sac_networks.make_sac_networks(
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=(256, 256),
        activation=jax.nn.relu,
        policy_network_layer_norm=True,
        q_network_layer_norm=True,
    )

eval_returns = []
def progress_fn(step, metrics):
    ret = metrics.get('eval/episode_reward', 0)
    sps = metrics.get('training/sps', 0)
    alpha = metrics.get('training/alpha', 0)
    cl = metrics.get('training/critic_loss', 0)
    print(f"  step={step:>8d}  ret={float(ret):>7.1f}  "
          f"alpha={float(alpha):.4f}  cl={float(cl):.3f}  sps={float(sps):.0f}")
    eval_returns.append((step, float(ret)))

# Matching our PyTorch config:
# LR=5e-3, GAMMA=0.995, ACTION_REPEAT=2, BATCH_SIZE=128
# INIT_ALPHA=0.1, TAU=0.005, WARMUP_STEPS=512, BUFFER=100k
t0 = time.time()
make_policy, params, metrics = sac_train.train(
    environment=env,
    num_timesteps=100_000,
    episode_length=500,        # 1000 / action_repeat=2 = 500 decision steps
    action_repeat=2,
    num_envs=64,               # Vectorized envs for data collection
    num_eval_envs=16,
    learning_rate=5e-3,
    discounting=0.995,
    batch_size=128,
    seed=42,
    num_evals=6,
    min_replay_size=512,
    max_replay_size=100_000,
    grad_updates_per_step=1,   # UTD=1
    normalize_observations=False,
    reward_scaling=1.0,
    tau=0.005,
    wrap_env=True,
    wrap_env_fn=mp.wrapper.wrap_for_brax_training,
    network_factory=custom_network_factory,
    progress_fn=progress_fn,
)
elapsed = time.time() - t0

print(f"\nDone in {elapsed:.1f}s")
print(f"Final eval return: {float(metrics.get('eval/episode_reward', 0)):.1f}")
print(f"Effective steps/sec: {100_000/elapsed:.0f}")
