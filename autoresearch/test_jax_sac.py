"""Quick benchmark: Brax SAC on MuJoCo Playground CheetahRun via MJX."""
import time
import jax
import jax.numpy as jnp
from brax.training.agents.sac import train as sac_train
from brax import envs
from brax.training import types
import mujoco_playground as mp

print(f"JAX devices: {jax.devices()}")

env = mp.dm_control_suite.load('CheetahRun')
print(f"CheetahRun: obs={env.observation_size}, act={env.action_size}")

eval_returns = []
def progress_fn(step, metrics):
    ret = metrics.get('eval/episode_reward', 0)
    print(f"  step={step:>8d} eval_return={float(ret):.1f} "
          f"sps={metrics.get('training/sps', 0):.0f}")
    eval_returns.append((step, float(ret)))

t0 = time.time()
make_policy, params, metrics = sac_train.train(
    environment=env,
    num_timesteps=100_000,
    episode_length=1000,
    num_envs=64,
    num_eval_envs=16,
    learning_rate=3e-4,
    discounting=0.99,
    batch_size=256,
    seed=42,
    num_evals=5,
    min_replay_size=1000,
    max_replay_size=100_000,
    grad_updates_per_step=1,
    normalize_observations=True,
    reward_scaling=1.0,
    tau=0.005,
    wrap_env=True,
    wrap_env_fn=mp.wrapper.wrap_for_brax_training,
    progress_fn=progress_fn,
)
elapsed = time.time() - t0

print(f"\nDone in {elapsed:.1f}s")
print(f"Final eval return: {float(metrics.get('eval/episode_reward', 0)):.1f}")
print(f"Steps/sec: {100_000/elapsed:.0f}")
for step, ret in eval_returns:
    print(f"  {step}: {ret:.1f}")
