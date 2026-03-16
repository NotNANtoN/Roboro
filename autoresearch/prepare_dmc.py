"""Autoresearch DMControl: fixed evaluation harness and task specifications.

DO NOT MODIFY — this file is read-only for the autoresearch agent.
The agent only modifies train_dmc.py.
"""

import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
from gymnasium import spaces
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ═════════════════════════════════════════════════════════════════════════════
# MINIMAL DMC → GYMNASIUM WRAPPER (avoids shimmy/labmaze dependency)
# ═════════════════════════════════════════════════════════════════════════════
class DMCGymWrapper(gym.Env):
    """Thin wrapper: dm_control env → Gymnasium API with flat obs."""

    def __init__(self, domain: str, task: str):
        from dm_control import suite
        self._env = suite.load(domain, task)
        self._domain = domain
        self._task = task

        obs_spec = self._env.observation_spec()
        obs_dim = int(sum(np.prod(v.shape) for v in obs_spec.values()))
        act_spec = self._env.action_spec()

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float64)
        self.action_space = spaces.Box(
            low=act_spec.minimum.astype(np.float32),
            high=act_spec.maximum.astype(np.float32),
            dtype=np.float32,
        )

    def _flatten_obs(self, time_step) -> np.ndarray:
        return np.concatenate([v.flatten() for v in time_step.observation.values()])

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self._env = __import__('dm_control').suite.load(
                self._domain, self._task, task_kwargs={'random': seed}
            )
        time_step = self._env.reset()
        return self._flatten_obs(time_step).astype(np.float32), {}

    def step(self, action):
        time_step = self._env.step(action)
        obs = self._flatten_obs(time_step).astype(np.float32)
        reward = float(time_step.reward or 0.0)
        terminated = time_step.last()
        return obs, reward, terminated, False, {}

    def close(self):
        self._env.close() if hasattr(self._env, 'close') else None


def make_dmc_env(env_id: str) -> DMCGymWrapper:
    """Parse 'domain-task' format and create wrapped env."""
    parts = env_id.split("-")
    domain, task = parts[0], "-".join(parts[1:])
    return DMCGymWrapper(domain, task)


@dataclass(frozen=True)
class TaskSpec:
    """Fixed specification for an autoresearch task."""

    env_id: str
    step_budget: int
    eval_episodes: int
    eval_seed: int
    max_return: float
    min_return: float
    description: str


TASKS = {
    "cheetah": TaskSpec(
        env_id="cheetah-run",
        step_budget=100_000,
        eval_episodes=20,
        eval_seed=10_000,
        max_return=1000.0,
        min_return=0.0,
        description="SAC on cheetah-run (6D action, fast locomotion)",
    ),
    "humanoid": TaskSpec(
        env_id="humanoid-walk",
        step_budget=100_000,
        eval_episodes=10,
        eval_seed=10_000,
        max_return=1000.0,
        min_return=0.0,
        description="SAC on humanoid-walk (21D action, high-dim balance + walk)",
    ),
}

TIME_LIMIT_SECONDS = 600  # 10 minutes total for both tasks combined

PolicyFn = Callable[[np.ndarray], np.ndarray]


# ═════════════════════════════════════════════════════════════════════════════
# TIME LIMIT ENFORCEMENT
# ═════════════════════════════════════════════════════════════════════════════
class TimeLimitExceeded(SystemExit):
    """Raised when the global time limit is exceeded."""

    def __init__(self, elapsed: float, limit: float) -> None:
        super().__init__(
            f"\n*** TIME LIMIT EXCEEDED: {elapsed:.1f}s > {limit}s ***\n"
            "Training took too long. Optimize for wall-clock time."
        )


_start_time: float | None = None


def start_timer() -> None:
    """Call once at the beginning of train_dmc.py's __main__ block."""
    global _start_time
    _start_time = time.time()

    def _alarm_handler(signum: int, frame: object) -> None:
        elapsed = time.time() - _start_time if _start_time else 0
        raise TimeLimitExceeded(elapsed, TIME_LIMIT_SECONDS)

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TIME_LIMIT_SECONDS + 5)  # hard kill 5s after limit


def check_time() -> float:
    """Return elapsed seconds since start_timer(). Raise if over limit."""
    if _start_time is None:
        raise RuntimeError("Call start_timer() before check_time()")
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT_SECONDS:
        raise TimeLimitExceeded(elapsed, TIME_LIMIT_SECONDS)
    return elapsed


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
def evaluate(
    env_id: str,
    policy_fn: PolicyFn,
    n_episodes: int = 20,
    seed: int = 10_000,
) -> float:
    """Ground truth evaluation — run n greedy episodes, return mean total reward."""
    env = make_dmc_env(env_id)
    rewards: list[float] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)
    env.close()
    return sum(rewards) / len(rewards)


def normalize_return(task: TaskSpec, eval_return: float) -> float:
    """Normalize eval_return to [0, 1] range for cross-env comparison."""
    return max(0.0, (eval_return - task.min_return) / (task.max_return - task.min_return))


def print_summary(
    cheetah_return: float,
    humanoid_return: float,
    training_seconds: float,
    total_seconds: float,
    num_params_cheetah: int,
    num_params_humanoid: int,
    device: str = "cpu",
) -> None:
    """Print standardized summary for autoresearch result parsing."""
    ch = TASKS["cheetah"]
    hu = TASKS["humanoid"]

    cheetah_norm = normalize_return(ch, cheetah_return)
    humanoid_norm = normalize_return(hu, humanoid_return)
    score = (cheetah_norm + humanoid_norm) / 2.0

    print("\n---")
    print(f"score:              {score:.4f}")
    print(f"cheetah_return:     {cheetah_return:.2f}")
    print(f"cheetah_norm:       {cheetah_norm:.4f}")
    print(f"humanoid_return:    {humanoid_return:.2f}")
    print(f"humanoid_norm:      {humanoid_norm:.4f}")
    print(f"training_seconds:   {training_seconds:.1f}")
    print(f"total_seconds:      {total_seconds:.1f}")
    print(f"num_params_cheetah: {num_params_cheetah}")
    print(f"num_params_humanoid:{num_params_humanoid}")
    print(f"device:             {device}")
