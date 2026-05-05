"""Autoresearch: fixed evaluation harness and task specifications.

DO NOT MODIFY — this file is read-only for the autoresearch agent.
The agent only modifies train.py.
"""

import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


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
    "hopper": TaskSpec(
        env_id="Hopper-v5",
        step_budget=100_000,
        eval_episodes=20,
        eval_seed=10_000,
        max_return=3500.0,
        min_return=0.0,
        description="Continuous SAC on Hopper-v5 (3D action, balance + hop)",
    ),
    "walker": TaskSpec(
        env_id="Walker2d-v5",
        step_budget=100_000,
        eval_episodes=20,
        eval_seed=10_000,
        max_return=5000.0,
        min_return=0.0,
        description="Continuous SAC on Walker2d-v5 (6D action, balance + walk)",
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
    """Call once at the beginning of train.py's __main__ block."""
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
    """Ground truth evaluation — run n greedy episodes, return mean total reward.

    Args:
        env_id: Gymnasium environment ID.
        policy_fn: Maps numpy observation -> action (ndarray for continuous).
        n_episodes: Number of evaluation episodes.
        seed: Base seed for episode resets (episode i uses seed + i).
    """
    env = gym.make(env_id)
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
    hopper_return: float,
    walker_return: float,
    training_seconds: float,
    total_seconds: float,
    num_params_hopper: int,
    num_params_walker: int,
    device: str = "cpu",
) -> None:
    """Print standardized summary for autoresearch result parsing."""
    hop = TASKS["hopper"]
    walk = TASKS["walker"]

    hopper_norm = normalize_return(hop, hopper_return)
    walker_norm = normalize_return(walk, walker_return)
    score = (hopper_norm + walker_norm) / 2.0

    print("\n---")
    print(f"score:              {score:.4f}")
    print(f"hopper_return:      {hopper_return:.2f}")
    print(f"hopper_norm:        {hopper_norm:.4f}")
    print(f"walker_return:      {walker_return:.2f}")
    print(f"walker_norm:        {walker_norm:.4f}")
    print(f"training_seconds:   {training_seconds:.1f}")
    print(f"total_seconds:      {total_seconds:.1f}")
    print(f"num_params_hopper:  {num_params_hopper}")
    print(f"num_params_walker:  {num_params_walker}")
    print(f"device:             {device}")
