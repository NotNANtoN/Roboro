"""Autoresearch: fixed evaluation harness and task specifications.

DO NOT MODIFY — this file is read-only for the autoresearch agent.
The agent only modifies train.py.
"""

import os
import sys
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
    "cartpole": TaskSpec(
        env_id="CartPole-v1",
        step_budget=50_000,
        eval_episodes=20,
        eval_seed=10_000,
        max_return=500.0,
        min_return=0.0,
        description="Discrete Q-learning on CartPole-v1",
    ),
    "pendulum": TaskSpec(
        env_id="Pendulum-v1",
        step_budget=50_000,
        eval_episodes=20,
        eval_seed=10_000,
        max_return=0.0,
        min_return=-1600.0,
        description="Continuous actor-critic on Pendulum-v1",
    ),
}

TIME_LIMIT_SECONDS = 300  # 5 minutes total for both tasks combined

PolicyFn = Callable[[np.ndarray], int | np.ndarray]


def evaluate(
    env_id: str,
    policy_fn: PolicyFn,
    n_episodes: int = 20,
    seed: int = 10_000,
) -> float:
    """Ground truth evaluation — run n greedy episodes, return mean total reward.

    Args:
        env_id: Gymnasium environment ID.
        policy_fn: Maps numpy observation -> action (int for discrete, ndarray for continuous).
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
    return (eval_return - task.min_return) / (task.max_return - task.min_return)


def print_summary(
    cartpole_return: float,
    pendulum_return: float,
    training_seconds: float,
    total_seconds: float,
    num_params_q: int,
    num_params_ac: int,
    device: str = "cpu",
) -> None:
    """Print standardized summary for autoresearch result parsing."""
    cp = TASKS["cartpole"]
    pend = TASKS["pendulum"]

    cartpole_norm = normalize_return(cp, cartpole_return)
    pendulum_norm = normalize_return(pend, pendulum_return)
    score = (cartpole_norm + pendulum_norm) / 2.0

    print("\n---")
    print(f"score:              {score:.4f}")
    print(f"cartpole_return:    {cartpole_return:.2f}")
    print(f"cartpole_norm:      {cartpole_norm:.4f}")
    print(f"pendulum_return:    {pendulum_return:.2f}")
    print(f"pendulum_norm:      {pendulum_norm:.4f}")
    print(f"training_seconds:   {training_seconds:.1f}")
    print(f"total_seconds:      {total_seconds:.1f}")
    print(f"num_params_q:       {num_params_q}")
    print(f"num_params_ac:      {num_params_ac}")
    print(f"device:             {device}")
