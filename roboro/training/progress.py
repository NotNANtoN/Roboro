"""Progress tracking — keeps the training loop clean."""

from typing import Self

from tqdm import tqdm

from roboro.actors.base import BaseActor


class ProgressTracker:
    """Wraps tqdm and accumulates episode / eval statistics.

    Usage::

        with ProgressTracker(total_steps=50_000) as tracker:
            for step in range(1, total_steps + 1):
                ...
                tracker.step(step, loss=loss)
                if done:
                    tracker.log_episode(ep_reward)
                if step % eval_interval == 0:
                    tracker.log_eval(mean_eval_reward)
    """

    def __init__(
        self,
        total_steps: int,
        *,
        show: bool = True,
        log_interval: int = 500,
    ) -> None:
        self._log_interval = log_interval
        self._pbar: tqdm[int] | None = (
            tqdm(
                total=total_steps,
                desc="Training",
                dynamic_ncols=True,
            )
            if show
            else None
        )

        self._episode_rewards: list[float] = []
        self._eval_rewards: list[float] = []

    # ── public API ───────────────────────────────────────────────────────────

    def step(
        self,
        step: int,
        *,
        loss: float = float("nan"),
        metrics: dict[str, float] | None = None,
        actor: BaseActor | None = None,
        buf_size: int = 0,
    ) -> None:
        """Advance one step and optionally update the display."""
        if self._pbar is None:
            return
        self._pbar.update(1)

        if step % self._log_interval != 0:
            return

        parts: list[str] = []

        # Episode count + recent episode reward (rolling window of 20)
        if self._episode_rewards:
            recent = self._episode_rewards[-20:]
            parts.append(f"ep={len(self._episode_rewards)}  ep_r={sum(recent) / len(recent):.0f}")

        # Latest eval
        if self._eval_rewards:
            parts.append(f"eval={self._eval_rewards[-1]:.0f}")

        parts.append(f"loss={loss:.4f}")

        # Add custom metrics if provided
        if metrics:
            for k, v in metrics.items():
                if "dynamics" in k or "value" in k or "policy" in k:
                    short_name = k.replace("loss/", "").replace("dynamics_", "dyn_")
                    parts.append(f"{short_name}={v:.4f}")

        # Optional exploration info
        if actor is not None and hasattr(actor, "epsilon"):
            parts.append(f"ε={actor.epsilon:.2f}")

        if buf_size:
            parts.append(f"buf={buf_size}")

        self._pbar.set_postfix_str("  ".join(parts))

    def log_episode(self, reward: float) -> None:
        """Record a completed training episode."""
        self._episode_rewards.append(reward)

    def log_eval(self, reward: float) -> None:
        """Record an evaluation result."""
        self._eval_rewards.append(reward)

    def close(self) -> None:
        if self._pbar is not None:
            self._pbar.close()

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
