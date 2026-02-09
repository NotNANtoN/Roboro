"""Device selection and compilation utilities.

Default: CPU — fast for small envs, reproducible.
Use ``get_accelerator()`` when the workload benefits from GPU.

Why CPU by default?
  Small envs (CartPole, Pendulum) + small networks (128 hidden) are
  ~10x faster on CPU than MPS because MPS kernel-launch overhead
  dominates the tiny matmuls.  CUDA can be faster, but CPU is the safe
  portable default.
"""

import platform
import warnings

import torch
from torch import nn


def get_device() -> torch.device:
    """Return CPU — the safe, reproducible default.

    Use :func:`get_accelerator` to auto-select a GPU when needed.
    """
    return torch.device("cpu")


def get_accelerator() -> torch.device:
    """Return the best available accelerator: CUDA → MPS → CPU.

    Use this when the model / batch size is large enough to benefit
    from GPU parallelism (e.g. image encoders, large latent dynamics).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def maybe_compile(model: nn.Module) -> nn.Module:
    """Compile *model* with ``torch.compile`` if the platform supports it.

    ``torch.compile`` (Inductor backend) requires a working C++ toolchain.
    On macOS this is currently broken (``libc++.1.dylib`` not found), so we
    fall back to the ``aot_eager`` backend which still traces the graph but
    skips C++ code-generation.  On Linux/CUDA the full Inductor backend is
    used.

    Returns the (possibly compiled) model — always safe to call.
    """
    if platform.system() == "Darwin":
        warnings.warn(
            "torch.compile Inductor backend is broken on macOS — "
            "falling back to 'aot_eager' (graph capture only, no C++ codegen). "
            "Use Linux/CUDA for full torch.compile speedups.",
            stacklevel=2,
        )
        return torch.compile(model, backend="aot_eager")  # type: ignore[return-value]
    return torch.compile(model)  # type: ignore[return-value]
