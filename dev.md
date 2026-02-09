# Development Notes

## Optional: Tensor Shape Annotations with jaxtyping + beartype

**Status:** Not yet implemented — nice-to-have for documentation & runtime shape checking.

### What

[jaxtyping](https://github.com/patrick-kidger/jaxtyping) provides shape-annotated tensor types that work with PyTorch. Combined with [beartype](https://github.com/beartype/beartype), it validates shapes at runtime (e.g. during tests).

### Why

- **Self-documenting shapes** — `Float[Tensor, "batch obs_dim"]` beats `torch.Tensor` + a comment
- **Catches shape bugs at call boundaries** — mismatched dimensions fail immediately with a clear error
- **Named dimensions** — `"batch"` must match across all arguments, enforcing consistency
- **Zero production overhead** — checking is only active when decorated with `@jaxtyped`

### Example (SquashedGaussianActor)

```python
from jaxtyping import Float
from torch import Tensor

# Shapes are part of the signature — no comments needed
def forward(
    self, obs: Float[Tensor, "batch obs_dim"]
) -> tuple[Float[Tensor, "batch action_dim"], Float[Tensor, " batch"]]:
    ...
```

### Where to apply

Priority files (core data flow, most shape-sensitive):

1. `roboro/core/types.py` — `Batch` fields (`obs`, `actions`, `rewards`, etc.)
2. `roboro/actors/*.py` — `act()` and `forward()` signatures
3. `roboro/critics/*.py` — `forward()` signatures (discrete vs continuous shapes)
4. `roboro/updates/*.py` — local variables in `update()` for clarity
5. `roboro/nn/blocks.py` — `MLPBlock.forward()`

Lower priority (less shape-sensitive):

6. `roboro/data/replay_buffer.py`
7. `roboro/encoders/*.py`
8. `roboro/training/trainer.py`

### Install

```bash
pip install jaxtyping beartype
```

### Runtime checking (optional, for tests)

To enable runtime shape validation during testing, decorate functions:

```python
from beartype import beartype
from jaxtyping import jaxtyped

@jaxtyped(typechecker=beartype)
def forward(self, obs: Float[Tensor, "batch obs_dim"]) -> Float[Tensor, "batch n_actions"]:
    ...
```

Or apply globally in `conftest.py` for test runs only.

### Notes

- `from __future__ import annotations` was removed from the codebase (Python 3.12 doesn't need it), which unblocks jaxtyping's runtime inspection.
- jaxtyping works purely as documentation even without beartype — shapes show up in IDE hover and function signatures.
- The `" batch"` syntax (leading space) means a single named dimension; `"batch dim"` means two dimensions.
