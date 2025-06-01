# Development Setup Guide

## Pre-commit Hooks

This repository uses pre-commit hooks to maintain code quality. The hooks include:

- **Basic checks**: trailing whitespace, end-of-file-fixer, yaml validation
- **isort**: Import sorting (compatible with black)
- **black**: Code formatting
- **ruff**: Fast Python linting and formatting
- **pyright**: Type checking

### Installation

```bash
# Install pre-commit (if not already installed)
pip install pre-commit

# Install the git hooks
pre-commit install
```

### Usage

Pre-commit will run automatically on `git commit`. To run manually:

```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files path/to/file.py

# Skip hooks for emergency commits
git commit --no-verify -m "emergency commit"
```

### Common Issues and Solutions

#### Ruff Issues
Most ruff issues are automatically fixable:
```bash
# Fix all auto-fixable ruff issues
ruff check --fix .

# Use unsafe fixes for more aggressive fixes
ruff check --fix --unsafe-fixes .
```

#### Type Issues (Pyright)
Common patterns for ML/RL code:

1. **Optional parameters**: Use `Optional[Type]` or `Type | None`
2. **Config objects**: Add type hints for Hydra configs
3. **Tensor operations**: Be explicit about tensor vs numpy types

#### Variable Naming (N806)
Use lowercase for variables even if they represent classes:
```python
# Instead of: PolicyClass = SomePolicy
# Use: policy_class = SomePolicy
```

### Configuration

- **pyproject.toml**: Contains tool configurations
- **.pre-commit-config.yaml**: Pre-commit hook setup
- Exclude patterns are configured for ML artifacts (checkpoints, outputs, mlruns)

### Tips

1. Run `pre-commit run --all-files` before major commits
2. Use `ruff check --fix .` to auto-fix most linting issues
3. Type checking is set to "basic" mode - can be made stricter in pyproject.toml
4. Black line length is set to 88 characters
