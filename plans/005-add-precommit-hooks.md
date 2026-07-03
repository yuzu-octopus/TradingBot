# Plan 005: Add pre-commit hooks for ruff check + format

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result.
>
> **Drift check (run first)**: `git diff --stat 8473bcf..HEAD -- .pre-commit-config.yaml .gitignore pyproject.toml`

## Status

- **Priority**: P3
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: dx
- **Planned at**: commit `8473bcf`, 2026-06-28

## Why this matters

No local pre-commit hooks exist. Developers push formatting/lint issues and rely on CI to catch them. This wastes CI cycles and creates noisy diffs. The project already runs `ruff check` and `ruff format --check` in CI (`.github/workflows/ci.yml`); adding the same checks as pre-commit hooks catches issues before push.

## Current state

- CI runs: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy .`, `uv run pytest -q`
- No `.pre-commit-config.yaml` exists
- `pyproject.toml` has ruff config already
- `uv` is the package manager (not pip)

## Steps

### Step 1: Install pre-commit

```bash
uv add --dev pre-commit
```

### Step 2: Create `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.13
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format
```

### Step 3: Install hooks

```bash
uv run pre-commit install
```

### Step 4: Verify hooks work

```bash
uv run pre-commit run --all-files
```

Expected: ruff-check and ruff-format run successfully. If there are auto-fixable issues, they'll be fixed in-place; the user should review and commit the fixes.

### Step 5: Full validation

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

Expected: all checks passed, 79 tests pass.

## Done criteria

- [ ] `.pre-commit-config.yaml` exists with ruff hooks
- [ ] `pre-commit` in dev dependencies
- [ ] `uv run pre-commit run --all-files` exits 0
- [ ] `uv run pytest -q` exits 0

## STOP conditions

- If `ruff-pre-commit` version doesn't match the ruff version in `pyproject.toml`, use the pyproject.toml version
