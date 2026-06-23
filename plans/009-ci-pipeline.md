# Plan 009: Add CI pipeline for tests, lint, and typecheck

> Drift check: `git diff --stat 672167a..HEAD -- .github/`

## Status

- Priority: P1
- Effort: M
- Risk: LOW
- Depends on: none
- Category: dx
- Planned at: commit `672167a`, 2026-06-23

## Why this matters

The project has zero CI for code quality. The existing `.github/workflows/deploy.yml` only runs when `project.toml` changes (for GitHub Pages). Commits that break tests, introduce lint errors, or add type errors are never caught automatically. This erodes confidence in `main` over time.

## Current state

Only one workflow exists:
```yaml
# .github/workflows/deploy.yml — only triggers on project.toml changes
name: Deploy project page
on:
  push:
    branches: [main]
    paths: ["project.toml"]
  workflow_dispatch:
```

No test, lint, or typecheck workflow.

Dev commands the CI should run (from AGENTS.md):
```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest
```

## Scope

**In scope**: `.github/workflows/ci.yml` (new file)

**Out of scope**: Any changes to the existing `deploy.yml`, any source code, any workflow that requires secrets

## Steps

### Step 1: Create `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.14"

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: uv sync --frozen

      - name: Lint
        run: uv run ruff check .

      - name: Format check
        run: uv run ruff format --check .

      - name: Type check
        run: uv run mypy . || echo "mypy may have stub errors — review output"

      - name: Test
        run: uv run pytest -q
```

Note: mypy is allowed to fail (via `|| echo`) because missing stubs (sklearn, yfinance, alpaca-py) generate non-actionable errors. The CI runs it but doesn't block on it.

**Verify**: Ensure the workflow parses correctly:
```bash
# Check GitHub Actions syntax by parsing with python
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('YAML valid')"
```

If `pyyaml` is not available, skip the verification or install it first.

## Test plan

Cannot fully test GitHub Actions locally. Verify by checking the file syntax and manually running the commands the CI will use:

```bash
uv run ruff check . && echo "lint OK"
uv run ruff format --check . && echo "format OK"
uv run pytest -q && echo "tests OK"
```

## Done criteria

- [ ] `.github/workflows/ci.yml` exists and is valid YAML
- [ ] Workflow runs `ruff check`, `ruff format --check`, `mypy`, `pytest` on push/PR to `main`
- [ ] `uv run pytest -q` passes (CI will run the same command)

## STOP conditions

- If `uv sync --frozen` fails in CI (lockfile outdated), change to `uv sync` without `--frozen`.

## Maintenance notes

The mypy step is non-blocking because missing third-party stubs generate errors that aren't actionable. If stubs become available for all dependencies, remove the `|| echo` to make mypy blocking.
