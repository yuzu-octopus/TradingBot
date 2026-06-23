# Plan 003: Document that msrr_loss is a heuristic, not canonical Sharpe optimization

> Drift check: `git diff --stat 672167a..HEAD -- training/train.py README.md AGENTS.md`

## Status

- Priority: P2
- Effort: S
- Risk: LOW
- Depends on: none
- Category: docs
- Planned at: commit `672167a`, 2026-06-23

## Why this matters

The loss function named `msrr_loss` is `((1 - portfolio_return) ** 2).mean()` — MSE around the constant 1, ignoring variance entirely. The README and AGENTS.md describe it as "Directly optimizes portfolio Sharpe" which is false. A user relying on this description would expect the model to maximize risk-adjusted returns when it's actually minimizing squared error of portfolio return around 1.0. This misleads users about what the model optimizes.

## Current state

```python
# training/train.py:57-59
def msrr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    portfolio_return = (pred * target).sum(dim=1)
    return ((1 - portfolio_return) ** 2).mean()
```

README.md line 145: `**MSRR** — outputs portfolio weights directly. Loss: `(1 - w'R)²`. Optimizes portfolio Sharpe.`

## Scope

**In scope**: `training/train.py` (rename function + add docstring), `README.md` (correct description), `AGENTS.md` (correct description)

**Out of scope**: Implementing a true Sharpe-optimizing loss (that's a research task, not a doc fix). Any other file.

## Steps

### Step 1: Rename the function

Change `msrr_loss` to `portfolio_mse_loss` in `training/train.py:57-59` and add a docstring:

```python
def portfolio_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalizes portfolio return deviation from 1.0.

    Heuristic approximation: encourages prediction*return correlation.
    Does NOT directly optimize Sharpe ratio (variance is ignored).
    """
    portfolio_return = (pred * target).sum(dim=1)
    return ((1 - portfolio_return) ** 2).mean()
```

### Step 2: Update all references

- `training/train.py:186` — update import: `from training.train import listnet_loss, margin_ranking_loss, portfolio_mse_loss`
- `training/pretrain.py:15` — same import update
- `training/train.py:147` — `"msrr"` loss mode still uses this function. Change the variable or the mapping key. Keep `"msrr"` as CLI arg for backwards compatibility, just point it to the renamed function.

### Step 3: Update README.md

Replace the MSRR description with accurate text:

```
**Portfolio MSE** — penalizes deviation of the portfolio return from 1.0.
Encourages the model to produce scores that correlate with next-day returns.
A heuristic; does not optimize risk-adjusted returns like true MSRR would.
```

**Verify**: `uv run ruff check . && uv run ruff format . && uv run pytest -q` → all pass

## Test plan

Update `tests/test_training.py` if it references the old function name. Run `grep -rn "msrr_loss" src/ training/ tests/` to confirm zero remaining references after rename.

## Done criteria

- [ ] `grep -rn "msrr_loss" src/ training/ tests/` returns 0 matches
- [ ] New function `portfolio_mse_loss` has docstring explaining it's a heuristic
- [ ] README.md and AGENTS.md no longer claim it optimizes Sharpe
- [ ] `uv run pytest -q` → 60 passed

## STOP conditions

- If tests fail because they depend on the old function name, update the test imports and continue.

## Maintenance notes

If someone implements a true differentiable Sharpe loss (e.g., `-(mean(ret) / std(ret))`), add it as a new function under a new CLI flag (`--loss sharpe`) rather than modifying `portfolio_mse_loss`.
