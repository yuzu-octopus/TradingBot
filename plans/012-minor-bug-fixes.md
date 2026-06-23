# Plan 012: Minor bug fixes — NaN clipping, RSI zero-divide, gradient accumulation

> Drift check: `git diff --stat 672167a..HEAD -- src/features.py training/train.py`

## Status

- Priority: P3
- Effort: S
- Risk: LOW
- Depends on: none
- Category: bug
- Planned at: commit `672167a`, 2026-06-23

## Why this matters

Three independent low-severity bugs: (1) NaN targets are filled with 0 before quantile clipping, biasing the clip bounds if many stocks have missing data on the same day; (2) RSI returns 0 (extremely oversold) when both gain and loss are zero instead of 50 (neutral); (3) the last gradient accumulation batch is silently dropped when `len(loader) % grad_accum_steps != 0`.

## Current state

**F11** — NaN fill before quantile:
```python
# features.py:131-136
targets[np.isnan(targets)] = 0.0  # line 131 — NaN filled
if clip_extreme:
    flat = targets.flatten()
    lower = np.quantile(flat, LABEL_CLIP_PCT)  # lines 134-135 — quantile on NaN-filled data
    upper = np.quantile(flat, 1 - LABEL_CLIP_PCT)
```

**F12** — RSI zero-divide:
```python
# features.py:20-25
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)  # both zero → NaN → 0 (wrong)
```

**F15** — Gradient accumulation drops partial batch:
```python
# train.py:237
if (step + 1) % grad_accum_steps == 0:  # only fires at exact multiples
```

## Scope

**In scope**: `src/features.py`, `training/train.py`

**Out of scope**: Tests (covered by plan 010 or trust existing tests), any other file

## Steps

### Step 1: Fix NaN-clip ordering in `build_targets`

Swap the NaN fill and quantile compute:
```python
# features.py:131-136
if clip_extreme:
    flat = targets.flatten()
    # Compute quantiles EXCLUDING NaN positions
    valid = flat[~np.isnan(flat)]
    lower = np.quantile(valid, LABEL_CLIP_PCT)
    upper = np.quantile(valid, 1 - LABEL_CLIP_PCT)
    targets = np.clip(targets, lower, upper)
targets[np.isnan(targets)] = 0.0  # fill AFTER clipping
```

### Step 2: Fix RSI zero-divide

Add a guard for the zero-gain-zero-loss case:
```python
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    # When both gain and loss are zero, RSI should be 50 (neutral)
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)  # neutral when no movement
    return rsi
```

### Step 3: Fix gradient accumulation for final partial batch

```python
# train.py:237
if (step + 1) % grad_accum_steps == 0 or step == len(train_loader) - 1:
```

**Verify**: `uv run ruff check . && uv run ruff format . && uv run pytest -q` → all pass

Manual RSI verification:
```bash
uv run python -c "
import pandas as pd
from src.features import compute_rsi
s = pd.Series([100]*20)  # flat prices → RSI should be 50
rsi = compute_rsi(s)
print(f'RSI for flat prices: {rsi.iloc[-1]:.1f}')  # should be 50.0
"
```

## Done criteria

- [ ] `uv run ruff check .` exits 0
- [ ] `uv run pytest -q` → 60 passed
- [ ] RSI for flat price series returns 50.0
- [ ] NaN values in targets are filled AFTER quantile clipping, not before
- [ ] Final partial gradient accumulation batch is no longer silently dropped

## STOP conditions

- If `targets.flatten()` after the swap produces different results due to NaN positions in the flattened tensor, adjust the `valid = flat[~np.isnan(flat)]` to handle multi-dimensional arrays correctly.

## Maintenance notes

The RSI fix changes the output for edge-case inputs (flat prices). Any downstream code that depends on RSI being 0 in these cases will break — but 0 is incorrect, so this is a correctness fix.
