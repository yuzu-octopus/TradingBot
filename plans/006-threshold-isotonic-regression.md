# Plan 006: Fix threshold optimization — replace IsotonicRegression with Platt scaling

> Drift check: `git diff --stat 672167a..HEAD -- training/threshold.py`

## Status

- Priority: P2
- Effort: M
- Risk: MED
- Depends on: none
- Category: bug
- Planned at: commit `672167a`, 2026-06-23

## Why this matters

The current threshold optimization uses `IsotonicRegression` with `np.sign(target)` (±1) as labels. This collapses continuous model scores into near-binary values before the Sharpe grid search, discarding the fine-grained ranking information the model learned. The threshold grid search then has little signal to differentiate between thresholds, producing unstable buy/sell boundaries.

## Current state

```python
# threshold.py:33-38
iso = IsotonicRegression(out_of_bounds="clip")
cal_scores[mask] = iso.fit_transform(
    flat_scores[mask], np.sign(flat_targets[mask])
)
```

The fit targets are `np.sign(flat_targets[mask])` which is ±1 or 0. The resulting calibration maps continuous scores to near-binary outputs.

## Scope

**In scope**: `training/threshold.py`

**Out of scope**: The grid search algorithm itself, any other file, tests for this change (update existing test if it references the old code)

## Steps

### Step 1: Replace IsotonicRegression with a sigmoid calibration (Platt scaling)

Replace the isotonic regression block with logistic regression:

```python
from sklearn.linear_model import LogisticRegression

# Replace isotonic calibration
cal_scores = flat_scores.copy()
if mask.sum() > 10:
    lr = LogisticRegression(C=1.0, class_weight="balanced")
    # Use raw scores as the single feature
    lr.fit(flat_scores[mask].reshape(-1, 1), (flat_targets[mask] > 0).astype(int))
    # Predict probabilities as calibrated scores
    cal_probs = lr.predict_proba(flat_scores.reshape(-1, 1))[:, 1]
    cal_scores = 2.0 * cal_probs - 1.0  # map [0,1] → [-1,1]
```

This maps scores to [-1, 1] range preserving ordering while providing calibrated probability-like outputs.

### Step 2: Remove unused import

Remove `from sklearn.isotonic import IsotonicRegression` (if it was the only use) or keep it if used elsewhere. Add `from sklearn.linear_model import LogisticRegression`.

**Verify**: `uv run ruff check . && uv run ruff format . && uv run pytest -q` → all pass

## Test plan

Manually verify threshold optimization runs:
```bash
uv run python -c "
from config import Config
from training.threshold import run_threshold_optimization
# This will fail if data/features/ doesn't exist — expected for a fresh checkout
print('Module imported OK')
"
```

Existing test `test_optimize_threshold_runs` in `tests/test_training.py` should still pass.

## Done criteria

- [ ] `uv run ruff check .` exits 0
- [ ] `uv run pytest -q` → 60 passed
- [ ] No `IsotonicRegression` import in `threshold.py`
- [ ] `LogisticRegression` is used instead in `threshold.py`

## STOP conditions

- If `LogisticRegression` with `class_weight="balanced"` and a tiny validation set (< 20 samples) raises a convergence warning, add `max_iter=1000` and continue.

## Maintenance notes

The Platt scaling approach produces smoother calibrated scores than isotonic regression, but assumes a logistic relationship between raw scores and direction accuracy. If this assumption is wrong for a particular market regime, the calibration may underperform. The original isotonic approach is more flexible but prone to overfitting.
