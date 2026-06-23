# Plan 004: Walk-forward fold isolation — per-fold checkpoints, test data, dead code

> Drift check: `git diff --stat 672167a..HEAD -- main.py`

## Status

- Priority: P1
- Effort: M
- Risk: MED
- Depends on: none
- Category: bug
- Planned at: commit `672167a`, 2026-06-23

## Why this matters

Three walk-forward bugs: (1) every fold overwrites the same `best.pt` — only the last fold's model survives, (2) fold test data is computed but never saved or used, (3) walk-forward with cached features triggers a wasteful full re-download. This undermines the entire walk-forward validation system.

## Current state

**F2** — Fold loop calls `run_training()` which saves to the same `config.model_save_path`:

```python
# main.py:443-459 (simplified)
for fold in range(fold_count):
    if args.walk_forward:
        rt_args = {
            "config": config,
            "train_path": f".../fold_{fold}_train.npz",
            "val_path": f".../fold_{fold}_val.npz",
        }
        run_training(**rt_args)  # saves to config.model_save_path = data/models/best.pt
```

**F1** — `prepare_walk_forward_splits` computes `test_idx` but never saves it:

```python
# main.py:52-71
test_idx = np.array([val_end < d <= test_end for d in date_objs])
folds.append((train_idx, val_idx, test_idx, f"..."))
...
for i, (tr, va, _te, _label) in enumerate(folds):
    # saves train and val only, _te is discarded
```

**F3** — When features exist and walk-forward is requested, `n_folds=0` triggers a full re-download:

```python
# main.py:424-440
if args.walk_forward and n_folds <= 1 and n_folds == 0:
    raw_data = fetch_stock_data(...)  # re-downloads 503 stocks
    features, tickers, dates = build_feature_matrix(raw_data)
```

## Scope

**In scope**: `main.py` only

**Out of scope**: `training/train.py`, `training/threshold.py`, tests

## Steps

### Step 1: Add per-fold model paths

In the walk-forward loop, pass a fold-specific save path:

```python
for fold in range(fold_count):
    if args.walk_forward:
        fold_model_path = config.model_save_path.replace(".pt", f"_fold{fold}.pt")
        config.model_save_path = fold_model_path
        rt_args = {
            "config": config,
            ...
        }
        run_training(**rt_args)
```

After the loop, keep only the last fold's path as the canonical `best.pt`, or keep all as `best_fold0.pt`, `best_fold1.pt`, etc.

### Step 2: Save fold test data (or remove dead computation)

Option A (recommended): Save test data alongside train/val:

```python
for i, (tr, va, te, _label) in enumerate(folds):
    np.savez(f".../fold_{i}_test.npz",
        features=features[te],
        targets=targets[te],
        market_state=market_state[te],
    )
```

Option B: Remove `test_idx` computation and the `te` variable entirely. Choose A if you plan to use test sets for evaluation later; B if not. Implement A.

### Step 3: Fix the `n_folds=0` cache issue

When features exist and walk-forward is requested, skip the redundant re-download. Change the condition or restructure the control flow so `prepare_data` is called when needed rather than having two code paths.

Simplest fix: In the `if args.walk_forward and n_folds <= 1 and n_folds == 0:` block, check if fold files already exist:

```python
from pathlib import Path
if args.walk_forward and n_folds <= 1:
    # Check if fold files already exist
    fold_dir = Path(config.features_path)
    existing_folds = list(fold_dir.glob("fold_*_train.npz"))
    if existing_folds:
        n_folds = len(existing_folds)
    else:
        # re-download and rebuild
        ...
```

Also fix the redundant guard `n_folds <= 1 and n_folds == 0` → `n_folds == 0`.

**Verify**: `uv run ruff check . && uv run ruff format . && uv run pytest -q` → all pass

Manually verify fold paths diverge: `uv run python -c "from config import Config; print(Config().model_save_path)"` → `data/models/best.pt`

## Test plan

Run walk-forward with `--force-features --walk-forward --max-epochs 2` and confirm multiple `.pt` files exist:
```bash
uv run python main.py --mode train --loss mse --seeds 1 --grad-accum 1 --walk-forward --force-features --max-epochs 2 2>&1 | tail -5
ls data/models/ | grep fold
```

## Done criteria

- [ ] `uv run pytest -q` → 60 passed
- [ ] `uv run ruff check .` exits 0
- [ ] Walk-forward folds produce unique `best_fold{N}.pt` files
- [ ] Fold test data is saved as `fold_{i}_test.npz`
- [ ] Dead `n_folds <= 1 and n_folds == 0` is simplified to `n_folds == 0`
- [ ] No redundant re-download when fold files already exist

## STOP conditions

- If walk-forward tests take >2 minutes with --force-features, stop and report (data download issues).
- If `model_save_path` is used elsewhere (e.g., resume logic) in a way that conflicts with per-fold paths, report and adjust.

## Maintenance notes

After this change, walk-forward produces multiple model files. The final threshold optimization uses whichever `model_save_path` was set last. Consider whether threshold should be optimized per-fold and averaged, or only on the last fold.
