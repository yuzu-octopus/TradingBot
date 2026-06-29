# Codebase Fixes — Completed

## Crypto crash fixes (CRITICAL)

| Fix | File | Details |
|-----|------|---------|
| Dynamic %-based date split | `main.py` | 60/20/20 train/val/test split from actual available dates for crypto |
| Empty-data guard in pretrain | `training/pretrain.py` | Guard moved BEFORE `prepare_mpp`/`prepare_top` (was after, never fired) |
| Empty-data guard in train | `training/train.py` | Check for zero-row features/targets at top of `train()` |
| Min-row check in pretrain | `training/pretrain.py` | `T >= pretrain_top_n_days` guard prevents `np.stack` crash |

## Adversarial review fixes

| Fix | File | Details |
|-----|------|---------|
| Stock embedding bug | `models/stock_model.py` | `self.stock_embed(perm)` instead of `arange` — shuffled stocks get correct embeddings |
| Walk-forward test evaluation | `main.py` | Save test splits, evaluate with val-derived thresholds (no leakage), `try/finally` for config |
| Validation batching | `training/train.py` | Batch validation in chunks to avoid OOM on large universes |
| MPS GradScaler guard | `train.py`, `pretrain.py` | Only create `GradScaler` for CUDA (MPS handles float16 natively) |
| Double NaN warmup | `src/features.py` | `rolling_1y` uses `min_periods=1` — preserves smoothing without wasting 252 days |
| Crypto weekend cache | `src/inference.py` | Crypto uses `(now - 1 day)` instead of NYSE business-day logic |
| Dead imports removed | `main.py` | `time`, `datetime`, `ZoneInfo`, `run_inference` (left over from refactor) |
| `getattr` → direct attr | `training/pretrain.py` | `getattr(config, "no_amp", False)` → `config.no_amp` |

## Ruff fixes

| File | Error | Fix |
|------|-------|-----|
| `main.py` | F821 `te` undefined | `_te` → `te` in loop destructuring |
| `training/train.py` | RUF003 ambiguous `×` | `×` → `x`, `O(S²)` → `O(S^2)` |
| `src/colab_gen.py` | SIM115 no context manager | `with Path.open("rb") as f: tomllib.load(f)` |

## Code reviewer nits

| Fix | Details |
|-----|---------|
| Import hoisting | `from training.threshold import optimize_threshold` moved outside fold loop |
| Missing-val warning | Print warning when fold's `val_path` doesn't exist (default thresholds used) |

## Validation

- ruff: clean ✅
- ruff format: 31 files already formatted ✅
- mypy: 36 errors (baseline) ✅
- pytest: 79 passed ✅
