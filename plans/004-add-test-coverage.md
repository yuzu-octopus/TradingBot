# Plan 004: Add test coverage for untested critical paths

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result.
>
> **Drift check (run first)**: `git diff --stat 8473bcf..HEAD -- tests/ src/ main.py`

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: LOW
- **Depends on**: none
- **Category**: tests
- **Planned at**: commit `8473bcf`, 2026-06-28

## Why this matters

Five critical source files have ZERO test coverage: `main.py` (696 LOC, core training pipeline), `src/colab_gen.py` (189 LOC, colab script generation), `src/crypto_pipeline.py` (77 LOC, crypto data fetching), `src/inference.py` (129 LOC, model inference), and `training/threshold.py` (82 LOC, threshold optimization). A regression in any of these silently breaks the training or inference pipeline with no test catching it.

## Current state

- 79 tests passing across 11 test files in `tests/`
- `tests/test_paper_trader.py` (327 lines) — comprehensive coverage of PaperTrader
- `tests/test_pipeline.py` — covers data_pipeline and features
- No test files exist for: `main.py`, `src/colab_gen.py`, `src/crypto_pipeline.py`, `src/inference.py`, `training/threshold.py`

Key functions to test:
- `main.py:_split_date_range`, `main.py:_folds_match_config`, `main.py:_fold_metadata`
- `src/colab_gen.py:_redact_secrets`, `src/colab_gen.py:generate_colab_script`
- `src/inference.py:_last_business_day`, `src/inference.py:_is_nyse_holiday`
- `training/threshold.py:optimize_threshold`

## Steps

### Step 1: Create `tests/test_main.py`

Test the pure-logic functions (no network, no training):

```python
# tests/test_main.py
"""Tests for main.py data preparation and walk-forward utilities."""

import numpy as np
import pandas as pd

from config import Config


def test_split_date_range_basic():
    """_split_date_range returns boolean for date-in-range check."""
    from main import _split_date_range
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    assert _split_date_range(dates[0], "2020-01-01", "2020-01-10") is True
    assert _split_date_range(dates[0], "2020-01-05", "2020-01-10") is False


def test_folds_match_config_no_file():
    """_folds_match_config returns False when no meta file exists."""
    from main import _folds_match_config
    config = Config()
    assert _folds_match_config(config, "/nonexistent/path") is False


def test_fold_metadata_contains_expected_keys():
    """_fold_metadata returns a dict with expected keys."""
    from main import _fold_metadata
    config = Config()
    result = _fold_metadata(config)
    assert isinstance(result, dict)
```

Verify: `uv run pytest tests/test_main.py -v` → tests pass

### Step 2: Create `tests/test_inference.py`

Test business day calculation and holiday detection:

```python
# tests/test_inference.py
"""Tests for src/inference.py utilities."""

from datetime import date

from src.inference import _is_nyse_holiday, _last_business_day


def test_is_nyse_holiday_new_years():
    assert _is_nyse_holiday(date(2026, 1, 1)) is True


def test_is_nyse_holiday_not_saturday():
    """Saturday is not a NYSE holiday (it's a weekend)."""
    # 2026-01-03 is a Saturday
    assert _is_nyse_holiday(date(2026, 1, 3)) is False


def test_last_business_day_returns_string():
    result = _last_business_day()
    assert isinstance(result, str)
    # Should be in YYYY-MM-DD format
    parts = result.split("-")
    assert len(parts) == 3
```

Verify: `uv run pytest tests/test_inference.py -v` → tests pass

### Step 3: Create `tests/test_colab_gen.py`

Test secret redaction and script generation:

```python
# tests/test_colab_gen.py
"""Tests for src/colab_gen.py."""

from src.colab_gen import _redact_secrets


def test_redact_secrets_alpaca_key():
    result = _redact_secrets('ALPACA_API_KEY=pk_abc123def')
    assert "pk_abc123def" not in result
    assert "REDACTED" in result


def test_redact_secrets_alpaca_secret():
    result = _redact_secrets('ALPACA_SECRET_KEY=xyz_secret_789')
    assert "xyz_secret_789" not in result
    assert "REDACTED" in result


def test_redact_secrets_preserves_non_secrets():
    result = _redact_secrets("TICKER=AAPL")
    assert "AAPL" in result
```

Verify: `uv run pytest tests/test_colab_gen.py -v` → tests pass

### Step 4: Create `tests/test_threshold.py`

Test threshold optimization with synthetic data:

```python
# tests/test_threshold.py
"""Tests for training/threshold.py."""

import numpy as np
import torch

from config import Config


def test_optimize_threshold_returns_pair():
    """optimize_threshold returns (buy_t, sell_t) both >= 0."""
    from training.threshold import optimize_threshold
    config = Config()
    config.n_features = 10
    config.n_stocks = 5
    config.tickers = [str(i) for i in range(5)]

    # Create a simple mock model
    from src.utils import create_model
    model = create_model(config, torch.device("cpu"))

    # Synthetic val data
    val_features = np.random.randn(50, 5, 10).astype(np.float32)
    val_targets = np.random.randn(50, 5).astype(np.float32)

    buy_t, sell_t = optimize_threshold(config, model, val_features, val_targets)
    assert isinstance(buy_t, float)
    assert isinstance(sell_t, float)
    assert buy_t >= 0
    assert sell_t >= 0
```

Verify: `uv run pytest tests/test_threshold.py -v` → tests pass

### Step 5: Full validation

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

Expected: all checks passed, pytest shows ~87+ tests (79 existing + ~8 new).

## Done criteria

- [ ] `tests/test_main.py` exists and passes
- [ ] `tests/test_inference.py` exists and passes
- [ ] `tests/test_colab_gen.py` exists and passes
- [ ] `tests/test_threshold.py` exists and passes
- [ ] `uv run pytest -q` exits 0 with all tests passing
- [ ] `uv run ruff check .` exits 0
- [ ] No new dependencies added

## STOP conditions

- If `training/threshold.py:optimize_threshold` requires a real trained model to run, skip that test file and report back
- If any test requires network access (yfinance/Alpaca), mock it or skip
