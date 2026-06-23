# Plan 010: Add tests for untested files

> Drift check: `git diff --stat 672167a..HEAD -- tests/ src/inference.py training/pretrain.py trade.py src/utils.py`

## Status

- Priority: P2
- Effort: L
- Risk: LOW
- Depends on: 001 (paper_trader fixes), 003 (msrr rename), 005 (secrets redact)
- Category: tests
- Planned at: commit `672167a`, 2026-06-23

## Why this matters

Four modules have zero test coverage: `src/inference.py` (56 LOC, core of paper trading), `training/pretrain.py` (267 LOC, entire pretrain pipeline), `trade.py` (221 LOC, Rich display), and `src/utils.py` (86 LOC, model factory). Combined this is ~630 lines of untested code. Bugs in these modules won't be caught by the test suite.

## Current state

Test files and their counts:
- `tests/test_paper_trader.py` — 11 tests ✓
- `tests/test_features.py` — 10 tests ✓
- `tests/test_models.py` — 10 tests ✓
- `tests/test_ddp.py` — 7 tests ✓
- `tests/test_training.py` — 7 tests ✓
- `tests/test_pipeline.py` — 5 tests ✓
- `tests/test_config.py` — 4 tests ✓

The existing tests use `unittest.mock.MagicMock` and `tmp_path` fixtures. Pattern reference:
```python
# tests/test_paper_trader.py: (exemplar for mocking pattern)
from unittest.mock import MagicMock, patch
...
```

## Scope

**In scope**: 
- `tests/test_inference.py` (new)
- `tests/test_pretrain.py` (new)
- `tests/test_trade.py` (new)
- `tests/test_utils.py` (new)

**Out of scope**: Changes to untested source code itself (covered by other plans), any other test file

## Steps

### Step 1: Create `tests/test_inference.py`

Test `_last_business_day()` and `run_inference()`:

```python
"""Tests for src/inference.py."""
from datetime import date
from src.inference import _last_business_day

def test_last_business_day_returns_weekday():
    result = _last_business_day()
    d = date.fromisoformat(result)
    assert d.weekday() < 5  # Monday=0, Friday=4

def test_last_business_day_format():
    assert "-" in _last_business_day()
    parts = _last_business_day().split("-")
    assert len(parts) == 3

# run_inference tests require model + scaler files on disk.
# For now, test what can be tested without filesystem dependencies.
```

Use `pytest.mark.skipif` to guard against missing model files for `run_inference`.

### Step 2: Create `tests/test_utils.py`

Test `create_model`, `unwrap_model`, `load_threshold`:

```python
"""Tests for src/utils.py."""
from config import Config
from models.stock_model import StockTransformer
from src.utils import create_model, unwrap_model

def test_create_model_returns_transformer():
    cfg = Config()
    cfg.tickers = ["AAPL", "MSFT"]
    model = create_model(cfg)
    assert isinstance(unwrap_model(model), StockTransformer)

def test_unwrap_model_plain_module():
    from torch import nn
    inner = nn.Linear(4, 4)
    assert unwrap_model(inner) is inner

def test_load_threshold_default_when_no_file(tmp_path):
    from config import Config
    cfg = Config()
    cfg.features_path = str(tmp_path)
    from src.utils import load_threshold
    buy, sell = load_threshold(cfg)
    assert buy == 0.5
    assert sell == 0.5

def test_load_threshold_parses_file(tmp_path):
    from src.utils import load_threshold
    cfg = Config()
    cfg.features_path = str(tmp_path)
    f = tmp_path / "threshold.txt"
    f.write_text("0.3,0.4")
    buy, sell = load_threshold(cfg)
    assert buy == 0.3
    assert sell == 0.4
```

### Step 3: Create `tests/test_pretrain.py`

Test the loss helpers and data prep functions (not the full training loop):

```python
"""Tests for training/pretrain.py."""
import torch
import numpy as np
from training.pretrain import mpp_loss, top_loss, prepare_mpp, prepare_top

def test_mpp_loss_basic():
    pred = torch.ones(4, 10)
    target = torch.ones(4, 10)
    mask = torch.zeros(4, 10, dtype=torch.bool)
    mask[0, 0] = True
    pred[0, 0] = 0.0  # mismatch
    loss = mpp_loss(pred, target, mask)
    assert loss.item() == 1.0  # (0-1)^2 = 1

def test_top_loss_basic():
    logits = torch.randn(4, 6)
    labels = torch.randint(0, 6, (4,))
    loss = top_loss(logits, labels)
    assert loss.item() > 0  # cross-entropy is positive

def test_prepare_mpp_shape():
    features = np.random.randn(50, 10, 8)
    targets = np.random.randn(50, 10)
    masked, y, mask = prepare_mpp(features, targets, mask_ratio=0.2)
    assert masked.shape == features.shape
    assert y.shape == targets.shape
    assert mask.shape == (50, 10)

def test_prepare_top_shape():
    features = np.random.randn(50, 10, 8)
    windows, labels, n_classes = prepare_top(features, n_days=3)
    assert windows.shape == (48, 3, 10, 8)
    assert labels.shape == (48,)
    assert n_classes == 6  # 3! = 6
```

### Step 4: Create `tests/test_trade.py`

Test the Rich table builder functions (mock `Console` to avoid terminal output):

```python
"""Tests for trade.py — display functions only (no trading)."""
from trade import make_trade_table, build_layout

def test_make_trade_table_returns_table():
    signals = {"AAPL": {"score": 0.8, "signal": "BUY"}}
    positions = {}
    trades = [("AAPL", 10, "BUY")]
    account = {"equity": 100000, "cash": 50000, "day_change": 100}
    table = make_trade_table(signals, positions, trades, account, cycle=1, interval=900, now_str="2026-01-01 10:00:00 ET")
    assert table.title is not None
    assert "BUY" in table.title

def test_build_layout_returns_layout():
    from rich.table import Table
    t = Table()
    layout = build_layout(t)
    assert layout is not None
```

**Verify**: `uv run pytest -q tests/` → all new tests pass alongside existing 60

Run specific: `uv run pytest tests/test_inference.py tests/test_utils.py tests/test_pretrain.py tests/test_trade.py -v`

## Done criteria

- [ ] `uv run pytest -q` → 70+ passed (60 existing + 10+ new)
- [ ] `tests/test_inference.py` exists with tests for `_last_business_day`
- [ ] `tests/test_utils.py` exists with tests for `unwrap_model`, `load_threshold`, `create_model`
- [ ] `tests/test_pretrain.py` exists with tests for `mpp_loss`, `top_loss`, `prepare_mpp`, `prepare_top`
- [ ] `tests/test_trade.py` exists with tests for `make_trade_table`, `build_layout`
- [ ] `uv run ruff check .` exits 0

## STOP conditions

- If a test requires filesystem artifacts (model files, scaler files) that don't exist in CI, guard with `@pytest.mark.skipif(not Path("data/models/best.pt").exists(), reason="no model file")`.

## Maintenance notes

These tests are lightweight smoke tests — they verify the code runs without crashing and basic properties hold. They don't test the full training loop (which requires GPU and 30+ minutes). When adding features to these modules, add corresponding test cases.
