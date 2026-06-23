# Plan 007: Cache raw_data across trade cycles to avoid redundant I/O

> Drift check: `git diff --stat 672167a..HEAD -- src/inference.py src/paper_trader.py main.py trade.py`

## Status

- Priority: P2
- Effort: S
- Risk: LOW
- Depends on: none
- Category: perf
- Planned at: commit `672167a`, 2026-06-23

## Why this matters

Every paper-trading cycle calls `run_inference()` → `fetch_stock_data()` which reads all 503 CSVs from disk. This adds 3-8 seconds of I/O per 15-minute cycle. Since stock data only changes once per day (after market close), the raw data is identical across intraday cycles. Caching it in memory eliminates the redundant I/O.

## Current state

```python
# inference.py:22-24
def run_inference(config, ...):
    raw_data = fetch_stock_data(
        config.tickers, config.train_start, config.test_end, config.raw_data_path
    )
    # ... reads 503 CSVs every call
```

``inference.py` is called from both `main.py:467` (trade mode) and `trade.py:186`.

## Scope

**In scope**: `src/inference.py` only

**Out of scope**: `trade.py`, `main.py`, tests

## Steps

### Step 1: Add module-level cache for raw_data

```python
# inference.py
_raw_data_cache: dict[str, dict[str, pd.DataFrame]] = {}
```

### Step 2: Modify `run_inference` to use cache

Add a simple cache keyed by ticker list fingerprint:

```python
def run_inference(config, ...):
    cache_key = str(hash(tuple(config.tickers)))
    if cache_key not in _raw_data_cache:
        _raw_data_cache[cache_key] = fetch_stock_data(
            config.tickers, config.train_start, config.test_end, config.raw_data_path
        )
    raw_data = _raw_data_cache[cache_key]
    ...
```

The cache lives for the lifetime of the Python process, which matches the paper-trading loop pattern (single long-running process).

**Verify**: `uv run ruff check . && uv run ruff format . && uv run pytest -q` → all pass

Manually verify the cache works:
```bash
uv run python -c "
from config import Config, get_sp500_tickers
from src.inference import run_inference
c = Config()
c.tickers = get_sp500_tickers()[:5]  # just 5 tickers for speed
r = run_inference(c)  # first call — loads from disk
r2 = run_inference(c)  # second call — should be cached
print(f'Signals produced: {len(r)}')
" 2>&1 | tail -3
```

## Test plan

No new tests needed — existing inference tests exercise the code path. The cache is a performance optimization with no behavioral change.

## Done criteria

- [ ] `uv run ruff check .` exits 0
- [ ] `uv run pytest -q` → 60 passed
- [ ] `_raw_data_cache` exists in `src/inference.py`
- [ ] Multiple calls to `run_inference` with the same tickers don't re-read CSVs

## STOP conditions

- If the cache grows unbounded (different ticker lists across calls), add a simple eviction or limit (e.g., LRU with size 1, since the trade loop always uses the same tickers). Report if ticker lists can change mid-process.

## Maintenance notes

The cache uses a process-global dict. If inference is ever called from multiple threads with different ticker lists, this cache will waste memory. For the single-threaded trade loop, it's ideal.
