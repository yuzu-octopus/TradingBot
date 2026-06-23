# Plan 008: Add retry logic to yfinance download

> Drift check: `git diff --stat 672167a..HEAD -- src/data_pipeline.py pyproject.toml`

## Status

- Priority: P2
- Effort: S
- Risk: LOW
- Depends on: none
- Category: reliability
- Planned at: commit `672167a`, 2026-06-23

## Why this matters

The yfinance download loop iterates over 503 tickers with no retry logic. A single DNS hiccup, rate-limit response, or transient network error causes the entire multi-hour feature build to fail with no partial progress or retry. Given 503 network calls, the probability of at least one transient failure is significant.

## Current state

```python
# data_pipeline.py:27-31
df = yf.download(
    ticker, start=start, end=end, auto_adjust=True, progress=False
)
```

Called inside a `for ticker in tqdm(tickers, ...):` loop with no error handling or retry.

## Scope

**In scope**: `src/data_pipeline.py`, `pyproject.toml` (add `tenacity` dependency or use stdlib)

**Out of scope**: Any other file

## Steps

### Step 1: Choose retry approach

Two options:
- **A (recommended)**: Add `tenacity` as dependency — clean declarative retry with exponential backoff.
- **B**: Manual retry loop with `time.sleep` — avoids new dependency, but more code.

Implement option A.

### Step 2: Update `pyproject.toml`

Add `"tenacity>=9.0.0"` to dependencies.

### Step 3: Add retry to `fetch_stock_data`

```python
import tenacity

def fetch_stock_data(...):
    ...

    for ticker in tqdm(...):
        ...

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(3),
            wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        )
        def _download(t):
            df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df

        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            cached += 1
        else:
            try:
                df = _download(ticker)
            except Exception:
                tqdm.write(f"  Failed to download {ticker} after 3 retries, skipping")
                continue
            df.to_csv(path)
        data[ticker] = df
```

The function-level retry wraps only the download call, so cache hits are instant.

**Verify**: `uv run ruff check . && uv run ruff format . && uv run pytest -q` → all pass

Verify yfinance still works: `uv run python -c "import yfinance as yf; d=yf.download('AAPL', period='5d', progress=False); print(f'Downloaded {len(d)} rows')"` → valid

## Done criteria

- [ ] `uv run ruff check .` exits 0
- [ ] `uv run pytest -q` → 60 passed
- [ ] `grep "tenacity" pyproject.toml` returns a match
- [ ] `fetch_stock_data` has retry logic (can be confirmed by code review)

## STOP conditions

- If `tenacity` is incompatible with Python 3.14, fall back to option B (manual retry loop). Report which.
- If yfinance's own retry behavior makes this redundant, reduce retries to 1 (just catch and skip the ticker).

## Maintenance notes

The retry skips the ticker entirely after 3 failed attempts. This means the feature matrix may have missing data for some tickers on some days. The existing `np.nan_to_num(features, nan=0.0)` at features.py handles this silently. Consider logging skipped tickers at the end of the download loop.
