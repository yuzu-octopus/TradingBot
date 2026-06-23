# Plan 001: PaperTrader error-handling hardening

> Drift check: `git diff --stat 672167a..HEAD -- src/paper_trader.py`

## Status

- Priority: P1
- Effort: S
- Risk: LOW
- Depends on: none
- Category: bug
- Planned at: commit `672167a`, 2026-06-23

## Why this matters

Three independent bugs in `src/paper_trader.py`: (1) an uncaught `get_orders()` exception inside `cancel_open_orders` aborts the entire trading cycle, (2) a quote-API outage silently bypasses the `MAX_POS_CAP` guard (fail-open), and (3) `round()` can sell slightly more shares than held. Individually rare, collectively they create real-money-shaped risk in the trade loop.

## Current state

**F7** — `cancel_open_orders` lines 98–109: `get_orders()` is unprotected. If it raises, `reconcile()` aborts mid-cycle.

```python
# paper_trader.py:98-109
def cancel_open_orders(self, symbol: str | None = None):
    if symbol:
        orders = self.trade_client.get_orders(
            filter=GetOrdersRequest(symbols=[symbol] if symbol else None)
        )
    else:
        orders = self.trade_client.get_orders()
    for o in orders:
        try:
            self.trade_client.cancel_order_by_id(order_id=o.id)  # type: ignore[union-attr]
```

**F8** — `get_latest_quotes` lines 71–83: returns `{}` on any exception. Caller at reconcile line 152 sees no ask price and skips the cap check:

```python
# paper_trader.py:71-83
def get_latest_quotes(self, symbols: list[str]) -> dict[str, dict]:
    ...
    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=symbols)
        quotes = self.data_client.get_stock_latest_quote(req)
    except Exception:
        return {}
```

**F10** — `reconcile` line 167: `round()` can round up past held qty:

```python
held = round(abs(pos["qty"]))
```

## Scope

**In scope**: `src/paper_trader.py`

**Out of scope**: tests for these fixes (covered in plan 010), any other file.

## Steps

### Step 1: Wrap `get_orders()` in try/except

Wrap both branches of `cancel_open_orders`:
```python
def cancel_open_orders(self, symbol: str | None = None):
    try:
        if symbol:
            orders = self.trade_client.get_orders(...)
        else:
            orders = self.trade_client.get_orders()
    except Exception as e:
        logger.warning("Failed to fetch open orders: %s", e)
        return
    for o in orders:
        ...
```

### Step 2: Make `get_latest_quotes` fail-closed on quote errors

Keep the try/except but log the error. The caller already handles empty dicts — the fix is just logging so the operator knows quotes failed instead of silently proceeding without a cap check.

Add a `logger.warning()` inside the except.

### Step 3: Replace `round()` with `math.floor()` for sell qty

```python
import math
...
held = math.floor(abs(pos["qty"]))
```

Add `import math` at top.

**Verify**: `uv run ruff check . && uv run ruff format . && uv run pytest -q` → all pass

## Test plan

Tests for these fixes exist in `tests/test_paper_trader.py` (covered by plan 010).

## Done criteria

- [ ] `uv run ruff check .` exits 0
- [ ] `uv run ruff format --check .` exits 0
- [ ] `uv run pytest -q` → 60 passed
- [ ] Code in `paper_trader.py` wraps `get_orders()` in try/except
- [ ] Code in `paper_trader.py` logs on quote failure instead of silent `return {}`
- [ ] Code in `paper_trader.py` uses `math.floor()` for sell qty

## STOP conditions

- Code at cited lines doesn't match — drift detected.

## Maintenance notes

If `get_orders()` signature changes (Alpaca SDK update), the try/except ensures graceful degradation rather than a crash.
