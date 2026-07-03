# Plan 003: Add Juneteenth and Good Friday to NYSE holiday set

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result.
>
> **Drift check (run first)**: `git diff --stat 8473bcf..HEAD -- src/inference.py`

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `8473bcf`, 2026-06-28

## Why this matters

NYSE added Juneteenth as a holiday in 2022. Good Friday has been a NYSE holiday for decades. Neither is in the current `NYSE_HOLIDAYS` set at `src/inference.py:25-47`. On these dates, `_last_business_day()` will return the holiday date itself, causing inference to use non-trading-day data. This is a silent data-quality bug — the model will see stale prices and produce incorrect signals.

## Current state

`src/inference.py:25-47`:
```python
NYSE_HOLIDAYS: set[tuple[int, int]] = {
    (1, 1),    # New Year's Day
    (7, 4),    # Independence Day
    (12, 25),  # Christmas
}
```

Plus floating holidays computed by helper functions (MLK Day, Presidents Day, Memorial Day, Labor Day, Thanksgiving). Juneteenth (June 19) and Good Friday (varies) are missing.

## Steps

### Step 1: Add Juneteenth

Juneteenth is always June 19. Add `(6, 19)` to `NYSE_HOLIDAYS`.

### Step 2: Add Good Friday

Good Friday is 2 days before Easter. Easter is computed by the algorithmic method. Add a helper or hardcoded date. The simplest approach: add a `GOOD_FRIDAYS` set with the next 10 years of dates, or compute Easter algorithmically. The standard algorithm (Anonymous Gregorian) is ~10 lines:

```python
def _easter(year: int) -> date:
    """Compute Easter Sunday for a given year (Anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)
```

Good Friday = Easter - 2 days.

Add `_is_good_friday(d: date) -> bool` and check it in `_last_business_day`.

### Step 3: Verify

```bash
uv run ruff check src/inference.py && uv run pytest -q
```

Expected: all checks pass, 79 tests pass.

## Done criteria

- [ ] Juneteenth `(6, 19)` in `NYSE_HOLIDAYS`
- [ ] Good Friday detection in `_is_nyse_holiday` or `_last_business_day`
- [ ] `uv run ruff check .` exits 0
- [ ] `uv run pytest -q` exits 0

## STOP conditions

- If the Easter algorithm is too complex, fall back to a hardcoded `GOOD_FRIDAYS: set[tuple[int, int]]` for the next 10 years.
