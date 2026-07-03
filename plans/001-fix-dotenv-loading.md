# Plan 001: Fix .env loading — strip quotes and cover all entry points

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise.
>
> **Drift check (run first)**: `git diff --stat 8473bcf..HEAD -- textual_trader.py main.py trade.py .env.example`
> If any in-scope file changed, compare excerpts against live code before proceeding.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `8473bcf`, 2026-06-28

## Why this matters

Two entry points (`trade.py` and `main.py --mode trade`) never load `.env` files. Users who set Alpaca keys in `.env` (the documented setup in README) will get silent authentication failures when running via those paths — only `textual_trader.py` calls `_load_dotenv()`. Additionally, `_load_dotenv` doesn't strip quotes, so a `.env` like `ALPACA_API_KEY="pk_abc"` sends the literal quotes to Alpaca's API, causing auth failure.

## Current state

- `textual_trader.py:171-183` — `_load_dotenv()` is a hand-rolled parser. Line 183: `os.environ[key] = val` stores the raw value including any surrounding quotes.
- `textual_trader.py:802` — only call site: `_load_dotenv()` called at startup.
- `trade.py` — no `_load_dotenv` call. Alpaca keys must be exported manually.
- `main.py` — no `_load_dotenv` call. Same issue for `--mode trade`.
- `.env.example` shows `ALPACA_API_KEY=pk_...` (no quotes in example, but users often add quotes).

Excerpt — current `_load_dotenv`:
```python
# textual_trader.py:171-183
def _load_dotenv() -> None:
    """Load .env file if present (uv run doesn't auto-load it)."""
    env_path = Path(".env")
    if env_path.exists():
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key not in os.environ:
                os.environ[key] = val  # ← val includes surrounding quotes
```

Repo conventions: all source files use `from pathlib import Path`, relative paths from project root, no external dependencies beyond what's in `pyproject.toml`.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Lint | `uv run ruff check .` | All checks passed |
| Format | `uv run ruff format --check .` | All checks passed |
| Tests | `uv run pytest -q` | 79 passed |
| Typecheck | `uv run mypy . 2>&1 \| tail -3` | ≤ 17 errors |

## Scope

**In scope:**
- `textual_trader.py` — fix `_load_dotenv` to strip quotes; move to a shared location or make importable
- `main.py` — add `_load_dotenv()` call at startup (before Config() is constructed)
- `trade.py` — add `_load_dotenv()` call at startup (before Config() is constructed)

**Out of scope:**
- `src/paper_trader.py` — reads from `os.environ` as fallback; this is correct and stays.
- Switching to a `python-dotenv` dependency — not needed, the hand-rolled parser is 12 lines.

## Steps

### Step 1: Fix quote stripping in `_load_dotenv`

In `textual_trader.py`, change line 183 from:
```python
os.environ[key] = val
```
to:
```python
# Strip surrounding quotes (single or double) — common .env convention
val = val.strip("'\"")
os.environ[key] = val
```

**Verify**: `uv run ruff check textual_trader.py` → no new errors

### Step 2: Add `_load_dotenv` to `main.py`

In `main.py`, add near the top imports (after the existing `import` block):
```python
from textual_trader import _load_dotenv as _load_dotenv_env
```

Wait — this creates a dependency from main.py → textual_trader.py which is wrong (textual is optional). Instead, **copy the function into main.py** (it's 12 lines):

In `main.py`, add before `def main()`:
```python
def _load_dotenv() -> None:
    """Load .env file if present (uv run doesn't auto-load it)."""
    env_path = Path(".env")
    if env_path.exists():
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.strip("'\"")
            if key not in os.environ:
                os.environ[key] = val
```

Then in `main()`, add `_load_dotenv()` as the first line before `config = Config()`.

**Verify**: `uv run ruff check main.py` → no new errors

### Step 3: Add `_load_dotenv` to `trade.py`

Same pattern — copy the 12-line function into `trade.py` before `def main()`. Add `_load_dotenv()` as the first line in `main()`.

**Verify**: `uv run ruff check trade.py` → no new errors

### Step 4: Full validation

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

Expected: all checks passed, 79 tests pass.

## Test plan

- No new unit tests needed — this is a 3-line behavioral fix. Verification is via manual test: create `.env` with `ALPACA_API_KEY="pk_test"` (quoted) and confirm the value is loaded without quotes.

## Done criteria

- [ ] `uv run ruff check .` exits 0
- [ ] `uv run pytest -q` exits 0; 79 tests pass
- [ ] `textual_trader.py:_load_dotenv` strips quotes from values
- [ ] `main.py` calls `_load_dotenv()` before `Config()`
- [ ] `trade.py` calls `_load_dotenv()` before `Config()`
- [ ] No files outside scope modified

## STOP conditions

- If `Path` is not already imported in `main.py` or `trade.py` (check the imports section)
- If adding `_load_dotenv` creates a circular import

## Maintenance notes

- Future: consider extracting `_load_dotenv` into `src/utils.py` if a third entry point needs it. For now, two copies is simpler than a new module.
- The `_load_dotenv` function does NOT override existing env vars (line 182: `if key not in os.environ`). This is intentional — explicit env exports take precedence over `.env` file.
