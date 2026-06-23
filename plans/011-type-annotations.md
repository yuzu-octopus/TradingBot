# Plan 011: Add missing type annotations across the codebase

> Drift check: `git diff --stat 672167a..HEAD -- main.py src/paper_trader.py src/inference.py training/pretrain.py training/train.py src/features.py trade.py`

## Status

- Priority: P3
- Effort: L
- Risk: LOW
- Depends on: 010 (tests — to verify no regressions)
- Category: tech-debt
- Planned at: commit `672167a`, 2026-06-23

## Why this matters

48 functions lack type annotations (`ruff check --select ANN`). This means mypy can't verify correctness through most of the codebase. Over time, as the code evolves, untyped functions become a source of subtle bugs that could be caught statically.

## Current state

Run to see current count:
```bash
uv run ruff check --select ANN . 2>&1 | grep "AN" | wc -l
```

Key untyped functions:
- `main.py:27` — `prepare_walk_forward_splits(features, targets, market_state, dates, config)` — 5 params, all untyped
- `src/paper_trader.py` — `__init__`, `get_account`, `get_positions`, `cancel_open_orders`, `submit_market_order`, `reconcile` — 6 public methods with partial/no return types
- `training/pretrain.py` — `prepare_mpp`, `prepare_top`, `mpp_loss`, `top_loss`, `_csr_loss` — 5 functions untyped
- `trade.py:127` — `main()` → no return type annotation
- `src/features.py` — `compute_window_features` returns `pd.DataFrame` but no annotation

## Scope

**In scope**: `main.py`, `src/paper_trader.py`, `src/inference.py`, `training/pretrain.py`, `training/train.py`, `src/features.py`, `trade.py`

**Out of scope**: Test files, `src/colab_gen.py` (generated script scaffolding), any behavioral changes

## Steps

### Step 1: Run current annotation count

```bash
uv run ruff check --select ANN . 2>&1 | grep -c "ANN"
```

### Step 2: Annotate `main.py`

Add return types and parameter types to all top-level functions:

```python
# prepare_walk_forward_splits
def prepare_walk_forward_splits(
    features: np.ndarray,
    targets: np.ndarray,
    market_state: np.ndarray,
    dates: list[str],
    config: Config,
) -> int:

# _split_date_range (already partially typed — verify)
def _split_date_range(
    dates: list[str], config: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

# prepare_data
def prepare_data(config: Config) -> int:

# print_signals
def print_signals(results: dict[str, dict]) -> None:

# run_paper_trading
def run_paper_trading(config: Config, args: argparse.Namespace) -> None:

# main
def main() -> None:
```

### Step 3: Annotate `src/paper_trader.py`

```python
class PaperTrader:
    def __init__(self, config: Config) -> None:
    def get_account(self) -> dict:
    def get_positions(self) -> dict[str, dict]:
    def get_latest_quotes(self, symbols: list[str]) -> dict[str, dict]:
    def market_open(self) -> bool:
    def next_open(self) -> datetime:
    def next_close(self) -> datetime:
    def cancel_open_orders(self, symbol: str | None = None) -> None:
    def submit_market_order(self, symbol: str, qty: int, side: OrderSide) -> Order:
    def reconcile(self, signals: dict[str, dict]) -> list:
```

### Step 4: Annotate `training/pretrain.py`

```python
def _csr_loss(pred: torch.Tensor, target: torch.Tensor, loss_mode: str) -> torch.Tensor:

def prepare_mpp(
    features: np.ndarray, targets: np.ndarray, mask_ratio: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

def prepare_top(
    features: np.ndarray, n_days: int = 3, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, int]:

def mpp_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

def top_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
```

### Step 5: Add remaining annotations

- `src/features.py` — `compute_window_features(df: pd.DataFrame) -> pd.DataFrame`
- `trade.py:127` — `def main() -> None:`

**Verify**: 
```bash
uv run ruff check --select ANN . 2>&1 | grep -c "ANN"
# Should be 0 or near 0 (some third-party stubs may still trigger)
uv run ruff check . && uv run ruff format . && uv run pytest -q
# All pass
```

## Done criteria

- [ ] `uv run ruff check .` exits 0
- [ ] `uv run pytest -q` → 60+ passed
- [ ] Annotation count from `ruff --select ANN` is ≤ 5 (remaining may be from third-party stubs)

## STOP conditions

- Do NOT add `-> None` return types to test files — that's out of scope.
- If a function's type is genuinely complex (e.g., `Reconcile` returns a `list[tuple[str, int, str]]`), prefer a type alias over `list`.

## Maintenance notes

Adding annotations is incremental. Future PRs should require annotations on new functions. Consider adding `ruff check --select ANN` to CI (plan 009) with a `--ignore` list for existing violations.
