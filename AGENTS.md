# TradingBot

Multi-stock ML trading bot. Inputs multi-stock windows (1y, 1m, 1w, 1d), outputs per-stock buy/sell confidence scores (-1 to 1). Learns inter-stock relationships — all stocks pass through the same model in one forward pass.

## Setup

```bash
uv sync                         # Create venv + install all deps
uv add torch pandas numpy       # Core ML deps
uv add --dev ruff mypy pytest   # Dev deps (not in project yet)
uv run python main.py           # Run entrypoint
uv run python main.py --mode train --loss msrr --seeds 5 --grad-accum 4  # MSRR loss, ensemble, gradient accumulation
uv run python main.py --mode train --resume       # Resume from checkpoint
uv run python main.py --mode train --force-features  # Rebuild feature matrix from scratch
uv run python main.py --mode train --walk-forward    # Walk-forward validation (3-year windows)
uv run python main.py --mode train --loss margin     # Ranking loss (pairwise margin)
uv run python main.py --mode train --loss listnet    # Listwise ranking loss
uv run python trade.py --interval 15                 # Alpaca paper trading (Rich display)
uv run python trade.py --interval 15 --headless      # Paper trading (logs only)
uv run python main.py --mode trade --trade-interval 15  # Paper trading via main.py
```

Python 3.14 via `.python-version`. uv manages everything — no manual `.venv/bin/activate`.

## Dev commands

```bash
uv run ruff format --check .    # Format check
uv run ruff check .             # Lint
uv run mypy .                   # Typecheck (use `# type: ignore` for ML code)
uv run pytest                   # Tests
```

`mypy` accepts a single global mypy config in `pyproject.toml`:
- `src.features` is excluded from per-error checking — its rolling/Series
  chained operations trip `pandas-stubs` overload rules that are not
  runtime-relevant.
- Missing imports are ignored for `src.data_pipeline`, `src.utils`,
  `src.paper_trader`, `trade.py`, `main.py`, `training.train`, and
  `training.threshold` (sklearn, yfinance, alpaca-py don't ship stubs).


## Colab template

```bash
uv run python main.py --colab-template --loss msrr --grad-accum 4 --seeds 3
```
Generates a complete Colab script with all source embedded. Copies to clipboard. Paste into a Colab GPU runtime, run, then download the model zip to `data/models/colab/<run-name>/`. Evaluate with `--model colab/<run-name>`.

## Project structure

```
data/stocks/         # Per-stock CSVs (503 S&P 500 tickers)
data/features/       # Preprocessed feature matrices + market state
models/
  stock_model.py     # StockTransformer — decoder-only + RankGLU + MarketGate
training/
  train.py           # Training loop (mixed precision, checkpoint/resume)
  threshold.py       # Post-training Sharpe-based threshold optimization
src/
  data_pipeline.py   # Fetch OHLCV via yfinance
  features.py        # Window feature engineering + parallel build + market state
  inference.py       # On-demand inference (with market state)
  paper_trader.py    # Alpaca paper trading wrapper (TradingClient, reconcile, loop)
  utils.py           # Shared: model factory, scaler save/load, feature scaling
trade.py             # Standalone Alpaca paper trading script with Rich display
config.py            # Dataclass: tickers, windows, model params
main.py              # Entry point: --mode train|infer, --loss mse|msrr|margin|listnet
```

## Key architecture decisions

- **Decoder-only with causal mask**: Each stock attends to itself and preceding stocks. Research shows decoder-only beats encoder-only for stock prediction.
- **RankGLU output head**: Residual bottleneck GLU instead of linear head. Better ranking. From RankGLU paper (arXiv 2606.08930).
- **Market-guided gating**: SPY market state rescales features per day. From MASTER (AAAI 2024).
- **Cross-sectional z-score normalization**: Targets normalized per day (mean=0, std=1). Standard for ranking-aware models.
- **Threshold post-optimization**: Model outputs raw scores (-1 to 1). Post-training optimization finds separate buy/sell thresholds maximizing Sharpe ratio.
- **Alpaca paper trading**: Model scores → paper orders via Alpaca API. yfinance for training data, Alpaca for live execution. See `.env.example` for API keys.
- **No secrets**: No API keys, no env vars — stock data is public market data (except Alpaca API keys for paper trading; keep in `.env`).

## Paper-trading safety rules

These are tested in `tests/test_paper_trader.py` (mocked Alpaca clients)
and must be preserved when modifying `src/paper_trader.py`:

- **Per-ticker cancel scope**: `reconcile()` may only call
  `cancel_open_orders(symbol=t)` for tickers present in the current
  signal set. Never blanket-cancel — triggers duplicate fills and
  Alpaca rate-limit guards.
- **Fractional share rounding**: `qty = round(abs(pos["qty"]))` to
  preserve fractional-share positions without int-truncation drift.
- **Partial close**: SELL closes `min(held, trade_sell_qty)`;
  smaller-than-qty positions close fully.
- **Position cap**: BUY is checked against
  `equity * trade_max_position_pct` using the *real ask price* fetched
  from `get_latest_quotes()`. No hardcoded $100 placeholder. Because
  BUY only fires on tickers with no current position, the cap is a
  *per-entry new-notional* limit, not a cumulative per-ticker exposure
  limit. Multiple cycle-in/cycle-out sequences on the same ticker can
  exceed this number on aggregate.
- **No-equity guard**: if `account.equity <= 0`, all BUYs are blocked
  with `NO_EQUITY` rather than firing with bad notional math.
- **Order failures are captured** in the `trades` list as
  `<action>_FAIL:<exception>` instead of raising out of the cycle.

## Timezones

Use `zoneinfo.ZoneInfo` (Python 3.9+, natively supported on 3.14). The
older `pytz` library is deprecated; do not reintroduce it. All three of
`main.py`, `src/paper_trader.py`, `trade.py`, and `src/inference.py`
use `ZoneInfo("America/New_York")` consistently.

## Hardware acceleration

- **Apple Silicon (MPS)**: Auto-detected — uses `mps` backend for GPU acceleration
- **NVIDIA (CUDA)**: Auto-detected — uses `cuda` if available before falling back to MPS
- **CPU fallback**: Works on any machine, just slower
- **Mixed precision**: Supported on both CUDA and MPS — ~30-40% training speedup
- Model is small (~478K params) — MPS handles full batches easily (~17 MB per batch)

## Conventions

- Pure `pyproject.toml` deps — no `requirements.txt`
- All paths relative to project root
- Config via dataclasses or YAML
- Prefer readable, simple code — user is new to AI/ML

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `train` | `train`, `infer`, `pretrain`, or `trade` |
| `--trade-interval` | `15` | Minutes between trading cycles |
| `--trade-headless` | off | Run paper trading without Rich display |
| `--trade-buy-qty` | `10` | Shares to buy per long signal |
| `--trade-sell-qty` | `10` | Shares to sell per short signal |
| `--buy-threshold` | — | Override buy threshold |
| `--sell-threshold` | — | Override sell threshold |
| `--loss` | `mse` | `mse`, `msrr`, `margin`, `listnet` |
| `--seeds` | `1` | Ensemble seeds (train N, average predictions) |
| `--grad-accum` | `1` | Gradient accumulation steps |
| `--resume` | off | Resume from checkpoint |
| `--walk-forward` | off | Walk-forward validation (sliding windows) |
| `--force-features` | off | Rebuild feature matrix from scratch |
| `--model <path>` | — | Load model from `data/models/<path>/best.pt` |
| `--colab-template` | off | Generate self-contained Colab training script |
