# TradingBot

Multi-stock ML trading bot. Learns inter-stock relationships — all stocks pass through the same Transformer model in one forward pass.

**Input:** OHLCV + technical indicators for each stock across 4 lookback windows (1y, 1m, 1w, 1d)

**Output:** Per-stock buy/sell confidence score from -1 (strong sell) to +1 (strong buy)

**Universe:** S&P 500 (~503 stocks) fetched live from Wikipedia

## Quick start

```bash
uv sync                         # Create venv + install all deps
uv run python main.py --mode train      # First run: fetches data, builds features, trains model
```

First run fetches ~10 years of OHLCV data for all S&P 500 stocks, computes window features (1y/1m/1w/1d lookbacks), trains the Transformer model, and optimizes buy/sell thresholds.

Python 3.14 via `.python-version`. uv manages everything — no manual `.venv/bin/activate`.

## Usage

```bash
uv run python main.py --mode train                          # Train with MSE loss
uv run python main.py --mode train --loss msrr              # Portfolio MSE loss
uv run python main.py --mode train --seeds 5 --grad-accum 4 # Ensemble + gradient accumulation
uv run python main.py --mode train --walk-forward           # Walk-forward validation
uv run python main.py --mode train --resume                 # Resume from checkpoint
uv run python main.py --mode infer                          # Get today's trading signals
uv run python main.py --mode infer --model colab/run1       # Evaluate a Colab-trained model
uv run python main.py --mode trade --trade-interval 15      # Alpaca paper trading loop
uv run python trade.py --interval 15                        # Standalone paper trading (Rich display)
uv run python textual_trader.py --interval 15               # Textual TUI paper trading dashboard
uv run python main.py --mode pretrain                       # Self-supervised pre-training
uv run python main.py --mode train --asset-class crypto     # Train on crypto pairs (BTC, ETH, etc.)
uv run python trade.py --interval 15 --asset-class crypto   # Crypto paper trading
torchrun --nproc_per_node=N uv run python main.py --mode train  # Multi-GPU (DDP)
uv run python main.py --colab-template --loss msrr --seeds 3 --grad-accum 4  # Generate Colab/Kaggle script
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `train` | `train`, `infer`, `pretrain`, or `trade` |
| `--loss` | `mse` | `mse`, `msrr` (portfolio MSE), `margin`, `listnet` |
| `--seeds` | `1` | Ensemble size (multiple random seeds) |
| `--grad-accum` | `1` | Gradient accumulation steps |
| `--resume` | off | Resume training from last checkpoint |
| `--walk-forward` | off | Walk-forward validation (sliding chronological windows) |
| `--force-features` | off | Rebuild feature matrix from scratch |
| `--model <path>` | — | Load model from `data/models/<path>/best.pt` |
| `--trade-interval` | `15` | Minutes between trading cycles |
| `--trade-headless` | off | Run paper trading without Rich display |
| `--trade-buy-qty` | `10` | Shares to buy per long signal |
| `--trade-sell-qty` | `10` | Shares to sell per short signal |
| `--buy-threshold` | — | Override buy threshold |
| `--sell-threshold` | — | Override sell threshold |
| `--pretrain` | off | Initialize training from pre-trained weights |
| `--colab-template` | off | Generate self-contained Colab script |
| `--show-script` | off | Print colab script to terminal |
| `--asset-class` | `stocks` | `stocks` or `crypto` — switches data pipeline and model |
| `--crypto-pairs` | `top10` | `top10` or `all17` — crypto universe size |
| `--no-amp` | off | Disable mixed-precision training |
| `--tickers-file` | — | File with one ticker per line (overrides default) |

## Remote Training (Colab / Kaggle)

```bash
uv run python main.py --colab-template --loss msrr --grad-accum 4 --seeds 3
```

Generates a self-contained script with all source code embedded (copies to clipboard). The generated script **auto-detects** whether it's running on Colab or Kaggle at runtime and adapts paths, package installation, and output download accordingly.

- **Colab:** Paste into a GPU cell → model downloads automatically via `files.download()`
- **Kaggle:** Create a Notebook with GPU accelerator, paste into a cell → model saved to `/kaggle/working/`, download via Output sidebar

Place model zip in `data/models/colab/<run-name>/` and evaluate with `--model colab/<run-name>`.

## Architecture

```
main.py              → CLI entry point
config.py            → Dataclass: tickers, model params, training settings
models/stock_model.py → StockTransformer (decoder-only, RankGLU output, MarketGate)
src/data_pipeline.py  → yfinance data fetching with CSV caching
src/features.py       → Window feature engineering + parallel build
src/inference.py      → On-demand inference with market state
src/paper_trader.py   → Alpaca paper trading wrapper
src/utils.py          → Model factory, scaler save/load, threshold loading
training/train.py     → Training loop with mixed precision + checkpoint/resume
training/pretrain.py  → Self-supervised pre-training (D6)
training/threshold.py → Post-training Sharpe-based threshold optimization
trade.py              → Standalone paper trading script (Rich display)
```

## How it works

1. **Data:** Downloads OHLCV for all S&P 500 stocks via yfinance (cached to `data/stocks/`)
2. **Features:** For each date, computes window features from 1y, 1m, 1w, 1d lookbacks — including SMA, RSI, MACD, Bollinger Bands, volatility, returns, drawdown. **Parallelized** across stocks for speed.
3. **Model:** Encoder-only Transformer (full bidirectional self-attention) + RankGLU output head + market-guided gating (SPY state). Cross-stock self-attention.
4. **Training:** Supervised regression on next-day return. MSE, MSRR, margin ranking, or ListNet loss. Mixed precision, gradient clipping, weight decay, cosine annealing, early stopping.
5. **Threshold:** Post-training, calibrates scores via isotonic regression, then optimizes separate buy/sell thresholds **maximizing Sharpe ratio** (not just return).

## Model details

| Property | Value |
|----------|-------|
| Parameters | ~478K |
| Architecture | Encoder-only Transformer (full bidirectional), 3 layers, 4 heads |
| Output head | RankGLU (residual bottleneck GLU) |
| Conditioning | MarketGate (SPY-based gating) |
| d_model | 256 |
| d_ff | 256 |
| Activation | GELU |
| Init | Xavier uniform |
| Optimizer | AdamW (1e-4, weight decay 1e-4) |

## Performance

| Metric | Value |
|--------|-------|
| Feature build (first run) | ~1.5 min |
| Per epoch (MSE) | ~8-9s |
| Per epoch (MSRR, grad-accum=4) | ~9-10s |
| Full train (MSRR, ~40 epochs) | ~6.5 min |
| Inference (single date) | ~2s |
| GPU memory per batch | ~17 MB |

## Hardware

Auto-detects and uses: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU. Mixed precision training supported on both CUDA and MPS.

### Multi-GPU (DDP)

When multiple GPUs are available, use `torchrun` to launch DistributedDataParallel:

```bash
torchrun --nproc_per_node=N uv run python main.py --mode train
```

DDP auto-detects via `dist.is_initialized()` — no flags needed. Under DDP:
- `DistributedSampler` shards the training data across GPUs
- `unwrap_model()` is used for checkpoint save/load to keep files portable across DDP and non-DDP
- Checkpoint saves and prints are gated to rank 0 only (no file races)
- Per-seed checkpoint paths keep `--seeds N --resume` from leaking state between seeds
- `TemporalOrderHead` is also wrapped in DDP during pre-training so all ranks evolve the same head

## Terminology

### MSE vs MSRR Loss

**MSE** — predicts each stock's next-day return. Loss: `(predicted - actual)²` per stock. Simple, clean rankings.

**Portfolio MSE** — penalizes deviation of the portfolio return from 1.0. Loss: `(1 - w'R)²`. Encourages prediction*return correlation. A heuristic approximation — does not directly optimize Sharpe ratio (variance is ignored). Formerly called "MSRR" in earlier versions.

**Margin ranking** — pairwise loss, encourages correct ordering of stock returns. Good for ranking tasks.

**ListNet** — listwise loss, optimizes top-1 probability distribution. Good risk-adjusted returns.

### Mixed Precision

Uses float16 for compute-heavy operations (linear, matmul) while keeping critical ops (softmax, norm) in float32. Speeds up training ~30-40% on both CUDA and MPS with no accuracy loss.

### Encoder-Only

Uses full bidirectional attention — every stock attends to every other stock. Research shows this architecture captures cross-stock relationships effectively.

### RankGLU

Replaces the linear output head with a residual bottleneck GLU: a direct linear path + a bounded nonlinear branch. Preserves stable ordering while adding controlled interactions. From the RankGLU paper (arXiv 2606.08930).

### Market-State Gating

SPY market features (returns, volatility) are used to rescale each stock's features before the transformer. This lets the model adapt to bull/bear/high-volatility regimes. From MASTER (AAAI 2024).

## Paper Trading

Model scores can be evaluated in real-time via Alpaca's paper trading API (commission-free, $100K virtual account).

### Setup

1. Sign up at [alpaca.markets](https://alpaca.markets) → Dashboard → API Keys
2. Generate **paper** API keys (not live)
3. Copy `.env.example` to `.env` and fill in your keys:
   ```
   ALPACA_API_KEY=pk_...
   ALPACA_SECRET_KEY=...
   ALPACA_PAPER=True
   ```

### Run

```bash
uv run python trade.py --interval 15           # Rich live display (Dracula theme)
uv run python trade.py --interval 15 --headless  # Logs only
uv run python main.py --mode trade --trade-interval 15  # Via main.py
```

The trading loop: run inference on the latest cached business day → get BUY/SELL/HOLD signals → reconcile with Alpaca paper positions → display P&L → repeat every N minutes.

### Reconciliation logic

Each cycle, `PaperTrader.reconcile()` does the following:

1. **Bulk cancel:** calls `cancel_orders()` to clear ALL open open orders — a single API call per cycle instead of per-ticker. Cancelling stale HOLD-ticker orders is harmless; they won't be re-filled since reconcile only acts on BUY/SELL signals.
2. **Quote fetch:** for every fresh BUY ticker, fetches a latest ask quote. Skipped entirely if no new buys are needed.
3. **Portfolio cap:** if total existing position market value exceeds `equity * max_portfolio_pct` (default 50%), new BUYs are blocked with `PORTFOLIO_CAP`.
4. **Position cap:** `qty * ask_price` compared against `equity * trade_max_position_pct` (default 2%). Trades breaching the cap are skipped with `MAX_POS_CAP`; trades with no usable ask get `NO_ASK`.
5. **No-equity guard:** if `account.equity <= 0`, all BUYs blocked with `NO_EQUITY`.
6. **Partial close:** SELL sells `min(held, trade_sell_qty)` — for a 1000-share position with `trade_sell_qty=20`, only 20 are sold. Smaller positions close fully.
7. **Fractional shares:** `math.floor(abs(pos["qty"]))` for sell qty — prevents over-selling fractional positions. `round()` could flip a short into a long (e.g. covering 3.9 of a -4 short with round(3.9) = 4 leaves a spurious +0.1 long).
8. **Trade audit:** every trade is appended to `data/paper_trades.csvl` via `PaperTrader._audit()` — survives crashes, shared by all callers.
9. **Failure capture:** any rejected order is logged as `<action>_FAIL:<exception>` instead of crashing the cycle.

### Walk-Forward Validation

Multiple chronological folds (train → val → test), trained and evaluated sequentially. The gold standard for financial ML — captures regime changes and exposes overfitting.

### Cross-Sectional Z-Score

Training targets are normalized per trading day (mean=0, std=1) after winsorizing extremes. Makes scores comparable across stocks. Standard in ranking-aware models.
