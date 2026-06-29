# Fix Plan — Outstanding Issues

**Audit date:** 2026-06-28 | HEAD: `b86d894`
**Baseline:** `ruff check` ✅ · `ruff format --check` ✅ (31 files) · `pytest` 79/79 ✅ · `mypy` **34 errors**

All prior-audit items (UX-N1/N3, N-RETRY-1/2, N-MODEL-META-1, N-DATAQ-1, N-PORT-CAP-1, N-LIVE-LOCKOUT-1, N-KILL-1, N-OBSERVE-1, N-LOG-1, etc.) are verified resolved. This plan covers only **new findings** from a fresh end-to-end codebase audit.

---

## CRITICAL

### 1. Colab script crashes with ImportError — trading/UI deps imported at module level

**Where:** `main.py` lines 22–23.

**What happens:** `main.py` has these **top-level** imports that execute at module load before `main()` is even called:

```python
from src.paper_trader import PaperTrader, setup_logger   # → requires alpaca-py, pytz
from trade import build_layout, make_trade_table          # → requires rich
```

`src/paper_trader.py` imports `alpaca.data`, `alpaca.trading` at module level.
`trade.py` imports `rich.console`, `rich.live`, etc. at module level.

When the Colab script runs `exec(open("main.py").read(), globals())` in **training mode**, these imports execute immediately → **`ModuleNotFoundError: No module named 'alpaca'`** — even though training never uses trading or UI functionality.

The root cause: trading/UI dependencies are loaded unconditionally for ALL modes (`train`, `infer`, `pretrain`, `trade`). They should only load when `--mode trade` is requested.

**Fix:** Lazy-import `PaperTrader`/`setup_logger` and `build_layout`/`make_trade_table` inside `run_paper_trading()` instead of at module level. Also move `setup_logger` from `src/paper_trader.py` to `src/utils.py` so the logger setup (needed in all modes) doesn't pull in alpaca-py. ~15 min.

```python
# main.py — remove these top-level imports:
#   from src.paper_trader import PaperTrader, setup_logger
#   from trade import build_layout, make_trade_table

# main.py — add lazy imports inside run_paper_trading():
def run_paper_trading(config, args):
    from src.paper_trader import PaperTrader
    from trade import build_layout, make_trade_table
    ...

# src/utils.py — move setup_logger here (imports logging only, no alpaca-py)
# main.py — import setup_logger from src.utils instead
```

---

### 2. Walk-forward training crash in threshold optimization

**Where:** `main.py` lines 575–615.

**What happens:**

```
main.py --mode train --walk-forward:
  orig_save_path = config.model_save_path   # "data/models/best.pt"

  for fold in range(n_folds):
      config.model_save_path = f"best_fold{fold}.pt"   # temporarily reroute
      run_training(config, ...)                          # saves to best_fold{fold}.pt ✅
      config.model_save_path = orig_save_path           # restore to "best.pt"

  # CRASH: load_model(config) tries torch.load("data/models/best.pt")
  # best.pt was NEVER written during walk-forward!
  run_threshold_optimization(config)   # FileNotFoundError
```

`run_training` only saves to `config.model_save_path`. During walk-forward that path points to `_fold{N}.pt` files. The master `best.pt` is never created. `run_threshold_optimization` calls `load_model(config)` which tries `torch.load("data/models/best.pt")` → **crash**.

If a stale `best.pt` exists from a prior non-walk-forward run, it silently loads an **unrelated** model — silently-wrong Sharpe scores.

**Fix:** After the fold loop, promote the best fold model to `best.pt`. ~20 min.

```python
# After fold loop, before run_threshold_optimization:
import shutil
best_fold_path = None
for fold in range(fold_count):
    fp = Path(orig_save_path).with_name(f"best_fold{fold}.pt")
    if fp.exists():
        best_fold_path = fp
if best_fold_path:
    shutil.copy2(best_fold_path, orig_save_path)
```

---

### 3. Colab pretrain chaining doesn't work — trains from scratch twice

**Where:** `src/colab_gen.py` generated script, pretrain block.

**What happens:** When `--colab-template --pretrain` is used, the generated script does:

```python
# Run 1: sys.argv includes --pretrain
_run()   # → main.py sees args.pretrain=True
         # → pretrain_path = "data/models/pretrain/best.pt"
         # → BUT this file doesn't exist yet (we haven't pretrained!)
         # → Path(pretrain_path).exists() = False → silently skipped
         # → trains from scratch, saves to data/models/best.pt

# Run 2: sys.argv has --pretrain REMOVED
_run()   # → main.py sees args.pretrain=False
         # → pretrain_path = None
         # → trains from scratch AGAIN, ignoring run 1's weights entirely
```

The intended "pretrain → fine-tune" flow never happens. Both runs train from scratch independently, wasting a full GPU training cycle.

**Fix:** Restructure the generated script so Run 1 uses `--mode pretrain` (saves to `pretrain_weights_path`) and Run 2 uses `--mode train --pretrain` (loads those weights and fine-tunes). ~20 min.

```python
# In generate_colab_script, replace the pretrain block with:
if do_pretrain:
    # Run 1: D6 pre-training
    pretrain_argv = [a for a in flaglist]
    for i, a in enumerate(pretrain_argv):
        if a == "--mode":
            pretrain_argv[i+1] = "pretrain"
    print(f"[{time.time()-start:.0f}s] Pre-training...")
    sv = sys.argv; sys.argv = pretrain_argv; _run()
    print(f"[{time.time()-start:.0f}s] Pre-training done. Fine-tuning...")
    sys.argv = flaglist  # restore original (with --mode train --pretrain)
_run()
```

---

## HIGH

### 4. Causal mask creates alphabetical stock-order bias

**Where:** `models/stock_model.py` lines 99–106.

**The problem:**

```python
stock_ids = torch.arange(self.n_stocks, device=x.device)   # 0, 1, 2, ..., N-1
x = x + self.stock_embed(stock_ids).unsqueeze(0)

causal_mask = nn.Transformer.generate_square_subsequent_mask(self.n_stocks)
x = self.transformer(x, memory=x, tgt_mask=causal_mask, tgt_is_causal=True)
```

Tickers are sorted alphabetically (`get_sp500_tickers()` returns `sorted(...)`). The causal mask means stock 0 ("A") can only attend to itself, while stock 502 ("ZTS") can attend to all 503 stocks. This creates a **systematic informational asymmetry** based purely on ticker name — an arbitrary artifact with no economic meaning.

**Fix:** Shuffle stock indices per forward pass (random permutation). ~30 min.

---

## MEDIUM

### 5. No validation Sharpe monitoring during training

**Where:** `training/train.py` lines 241–280.

Early stopping and model selection use `val_loss` (MSE/ranking loss). But the downstream metric is **Sharpe ratio**, and MSE does not correlate perfectly with Sharpe. A model with better `val_loss` can have worse trading performance. The post-training threshold optimization has no visibility into whether an earlier checkpoint would have been better.

**Fix:** Compute val Sharpe every 10 epochs alongside the checkpoint save. ~45 min.

### 6. Feature cache invalidation only checks raw data, not feature code

**Where:** `src/features.py` lines 277–295.

`_data_hash(raw_data_dir)` hashes raw CSV files (CRC32 + mtime + size). If the feature computation code changes (new indicators, different window logic), the cache does NOT invalidate. Training silently uses stale features.

**Fix:** Include a hash of `features.py` source in the cache key. ~15 min.

### 7. Duplicate paper-trading loop between `main.py` and `trade.py`

**Where:** `main.py` `run_paper_trading()` (40 lines) vs `trade.py` `main()` (80 lines).

Identical orchestration logic in two places: configure `PaperTrader`, load thresholds, `while True` loop with `market_open()` check, `run_inference` + `reconcile`, Rich rendering. `main.py` even imports `make_trade_table`/`build_layout` from `trade.py`. Every bug fix must be made twice.

**Fix:** Have `main.py --mode trade` delegate to `trade.main()`, or extract the shared loop. ~45 min.

### 8. Colab script installs `pyperclip` but never uses it

**Where:** `src/colab_gen.py` generated script, Colab pip install line.

`pyperclip` is installed on Colab but the generated script never calls it (it's only used locally for clipboard copy). Wasteful install.

**Fix:** Remove `pyperclip` from the Colab install line. ~1 min.

---

## LOW

### 9. Vestigial `Console(theme=_THEME)` in `trade.py:132`

```python
if not args.headless:
    Console(theme=_THEME)  # init theme for Rich
```

Creates and discards a Console. Rich themes are per-console — this line has zero effect. ~1 min fix.

### 10. mypy 34 errors — known noise floor

**By file:** `src/paper_trader.py` (27), `src/crypto_pipeline.py` (2), `textual_trader.py` (1), tests (2), utils (1), data_pipeline (1).

All SDK-type-level mismatches. Project converged on `# type: ignore`. No runtime impact.

---

## Architecture / Data Improvements (future-phase)

| ID | Improvement | Rationale |
|----|-------------|-----------|
| **A2** | `listnet_loss` temperature (0.1) too aggressive | `target / 0.1` does hard top-1 selection, losing distribution-matching. |
| **A3** | Survivorship bias in S&P 500 universe | Current constituents only — delisted companies 2015–2025 excluded. |
| **A4** | No data augmentation | Gaussian noise, block-bootstrap, or regime-shift augmentation. |
| **A5** | Walk-forward: no ensemble of fold thresholds | Each fold has own threshold; inference uses single threshold. |
| **A6** | Pretrain `drop_last=True` on all loaders | ~3% data loss for small crypto datasets. |
| **B4** | `margin_ranking_loss` O(n²) memory | (batch, 500, 500) tensors — ~128 MB/batch. Scales poorly. |

---

## Recommended PR ordering

| Priority | PR | Items | Effort |
|----------|----|----|--------|
| **1 (now)** | Fix Colab generation | #1 (lazy-import deps) + #3 (pretrain chaining) + #8 (pyperclip) | ~30 min |
| **2 (now)** | Fix walk-forward crash | #2 (promote fold model) + A5 (fold threshold ensemble) | ~30 min |
| **3 (today)** | Architecture bias fix | #4 (stock order randomization) | ~30 min |
| **4 (this week)** | Training observability | #5 (val Sharpe) + #6 (feature hash) | ~1 hr |
| **5 (this week)** | De-duplicate trading loop | #7 (merge main.py/trade.py) | ~45 min |
| **6 (whenever)** | Cosmetic cleanup | #9 (vestigial Console) | ~1 min |
