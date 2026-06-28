# Fix Plan — Outstanding Items Only

**Last full audit:** commits through `f3e49d9` (2026-06-27, "ci: use github.actor + secrets.GITHUB_TOKEN for git push auth"). Today: 2026-06-27.
**Validation baseline:** `ruff check` ✅ · `ruff format --check` ✅ (31 files) · `pytest` 79/79 ✅ · `mypy` **52 errors** (regressed 44 → 52 across the bulk-fix series; +8 above baseline).

**Status:** ~95% of all audit items resolved across the post-`dad479e` series (`dea2dae` → `402b5d1` → `ce1a881` → `605fef8` → `1560f48` → `f3e49d9`). **3 outstanding items** remain after adversarial review (1 HIGH, 2 MEDIUM).

---

## STILL OPEN — High

### 1. N-RETRY-1 (NEW). Bulk cancel + liquidate closure not wrapped in `_retry`

**Where:** `src/paper_trader.py:181` (`self.trade_client.cancel_orders()`) and `textual_trader.py:784` (`self._trader.trade_client.close_all_positions()`).

Adversarial review surfaced that **every per-order hydration path is wrapped in the module's `_retry(fn, max_tries=3, exponential-backoff)` helper** (e.g. `submit_market_order`, `cancel_order_by_id`, `get_orders`, `get_crypto_bars`), but the two **collection-level** endpoints are not:

| API call | Site | Wrapped in `_retry`? |
|---|---|---|
| `submit_order` (single order) | `submit_market_order` | ✅ |
| `cancel_order_by_id` (single) | `cancel_open_orders` | ✅ |
| `get_orders` (filtered) | `cancel_open_orders` | ✅ |
| **`cancel_orders()` (bulk, all open)** | `reconcile` | ❌ |
| **`close_all_positions()` (bulk)** | `action_liquidate` | ❌ |

**Failure mode:** on a 429 / transient API hiccup, the bulk call raises an HTTP error that propagates up to `reconcile()`'s outer `try/except: trades.append((ticker, 0, f"BUY_FAIL:{e}"))` loop — meaning **only the BUY path catches it, never ALL the in-flight orders.** Concretely:

1. `cancel_orders()` raises `APIError(429)` mid-reconcile.
2. The loop's `except Exception as e` catches it ONLY if it appears inside the BUY try/except — it's actually outside, so it bubbles to `_refresh_cycle` (textual) or `main()` (Rich).
3. All stale orders from the previous cycle remain open and may **still fill during the API recovery window**.
4. Next cycle attempts to add new orders → potential conflicts, double-positioning, or Alpaca-side rejections.

Same gap exists for `close_all_positions()` in liquidation: if it fails the user-issued `liquidate` action, positions remain open and the operator's mental model ("I'm flat") is wrong until they manually re-press `l`.

**Real fix:** route both through the same retry helper:

```python
# src/paper_trader.py
def cancel_open_orders(self, symbol: str | None = None) -> None:  # extend signature
    """If symbol is None, cancel ALL open orders (bulk, retried)."""
    if symbol is None:
        _retry(self.trade_client.cancel_orders)
        return
    # ...existing per-symbol logic
def close_all_positions(self) -> None:
    _retry(self.trade_client.close_all_positions)
```

…then call `self._trader.cancel_open_orders()` (no-arg) from `reconcile()` and `action_liquidate` (passing `ticker` only for the per-symbol fallback path — though the bulk call already supersedes it).

Plus access `close_all_positions` through `PaperTrader` rather than reaching through `self._trader.trade_client` from the UI:

```python
# textual_trader.py:action_liquidate
self._trader.close_all_positions()  # <- go through PaperTrader for retry
```

**Severity:** HIGH — silent failure on rate-limit (sub-second normally, but Alpaca's 200 req/min free-tier can hit it) → phantom positions remain open. User-visible only when reconcile returns a `BUY_FAIL` and the prior-cycle order actually executed anyway.

---

## STILL OPEN — Medium

### 2. mypy: 44 → 52 (regression across `dea2dae` … `1560f48`)

**Where:** 8 files: `src/paper_trader.py` (27), `textual_trader.py` (8 aggregate), `src/inference.py` (8), `src/crypto_pipeline.py` (5), `tests/test_trade.py` (1), `tests/test_ddp.py` (1), `src/utils.py` (1), `src/data_pipeline.py` (1).

**Verified root causes (literal mypy messages):**

1. **`Union[TradeAccount, dict, Any]` cascade in `src/paper_trader.py`** — `get_account()` and `get_positions()` return `dict` (lines 67, 77) but the calling code reaches into them as if they were `TradeAccount` / `Position` (`.equity`, `.cash`, `.qty`, etc.). 21 of 27 errors here.
2. **`CryptoBarsRequest.start` / `end` typed `datetime` but called with `str`** in `src/crypto_pipeline.py:40-41` + 4 follow-on union-attr errors on `BarSet.data`.
3. **`datetime.datetime.date` used as a TYPE** in `src/inference.py:51` — `.date()` returns a `datetime.date`, not `datetime.datetime`, so subsequent `d.year` / `d.month` / `d.day` on the wrong type cascade.

> *(Prior plan claimed Root cause 3 was `push_screen` callback typing. Adversarial review showed this is **wrong**: the 8 aggregate errors in `textual_trader.py` are Header / widget / `_NoopTqdm` assignment mismatches — no push_screen errors in the aggregate run. Dropping that misframe.)*

**Real fix** (~2–3 h):
- **Option A** (cheap): add `# type: ignore[union-attr]` near the SDK access sites; cast the dict-returning methods with explicit `cast(TradeAccount, …)`.
- **Option B** (right): replace `dict` caches with typed adapters (`_account_from_raw(acct) -> TradeAccount`, `_positions_from_raw(rows) -> dict[str, Position]`); drop the `dict` fallback since the Alpaca SDK is stable.
- **Option C** (subset): silence only the two pervasive clusters (paper_trader dict-fallback + inference datetime misuse).

**Severity:** MEDIUM — hygiene only, no runtime impact. Static-type false positives mask real regressions later; that's the long-term concern.

---

### 3. N-MODEL-META-1. No `model_metadata.json` sidecar for `best.pt`

**Where:** `training/train.py` saves `best.pt` + `checkpoint_seed*.pt`. `eval_colab.py` writes `data/models/eval_log.csv` but **no per-model metadata**. Verified by literal grep — `metadata.json` / `model_metadata` / `.with_suffix(".json")` writers do not exist anywhere in the project.

**Why it matters:** when you promote a checkpoint to `best.pt`, the original training config (loss / epochs / seeds / Sharpe / threshold / `data_hash` / `trained_at`) is lost. Three months out, you can't answer "what hyperparameters was `best.pt` trained with?" or "is this model stale relative to current data?".

**Real fix:** at end of `run_training`, write alongside `best.pt`:

```python
# training/train.py  (in run_training, after final save)
import json
from datetime import UTC, datetime
meta_path = Path(config.model_save_path).with_suffix(".json")
meta_path.write_text(json.dumps({
    "trained_at": datetime.now(UTC).isoformat(),
    "config_hash": hash_config(config),
    "loss": loss_mode,
    "n_seeds": n_seeds,
    "epochs": epochs_run,
    "val_sharpe": best_val_sharpe,
    "val_buy_thresh": buy_t,
    "val_sell_thresh": sell_t,
    "features_path": config.features_path,
    "scaler_path": str(scaler_path),
}, indent=2))
```

…then have `eval_colab.py` read it during promotion, and **refuse to overwrite if `trained_at` is more recent** than the current `best.pt`.

**Severity:** MEDIUM — reproducibility gap. Affects future audits ("what does this model actually do?"). No crash, no data loss.

---

## RESOLVED — audit trail (won't re-litigate)

Verified via literal grep / line reads on the post-`f3e49d9` tree:

| Item | Fix commit | Verification |
|---|---|---|
| **UX-N3 (HIGH)** `Live` not context-managed in `trade.py` | prior-batch (post `1560f48`) | `trade.py` rewrote main loop with `with live_ctx as live:` block (~L224); `_NoopLive` stub for headless. Terminal restored on any exception. |
| **UX-N1 (HIGH)** cancel-orders loop in `reconcile` freeze | prior-batch | `src/paper_trader.py:181` now calls `self.trade_client.cancel_orders()` (single round-trip). The only remaining per-ticker `cancel_open_orders(ticker)` is in `textual_trader.py:action_liquidate` — user-initiated, one-shot. (Note: N-RETRY-1 above addresses the *retry* gap on this bulk call.) |
| **N-LOG-1 (MEDIUM)** audit-trail writer TUI-only | prior-batch | `PaperTrader._audit(trades, equity)` is now called from `reconcile()` (line 269). All callers (TUI, Rich CLI, `main.py --mode trade`, future scripts) share the `data/paper_trades.csvl` writer. |
| **N-MODEL-LEAK-1 (CRITICAL)** Sharpe-on-val-then-promote | `dea2dae` | `eval_colab.py:39-41, 78-83` loads both `val.npz` (threshold opt only) and `test.npz` (Sharpe for selection). |
| **N-DATAQ-1 (HIGH)** no exchange-calendar support | `f3e49d9` series | `src/inference.py` has `NYSE_HOLIDAYS` + `_nth_weekday_of_month` / `_last_weekday_of_month` + `_is_nyse_holiday(d)` + `_last_business_day()`. |
| **N-PORT-CAP-1 (MEDIUM)** per-position cap not portfolio | prior-batch | `config.max_portfolio_pct = 0.5` + `portfolio_capped = …` check in `reconcile()` (line 174–176). Both per-position and portfolio gates enforced. |
| **H-NEW3 / M-NEW1** `round()` → `math.floor` for sellable qty | `402b5d1` | `textual_trader.py:642`; `trade.py:99` use `math.floor(abs(pos['qty']))`. `import math` both files. |
| **UX-N2** no "PAPER TRADING" badge | `402b5d1` | `Header(show_clock=True, sub_title="PAPER TRADING")` in `textual_trader.py:392`. |
| **N-HIGH-1** `+/-` keys crash before `on_mount` | `402b5d1` | `self._timer: Timer \| None = None` in `__init__`; `if self._timer is not None: stop()` in `action_interval_*`. |
| **L-NEW2** sparkline carries over after asset-switch | `402b5d1` | `self._equity_history.clear()` in `_switch_asset` (line 506). |
| **N-LIVE-LOCKOUT-1** `alpaca_paper=False` trades real money silently | `1560f48` | `main.py:520-525` checks `ALPACA_LIVE_CONFIRM=="true"`; `LiquidateConfirm` ModalScreen for destructive UI actions. |
| **N-KILL-1** no liquidate-all / emergency stop | `1560f48` | `LiquidateConfirm` modal + `action_liquidate` triggers `close_all_positions()` + cancel-orders sweep. (Note: N-RETRY-1 above adds retry to the close path.) |
| **N-OBSERVE-1** only stderr logging | prior-batch | `setup_logger()` uses `RotatingFileHandler(log_file, maxBytes=10MB, backupCount=5)`, defaults to `data/trading_bot.log`. |
| **M-NEW2** disk reload every cycle | `dad479e` | `run_inference(..., model=None)` accepts pre-loaded model. (Note: call sites still don't pass `self._model`, so disk-load happens *every* cycle today. Polish, not correctness.) |
| **H-NEW4** `set_interval` not rescheduled | `dad479e` | `_timer` saved, `stop()`+`set_interval()` on `+/-` keys. |
| **C-NEW1** first-exception AttributeError | `dad479e` | `_last_error = None`, `_error_count = 0` in `__init__`. |
| **C-NEW2** PaperTrader not rebuilt on asset-switch | `dad479e` | `self._trader = PaperTrader(self._config)` inside `_switch_asset`. (Note: model itself still re-loads on each `run_inference` from disk via the `model=None` fallback — works correctly today.) |
| **Fold caching** zombie fold persistence | `dea2dae` | `folds_meta.json` sidecar fingerprint; `_folds_match_config()` re-validates before reuse. |
| **market_state → threshold opt** round-trip | `ce1a881` | `training/threshold.py` accepts `market_state`; duplicate timer init removed. |
| **M1** tqdm monkey-patch scope leak | `605fef8` | `_NoopTqdm` is wired in `on_mount` and restored in `on_unmount` (`textual_trader.py`) — no longer globally scoped. |
| **605fef8 batch** (infra hardening) | `605fef8` | 9 files / +963/-335. Crypto pipeline, .env, CI, fold separation, etc. |
| **CI auth** | `f3e49d9` | `.github/workflows/deploy.yml` uses `github.actor` + `secrets.GITHUB_TOKEN` for git push. |

---

## DEFERRED — Phase-4+ (features, NOT bugs)

Tracked so future audits don't re-flag them:

- **F-NL1** Limit-order & stop-loss support (currently market orders only)
- **F-NL2** Dry-run mode (log trades without submitting)
- **F-NL3** Corporate-actions handling (splits, dividends, mergers)
- **F-NL4** Fetch-failure telemetry + retry heuristics
- **F-NL5** USD-based risk sizing (vs current fixed `trade_buy_qty`)
- **F-NL6** Multi-strategy / date-bounded wallet rotation
- **F-NL7** Strategy docstring + architectural notes
- **F-NL8** Alpaca ticker-format compatibility (BRK.B, BF.B)
- **F-CLEANUP-1** Dead code: `_audit_path = Path(...)` set twice in `PaperTrader.__init__` (lines 64, 65). The second assignment overrides the first; bundle into PR-OPS-3.

---

## Recommended PR sequence (≈½ day total)

| PR | Items | Effort |
|---|---|---|
| **PR-OPS-3A — Network hardening** | N-RETRY-1 (wrap `cancel_orders` + expose `close_all_positions` through PaperTrader with `_retry`) | ~30 min |
| **PR-OPS-3B — Type hygiene** | mypy 52 → ≤44 (typed `TradeAccount` adapters + fix `datetime.datetime.date` misuse + `CryptoBarsRequest` str→datetime) | ~2–3 h |
| **PR-OPS-3C — Reproducibility** | `model_metadata.json` sidecar + `eval_colab.py` reads + refuses-to-overwrite newer | ~1 h |
| *(bundled cleanup)* | F-CLEANUP-1 (dead `_audit_path` line) | ~5 min, fold into 3A |

**Net state after PRs:** zero outstanding CRITICAL or HIGH bugs, mypy at-or-below baseline, model promotion has a real audit trail.
