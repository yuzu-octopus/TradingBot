# Fix Plan — Full Audit of Recent Changes

**Scope:** the 25 most recent commits (Textual TUI, crypto pipeline, Alpaca/Colab, Rich UI, DDP, threshold/inference refactors).
**Validation baseline:** `ruff check` ✅ · `ruff format --check` ✅ (28 files) · `pytest` 75/75 ✅ · `mypy` 48 errors (mostly import-untyped noise from unstubbed libs).

---

## CRITICAL (silent correctness regressions)

> _Findings here don't show up in tests, but will fire under realistic conditions._

### C1. `_fold_metadata()` fingerprint is missing `tickers` and `asset_class` — walk-forward cache can be silently reused across universe changes

**Where:** `main.py` · `_fold_metadata()` (≈lines 38-50).

**Issue:** `_fold_metadata` covers `wf_window_size`, `wf_val_size`, `wf_test_size`, `wf_step_size`, `train_start`, `test_end`, `label_max_return`, `n_features`. But `n_features = features_per_window * n_windows` does **not** depend on the ticker universe. So if you train S&P 500 (503 stocks) once with `--walk-forward`, then run `--walk-forward` against `--crypto-pairs all17` (17 pairs) and don't `--force-features`, the fingerprint matches — and the cached `fold_i_train.npz` slices contain a `(T, 503, 120)` array reshaped for the new `(T, 17, 120)` model. Downstream either crashes on `Emb(n_stocks)` mismatch, or worse, indexes garbage.

**Minimal fix:** add to the dict returned by `_fold_metadata`:
```python
"asset_class": config.asset_class,
"crypto_pairs": config.crypto_pairs,
"n_stocks": config.n_stocks,
```
**Severity:** **CRITICAL** — silent data corruption under realistic toggle scenarios.

---

### C2. `top_head` DDP wrap in `training/pretrain.py` has no CUDA-only guard — but currently masked by `create_model` raising first

**Where:** `training/pretrain.py` (≈lines 88-92).

**Issue:**
```python
top_head = TemporalOrderHead(...).to(device)
if is_distributed():
    top_head = nn.parallel.DistributedDataParallel(
        top_head, device_ids=[device.index] if device.type == "cuda" else None,
    )
```
There's no `if device.type != "cuda": raise`. The same bug class as the one previously fixed in `src/utils.create_model`. In practice it is masked because `model = create_model(config, device)` runs first and raises on MPS+DDP before reaching this line. So we'll only crash here if someone refactors the order or skips `create_model`.

**Minimal fix:** mirror the `create_model` guard:
```python
if is_distributed():
    if device.type != "cuda":
        raise RuntimeError(f"DDP requires CUDA. Got device={device}.")
    top_head = nn.parallel.DistributedDataParallel(top_head, device_ids=[device.index])
```
Or factor a `_wrap_ddp(module)` helper in `src/utils.py` so both call sites share the guard.

**Severity:** **CONDITIONAL CRITICAL** — false positive today, a footgun the moment the call order changes.

---

## HIGH (real bugs likely to fire)

### H1. `time.sleep(min(wait, 300))` will raise `ValueError` when `next_open < now`

**Where:** `trade.py` line 221, `main.py` line 264.

**Issue:** Right at market close, `clock.next_open` is sometimes briefly in the past while the clock state is being updated. `wait = (nxt - now).total_seconds()` then returns a negative number. `min(-1, 300) = -1`. `time.sleep(-1)` raises `ValueError: sleep length must be non-negative`. The whole trading loop dies instead of just yielding for 5 minutes.

**Minimal fix:** clamp both ends.
```python
time.sleep(max(0.0, min(wait, 300)))
```
**Severity:** **HIGH** — fires deterministically at market-close boundary conditions.

---

### H2. `crypto_pipeline.fetch_crypto_data` writes crypto feature cache to the same fixed path as stocks — feature matrix is silently overwritten when switching asset classes

**Where:** `src/features.py` lines 272-273.

```python
FEATURE_CACHE_PATH = "data/features/matrix.npz"
HASH_CACHE_PATH = "data/features/matrix_hash.txt"
```
Even when `config.features_path = "data/crypto/features"`, both `save_cached_features` and `load_cached_features` write and read from this fixed root path. Asset-class switch cleanly via the hash mismatch (`_data_hash` uses `raw_data_dir` so the hash differs), but every switch triggers a **full 30-minute rebuild** because the previously cached asset's feature matrix has been overwritten with the new one.

**Minimal fix:** parameterize cache path from the caller:
```python
def save_cached_features(features, tickers, dates, raw_data_dir, cache_pathdir):
    ...
    Path(cache_pathdir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{cache_pathdir}/matrix.npz", ...)
    Path(f"{cache_pathdir}/matrix_hash.txt").write_text(_data_hash(raw_data_dir))
```
Pass `cache_pathdir=config.features_path` from `main.py`/`textual_trader.py`. Remove the module-level constants.

**Severity:** **HIGH** — every asset-class switch costs ~30 minutes of CPU. Compound issue, not a correctness bug.

---

### H3. `[BUY` may execute without a position-cap check when `ask` is `None`

**Where:** `src/paper_trader.py` lines 167-184 (`reconcile`).

```python
ask = quotes.get(ticker, {}).get("ask")
if ask is None or ask <= 0:
    logger.warning("No usable ask for %s; skipping position-cap check", ticker)
elif qty * ask > max_pos_value:
    trades.append((ticker, 0, "MAX_POS_CAP"))
    continue
try:
    self.submit_market_order(ticker, qty, OrderSide.BUY)   # ← still submits!
    ...
```
The "skip the cap check" branch actually **falls through to a buy**. The test `test_position_cap_misses_ask_logs_warning` documents this as "paper mode" behavior, but the logic is exactly backwards: when a cap check cannot be performed, the safe default is to **not trade**, not to trade. Setting `config.alpaca_paper = False` (live trading) would still submit un-capped orders.

**Minimal fix:** invert the fallback.
```python
if ask is None or ask <= 0:
    logger.warning("No usable ask for %s; refusing to buy without position-cap check", ticker)
    trades.append((ticker, 0, "NO_ASK"))
    continue
```
And update `test_position_cap_misses_ask_logs_warning` to expect `"NO_ASK"` instead of `"BUY"`.

**Severity:** **HIGH** — specifically bad in the live-trading failure mode (silent over-allocation).

---

### H4. `get_orders()` in `cancel_open_orders(symbol=None)` path cancels ALL orders — dead but dangerous

**Where:** `src/paper_trader.py` lines 102-117.

```python
def cancel_open_orders(self, symbol: str | None = None) -> None:
    try:
        if symbol:
            orders = self.trade_client.get_orders(
                filter=GetOrdersRequest(symbols=[symbol] if symbol else None)
            )
        else:
            orders = self.trade_client.get_orders()    # ← blanket
    ...
```
The `if symbol:` branch already guarantees `symbol` truthy, so the inner ternary `symbols=[symbol] if symbol else None` is dead. The function is currently only called with concrete tickers from `reconcile`, so it's a no-op risk — but the public `symbol: str | None = None` signature advertises "blanket cancel" as a feature. A future caller (e.g., a daily-flatten utility) would accidentally cancel everything.

**Minimal fix:** tighten signature and remove the dead branch.
```python
def cancel_open_orders(self, symbol: str) -> None:
    orders = self.trade_client.get_orders(filter=GetOrdersRequest(symbols=[symbol]))
```
Update `test_cancel_does_not_call_blancket_cancel` (also typo: "blancket" → "blanket").

**Severity:** **HIGH** — API surface bug; latent footgun.

---

## MEDIUM

### M1. `optimize_threshold` runs an O(candidates²) grid search that grinds on large universes

**Where:** `training/threshold.py`.

With `--loss msrr` and the default `upper = max(0.5, min(max_abs, 2.0))`, the loop does up to `200 × 200 = 40_000` iterations; each one calls `np.where(...)` on a `(T, S)` array. For S=503 stocks × T≈300 days × ~40k iterations → ~6 B element ops; expect ~20-40 s of main thread blocking per training run.

**Minimal fix:** vectorize the inner loop, OR coarsen the grid (acceptable: 0.05 spacing), OR cache `(signals == 1).sum(axis=1)` once per candidate.
```python
# Quick win: grid spacing 0.05 instead of 0.01
candidates = np.arange(0.0, upper + 0.05, 0.05)
```
This drops the inner loop from 40k → 1.6k iterations, costing some threshold-resolution precision (<0.05 in worst case, fine for Sharpe-optimal tuning).

**Severity:** **MEDIUM** — UX/perf, not correctness.

---

### M2. DDP scaler comment is wrong; the broadcast is structurally redundant

**Where:** `training/train.py` lines 109-122.

```python
if is_distributed():
    import torch.distributed as dist
    # Each rank independently fit its own StandardScaler on its data
    # partition.
    mean_t = torch.tensor(scaler.mean_, dtype=torch.float32)
    var_t = torch.tensor(scaler.var_, dtype=torch.float32)
    dist.broadcast(mean_t, src=0)
    dist.broadcast(var_t, src=0)
    scaler.mean_ = mean_t.numpy()
    ...
```
`train_features` is loaded on every rank from the same `.npz` file before `DistributedSampler` slices the `DataLoader`. So every rank runs `scaler.fit(train_features.reshape(-1, F))` on the **same full data** and gets identical numbers. The broadcast is harmless-but-redundant, and the comment misleads.

**Minimal fix:** drop the broadcast block and rewrite the comment.
```python
# All ranks load the same npz, so scaler.fit sees the full data on
# every rank — results are already identical. No cross-rank sync needed.
```
**Severity:** **MEDIUM** — confusing comment + dead code path; reviewers will misunderstand the design.

---

### M3. `_raw_data_cache` is unbounded — no LRU, no eviction, no per-day cap

**Where:** `src/inference.py` line 16.

```python
_raw_data_cache: dict[tuple, dict[str, pd.DataFrame]] = {}
```
Each entry holds n_tickers × 5-column DataFrames (~50 KB at n_tickers=503). Over a week-long bot run in the same Python process, that grows linearly and is never reclaimed.

**Minimal fix:** either hook a size guard in `run_inference`, or limit to "today only".
```python
if len(_raw_data_cache) > 1:                       # keep only the latest day
    _raw_data_cache.clear()
_raw_data_cache[cache_key] = ...
```
**Severity:** **MEDIUM** — long-running-bot memory creep.

---

## LOW

### L1. `eval_colab` is hardcoded `cuda` else `cpu` — Apple Silicon users fall back to CPU

**Where:** `eval_colab.py` line ~67.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Minimal fix:**
```python
from config import get_device
device = get_device()
```
**Severity:** **LOW** — Apple Silicon eval is slow but converges; works for the target (Colab = CUDA).

---

### L2. `textual_trader` runs each tick interval without worker-exclusivity

**Where:** `textual_trader.py` `_on_timer` and `action_refresh`.

If `run_worker(self._refresh_cycle(), name="cycle")` fires while a previous cycle is still awaiting `loop.run_in_executor`, Textual queues the new worker. Overlapping cycles can issue duplicate Alpaca calls. Also no guard against re-entrancy on `action_refresh`.

**Minimal fix:**
```python
self.run_worker(self._refresh_cycle(), name="cycle", exclusive=True)
```
**Severity:** **LOW** — mitigates UI-API race but doesn't break correctness because `reconcile` is idempotent on Alpaca.

---

### L3. `textual_trader.py` globally monkey-patches `tqdm` at import-time

**Where:** `textual_trader.py` (top of file).

```python
_tqdm_std.tqdm = _NoopTqdm
```
Any module that imports `textual_trader` (e.g. for a quick `python -c "import textual_trader"`) silently replaces the global `tqdm` class. If a test imports it inadvertently, all tqdm progress bars in tests become no-ops, hiding real failures.

**Minimal fix:** scope the patch to the app, install + tear-down in `on_mount`/`on_unmount`. Or guard with `if __name__ == "__main__":` block plus a CLI flag.
**Severity:** **LOW** — debug & test hygiene.

---

### L4. `_NoopTqdm.write` references `sys` but it's only imported inside the top-of-file patch block — works by accident

**Where:** `textual_trader.py` lines 8-32.

`write` uses `print(s, file=file or sys.stderr, ...)` — `sys` is imported at module scope, so it works. But the `File "/Users/yuzu/Documents/.../textual_trader.py" line 32, in <module>` ordering: `import sys` happens after the `_NoopTqdm.write` definition. The diamond is fine because the function body is only evaluated on call, and the import is unconditional before the class is instantiated. Still, future refactors could break this.

**Minimal fix:** move `import sys` closer to its first use.
**Severity:** **LOW** — order-fragility.

---

### L5. `cfg.tickers = list(range(val_features.shape[1]))` in `eval_colab` is a stopgap

**Where:** `eval_colab.py` line 60.

Trade-off: works because `n_stocks = len(self.tickers)` and `stock_embed` tolerates `[0, n-1]` integer indices. But it's confusing — readers see `cfg.tickers = [0, 1, 2, ...]` and the real ticker-name support assumes strings.

**Minimal fix:** introduce a `set_n_stocks(cfg, n)` helper that sets the property without polluting `self.tickers`. Or hold a temporary `Config` subclass.
**Severity:** **LOW** — code-clarity.

---

## NITPICK (style / dead code / docs)

- **N1.** `paper_trader.py` `min(held, trade_sell_qty) if held > trade_sell_qty else held` simplifies to `min(held, trade_sell_qty)`. Verbose but correct.
- **N2.** `src/features.py` `_data_hash` hashes filename+mtime+size; doesn't hash CSV content. Replacing a CSV with the same content preserves the hash → cache hit on logically different data. Mitigation: include a CRC of first/last 4 KB.
- **N3.** `tests/test_paper_trader.py` test name `test_cancel_does_not_call_blancket_cancel` has typo "blancket" → "blanket".
- **N4.** `walk-forward` writes `fold_i_test.npz` but no consumer reads it (only `train.npz`/`val.npz` are loaded). ~15% wasted disk on every walk-forward build.
- **N5.** `run_threshold_optimization` writes `threshold.txt` non-atomically — if process is killed mid-write, file is empty; next reader parses it as a single empty string and returns `(0.0, 0.0)`. Use `tmp.rename(target)`.
- **N6.** `src/utils.load_scaler` doesn't validate JSON structure — corrupted file → silent default of `len(mean_) = 0` propagates downstream. Add `assert len(data["mean"]) == n_features_in`.
- **N7.** `reconcile` checks `BUY and not has_pos` — duplicates a position already held (e.g., BUY signal returns 10 of AAPL when 5 are already held, ending up with 15). Tolerable for momentum-style signal averaging, but worth a code comment noting the intentional "no pyramid" rule.
- **N8.** `textual_trader._fstring` usages contain Markdown table-row text that ruff's `tab` rule already accepts — using `_from_url` macros instead would shrink 30 lines.
- **N9.** `cancel_open_orders` passes `filter=GetOrdersRequest(symbols=[symbol])` even when no orders may exist — extra API call per cycle. Could check `len(actionable)` before iterating.
- **N10.** `mypy` 48 errors are dominated by `import-untyped` for `yfinance`/`unlockedpd`/`sklearn`/`alpaca-py`/`textual`. Add `# mypy: disable-error-code=import-untyped` to `pyproject.toml` to silence the noise floor and surface real signal.

---

## FALSE POSITIVES (validated, dropped from plan)

| ID | Candidate | Verdict |
|---|---|---|
| F1.1 | `math.floor` round-trip fractional shares | Confirmed correct, comment-documented |
| F1.2 | `compute_features_for_date` uses crypto `BTC/USD` for `market_state` | Not leakage — same as stocks using SPY which is in universe; intentional |
| F1.3 | DDP scaler rank-local stats | False positive — every rank loads the full npz, `scaler.fit` is identical by construction; the broadcast is redundant but harmless |
| F1.4 | `textual_trader` exhaustive worker coordination | Mitigated by `name="cycle"` singleton |
| F1.5 | `cancel_open_orders` blanket-cancel | Currently unreachable (all callers pass concrete ticker); see H4 above for signature tightening |

---

## UX / UI Improvements (Research-Backed)

> _Background research: Textual widget gallery, K9s/lazygit/helix keyboard conventions, Bloomberg Terminal/TOS/Robinhood table semantics, Alpaca-py streaming patterns. Findings filtered to ones that fit the existing architecture and don't require new tests to validate safety._

### Bucket A — Drop-in Textual Widgets (high impact, low risk, small diff)

| ID | Item | Where | Minimal change | Impact × Risk |
|----|------|-------|----------------|---------------|
| **UX1** | Replace manual `_sparkline()` string-builder with the built-in `Sparkline` widget | `textual_trader.py` — drop `_sparkline` method, replace usage in `_refresh_cycle` status | `from textual.widgets import Sparkline; yield Sparkline(id="equity-spark")` and `spark.data = self._equity_history` on each cycle | **H × L** |
| **UX2** | Use `Digits` for the "Equity" and "Cash" headline numbers — bigger, scan-friendly | `MetricCard` for equity/cash only | Yield `Digits(value)` from a subclass; keep Static for "Day Δ / Positions / Cycle" | **H × L** |
| **UX3** | Add a `RichLog` panel for trade audit trail (`[HH:MM:SS] BUY 10 AAPL @ $X`) | Below the `DataTable` | `from textual.widgets import RichLog`; in `_update_table`, for every executed trade call `log.write_line(...)` | **H × L** |
| **UX4** | `ProgressBar` (indeterminate) for "inference running" — replaces `status.update("Cycle #N — running...")` | Top right of metric-row | `pb = ProgressBar(total=None, show_eta=False)`; advance on each sub-step (account / inference / positions / reconcile) | **M × L** |
| **UX5** | `TabbedContent` for stocks/crypto + threshold tuning tabs (replaces the two `Button`s) | Top asset-row | `with TabbedContent(): with TabPane("Stocks"): ... ; with TabPane("Crypto"): ...; with TabPane("Tuning"): ...` and a `Switch` per tab event | **H × M** |
| **UX6** | `Collapsible` around the threshold-tuning controls (`[`, `]`, `{`, `}`) so the keys are scoped to it | Tuning tab | Yield a `Collapsible(title="Threshold tuning", collapsed=True)` holding the related widgets | **M × L** |
| **UX7** | `MarkdownViewer` for an in-app "Strategy Notes" screen — README-style explainer | New screen via `App.SCREENS` | `app.push_screen("notes")`, content is `README.md` rendered | **L × L** |

### Bucket B — Real-Time Feedback / Staleness (no new infrastructure)

| ID | Item | Where | Minimal change | Impact × Risk |
|----|------|-------|----------------|---------------|
| **UX8** | Bloomberg-style flash on equity change — animate background green/red 300ms when value goes up/down | `MetricCard.watch_value` | Save previous value; `self.set_class(direction, True)` for 0.3s, then reset with `self.set_timer(0.3, ...)` | **H × L** |
| **UX9** | "Last refreshed Xm ago" indicator on the right side of the header bar — turns red if interval + grace exceeded | `Header` right slot | Track `self._last_refresh_ts`; render `Static("[dim]5m ago[/]")`; tween to red if stale | **H × L** |
| **UX10** | Row-level position-size visual: gradient row tint that goes yellow → red as realized position value approaches `trade_max_position_pct` | `_update_table` (per row) | Compute `pos_pct = market_value / equity`; set `row_label = f"{pos_pct*100:.0f}%"` and a CSS class like `pos-warn` / `pos-cap` that backgrounds the row | **H × L** |
| **UX11** | Color-blind-safe pairings: P&L column uses `▲`/`▼` alongside red/green | `_update_table` (P&L formatting) | If `pl > 0`: `[green]▲ +$X[/]`; if `< 0`: `[red]▼ -$X[/]`; unchanged for `0` | **H × L** |
| **UX12** | Tooltips on the metric cards + main buttons | `MetricCard.__init__` + on-mount of `TradingApp` | `self.tooltip = "Account equity (mark-to-market)"`; on Buttons: `tooltip = "Switch to {asset_class}"` | **M × L** |
| **UX13** | De-duplicated error toasts (`self.notify`) | `_refresh_cycle` error branch | Track last error message + count; on repeat show `[red]Connection error (x5)[/]` rather than spamming | **M × L** |
| **UX14** | Hotkey hint footer bar — `lazygit`-style: status bar shows "tab: switch • q: quit • +: interval • [/]: buy threshold" | Replace or augment `Footer()` | Bind `F1` / `?` to `action_show_help` (already h); also add a `Static("#bottom-hint")` row above existing footer | **M × L** |

### Bucket C — Confirmation / Safety UX

| ID | Item | Where | Minimal change | Impact × Risk |
|----|------|-------|----------------|---------------|
| **UX15** | Liquidate-all confirmation modal: `L` opens a `ModalScreen` listing current positions, requires explicit `Y` then `Enter` to flatten | New `LiquidateModal` class + binding `("L", "liquidate", "Liquidate")` | Modal lists positions and unread-key ticks; `Y` + Enter triggers `self.app.cancel_all_and_flatten()` | **H × M** |
| **UX16** | Threshold-bound arrow-key confirmation: when `]` raises buy-threshold past 0.95, ask "Are you sure?" | `action_threshold_up` | Inject `if self._buy_t == 0.99: self.push_screen(ConfirmModal(...))` and only commit on confirmation | **M × M** |
| **UX17** | Disconnect-strike indicator on the market-dot when 3 consecutive cycle errors hit | `_refresh_cycle` error branch | Increment `self._err_strikes`; update dot to `[red]● Disconnected[/]` if >= 3, reset on success | **H × L** |

### Bucket D — Architectural (Bigger Work — Separate Roadmap)

> _These need new test scaffolding, refactoring the data flow, or new dependencies. Listed so they're not lost, but kept out of "Recommended order"._

| ID | Item | Effort | Notes |
|----|------|--------|-------|
| **UX18** | **Alpaca WebSocket**: replace 15-minute REST poll cycle with `TradingStream.subscribe_trade_updates` for instant fill events | 2-3 days | Requires async refactor of the reconcile loop, save-as-state on the order lifecycle (`new → filled / partial_fill / rejected`), and live updates to RichLog + table |
| **UX19** | **Live market data stream**: `StockDataStream` for active position symbols so P&L ticks between inference cycles | 1-2 days | Decoupled from inference; affects existing `_update_metrics` timing |
| **UX20** | **Multi-screen app stack**: `Dashboard` / `Strategy Tuning` / `Logs` / `Backtest` / `Notes` screens, nav via Tab/S-Tab | 1 day | Composes existing widgets; main architectural decision is `App.SCREENS` map |
| **UX21** | **User-configurable column set**: DataTable columns toggleable, sortable, saveable to `~/.tradingbot/layout.json` | 1 day | Needs `ListView` settings panel + persistence |
| **UX22** | **Strategy inspector**: before submitting trades, render side-by-side "Model output vs consensus (top-1 chg% from yfinance)" | 0.5 day | Read-only; useful for trust calibration |

### Bucket E — Quick Misc (≤5 lines each)

- **UX23.** Add `Ctrl+P` binding as a visible hint in `HelpScreen` (Textual's default palette key; users expect it)
- **UX24.** Numeric column alignment: ensure all numeric columns are right-aligned with fixed widths (look at `_update_table` columns `"Pos"`, `"%"`, `"Score"`, `"P&L"`)
- **UX25.** Persist user-set `interval`/`buy_t`/`sell_t` to a `data/last_session.json` so relaunch starts where the user left off
- **UX26.** On `--crypto-pairs top10 → all17` switch in UI, also refresh `equity_history` (currently grows across asset-class switches — stale stocks `$` values displayed under crypto)
- **UX27.** Status bar fault-tolerant: if `next_open()` raises unexpectedly (e.g. network blip), show `[yellow]● Open•?[/]` rather than crashing the cycle (currently `except Exception: ... [yellow]● ?[/]` already present in `_refresh_buttons` — apply consistent treatment to status bar)
- **UX28.** Add divider "Last filled:" line to RichLog feed — clearer audit granularity

---

## Combining bucket-by-bucket

### Bug fixes (out-of-band, P0 quality bar)

1. **C1** (fingerprint) — 5-line change, highest payoff.
2. **H1** (`time.sleep` clamp) — 1-line, fires deterministically.
3. **H3** (BUY-without-cap on `ask=None`) — invert fallback + 1 test update.
4. **H4** (cancel_open_orders signature) — tightens public surface.
5. **H2** (per-asset-class feature cache) — refactor, ~30-minute item.
6. **M1, M2, M3** — quality-of-life.
7. **L1-L5** — drive-by cleanups in the same commit or follow-up.
8. **N1-N10** — opportunistic, pick during PR review.

### UI workstream (separate PRs)

- **PR-UX1**: Buckets A + UX8 + UX9 + UX11 — drop-in widgets + flash + staleness + a11y arrows. Smallest review surface, immediate visual payoff. ~1 day.
- **PR-UX2**: UX15 (Liquidate-modal) + UX16 (threshold confirmation) + UX10 (row coloring) — safety + risk-visualization. ~0.5 day.
- **PR-UX3**: UX14 (hotkey footer) + UX12 (tooltips) + UX23-UX27 (misc a11y/persistence). ~0.5 day.
- **PR-UX4 (architecture)**: UX18 + UX19 — WebSocket migration only after buckets 1-3 ship and land. ~1 week.
- **PR-UX5 (future)**: UX20-UX22 — multi-screen app.

---

## Citations & references

**Textual widgets and conventions**
- Textual docs: <https://textual.textualize.io/widget_gallery/> (DataTable, Sparkline, Digits, TabbedContent, RichLog, Collapsible, ProgressBar)
- Textual reactivity & watch methods: <https://textual.textualize.io/guide/reactivity/>
- Textual Worker / concurrency: <https://textual.textualize.io/guide/workers/>
- ModalScreen & Screen stacking: <https://textual.textualize.io/guide/screens/>

**Trading-platform UX conventions**
- Bloomberg Terminal / ThinkOrSwim: dense tabular data, color semantics (green gain / red loss), fixed-width numeric alignment
- Robinhood / Public: clean order ticket, transparent Daily P&L
- QuantConnect: integrated Backtest / Live Logs / Equity chart paneling

**TUI keyboard-first patterns**
- K9s (gold standard for context-sensitive `?` help + Tab/Enter confirm): <https://github.com/derailed/k9s>
- Lazygit (exemplar of TUI menu / sub-command navigation): <https://github.com/jesseduffield/lazygit>
- Helix (modal key-mapping, selection-based ops): <https://helix-editor.com/>
- Lollypop Design — Trading App Design Guide 2026: <https://lollypop.design/blog/2026/june/trading-app-design/>

**Alpaca community UI references**
- Streamlit Alpaca dashboard (canonical pattern): <https://levelup.gitconnected.com/a-streamlit-dashboard-for-the-alpaca-api-algo-trading-platform-9a7194aa7844>
- Alpaca WebSocket streaming: <https://docs.alpaca.markets/us/docs/websocket-streaming>
- Alpaca-py `TradeStream` / `StockDataStream` for fills + market data
- Alpaca MCP server (AI-driven control plane): <https://github.com/alpacahq/alpaca-mcp-server>
- Third-party Alpaca UIs on GitHub:
  - <https://github.com/CruddyShad0w/Alpaca-Trading-Analytics-Dashboard>
  - <https://github.com/huygiatrng/AlpacaTradingAgent>
  - <https://github.com/lacymorrow/openclaw-alpaca-trading-skill>
