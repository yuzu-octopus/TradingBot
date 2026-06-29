# Model Performance Fix Plan — June 29, 2026

## Problem

The StockTransformer model produces near-constant output scores across all stocks and dates:

| Model | Score Mean | Score Std | Test Sharpe | $100K result |
|-------|-----------|-----------|-------------|-------------|
| Colab MSRR (5-seed) | -0.047 | 0.015 | 0.87 | $100,637 (+0.6%) |
| Local MSE (1-seed) | -0.993 | 0.004 | N/A | Model broken — tanh saturated at -1 |

The model has learned to output essentially the same score for every stock, every day. It cannot discriminate between winners and losers, rendering it useless for trading.

## Root Causes

Three compounding design flaws cause this:

### 1. `tanh` output saturation (CRITICAL)
- `torch.tanh(x)` in `StockTransformer.forward()` squashes outputs to [-1, 1]
- Without careful init and loss design, gradients vanish as tanh saturates
- The local model's pre-tanh activation drifted permanently negative → tanh locked at -0.99
- Even the "better" Colab model collapsed to a tight cluster at -0.05 (bias, not signal)
- **Fix**: Replace `torch.tanh(x)` with `torch.clamp(x, -1, 1)`. Clamp bounds scores without causing gradient vanishing — it only kills gradients when the value is exactly outside [-1, 1], not gradually as it approaches the boundary. This prevents the score explosion that pure margin loss would cause (adversarial review finding), while allowing the model to produce full-range differentiated signals.

### 2. Decoder-only causal mask prevents cross-stock learning (CRITICAL)
- `TransformerDecoder` with `tgt_mask=causal_mask` limits each stock to attending only to preceding stocks (alphabetical order)
- Stock "A" sees nothing but itself; stock "ZTS" sees 502 others
- This systematic asymmetry forces the model to average information across positions → uniform output
- Research consensus: encoder-only (full bidirectional attention) is strictly better for cross-sectional prediction since stocks have no natural ordering
- **Fix**: Replace `nn.TransformerDecoder` with `nn.TransformerEncoder`. Remove causal mask, random permutation, and unshuffle logic. Drop `stock_embed` entirely — encoder full attention makes positional identity embeddings redundant, and sequential stock IDs tied to alphabetical order are brittle across ticker list changes (adversarial review finding).

### 3. MSRR/MSE loss has no incentive to discriminate (CRITICAL)
- `MSRR = (1 - w'R)^2`: if w is constant, w'R ≈ 0 (R is z-scored) → loss ≈ 1.0. No gradient to diversify w.
- `MSE = (w - R)^2`: if R ≈ 0, predicting 0 minimizes MSE. With tanh, this forces pre-tanh → 0, but the model found a local minimum at tanh → -1 instead.
- **Fix**: Switch default loss to `margin` (pairwise ranking). Margin loss explicitly penalizes incorrect relative ordering of stock returns and cannot collapse to constant — constant output produces 50% wrong orderings.

### 4. RankGLU gamma suppresses nonlinear branch (HIGH)
- `self.gamma = nn.Parameter(torch.ones(1) * 0.5)` initializes the nonlinear contribution at 50%
- Combined with tanh, the linear shortcut dominates and the nonlinear branch never activates
- **Fix**: Initialize gamma at 1.0 so both branches contribute equally at start

### 5. No learning rate warmup (MEDIUM)
- CosineAnnealingLR starts at full LR without warmup
- Transformer training is unstable early epochs without warmup
- **Fix**: Add linear warmup for first 5% of steps via `SequentialLR` (LinearLR → CosineAnnealingLR)

### 6. Xavier init wrong gain for tanh (MEDIUM)
- `nn.init.xavier_uniform_(m.weight)` uses default gain=1.0 (for linear/sigmoid)
- tanh requires gain ≈ 1.667 to maintain proper variance
- **Fix**: Remove tanh (P0 fix #1) makes this moot, but if tanh remains anywhere, use `gain=nn.init.calculate_gain('tanh')`

## Implementation Plan

### P0 — Fixes (guaranteed to improve discrimination)

#### Task 1: Replace tanh with clamp(-1, 1)
**File**: `models/stock_model.py`
**Change**: Replace `x = torch.tanh(x)` with `x = torch.clamp(x, -1, 1)`.
**Why clamp not identity**: Margin ranking loss without bounded output causes score explosion (scores → ±∞ trivially satisfy margin). Clamp bounds scores while preserving gradients in the active range, unlike tanh which kills gradients gradually.
**Expected**: Score std increases from 0.015 → >0.05. Scores stay in [-1, 1] so threshold optimization and UI rendering still work.

#### Task 2: Switch to encoder-only self-attention
**File**: `models/stock_model.py`
**Changes**:
- Replace `nn.TransformerDecoder` with `nn.TransformerEncoder`
- Remove `nn.Transformer.generate_square_subsequent_mask` and `tgt_mask`/`tgt_is_causal`
- Remove random `perm` permutation logic (encoder attends bidirectionally — ordering doesn't matter)
- Remove `perm.argsort()` unshuffle
- Remove `stock_embed` entirely OR apply it to stocks in their natural order (no shuffling)
- `memory=` parameter goes away (encoder doesn't need memory/cross-attention)
**Forward pass becomes**:
```python
def forward(self, x, market_state=None):
    if market_state is not None and self.market_state_size > 0:
        x = self.market_gate(x, market_state)
    x = self.input_proj(x)
    x = self.dropout(x)
    x = self.transformer(x)  # encoder: full bidirectional attention
    x = self.norm(x)
    x = self.output_head(x)  # RankGLU
    x = torch.clamp(x, -1, 1)  # bounded, no gradient vanishing
    return x.squeeze(-1)
```
- **Removed**: `perm`, `perm.argsort()`, `stock_embed`, `causal_mask`, `memory=x`, `tgt_mask`, `tgt_is_causal`
- **Removed**: `self.stock_embed` — full attention makes stock ID embeddings redundant; sequential IDs are brittle
- `__init__` removes `self.stock_embed = nn.Embedding(...)` line

**Compatibility note**: Old checkpoints will NOT load after this change. TransformerDecoder state_dict keys differ from TransformerEncoder (e.g. `multihead_attn` vs `self_attn`). This is documented breaking change. Users should retrain.

#### Task 3: Switch default loss to margin ranking
**File**: `main.py`
**Change**: Change default loss in argparse from `"mse"` to `"margin"`.
**No code changes** needed in `train.py` — margin loss already implemented.
**Expected**: Loss explicitly penalizes incorrect ordering. Cannot collapse to constant output.

#### Task 4: Initialize RankGLU gamma at 1.0
**File**: `models/stock_model.py`
**Change**: `self.gamma = nn.Parameter(torch.ones(1))` (remove `* 0.5`)
**Expected**: Nonlinear GLU branch contributes equally from epoch 1, not suppressed.

### P1 — Training improvements

#### Task 5: Add learning rate warmup
**File**: `training/train.py`
**Changes**:
- After creating `CosineAnnealingLR`, wrap with `LinearLR` warmup
- First 5% of max_epochs: LR ramps from 0 to peak
- Then cosine decay to 0
```python
warmup_epochs = max(1, int(config.max_epochs * 0.05))
warmup = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, total_iters=warmup_epochs
)
cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs - warmup_epochs)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
)
```
**Expected**: Stable early training, better convergence.

#### Task 6: Double model capacity
**File**: `config.py`
**Changes**:
- `d_model`: 128 → 256
- `num_layers`: 3 → 4
- `nhead`: 4 → 8
- `dim_feedforward`: 256 → 512
- Params: ~478K → ~1.9M (still small enough for Colab GPU)
**Risk**: ~4x compute increase. Mitigated by encoder (no causal mask = shorter forward pass).
**Expected**: More capacity to learn inter-stock relationships.

### P2 — Future (not in this sprint)

- Target engineering: volatility-scaled returns or quintile classification
- Sector embeddings (GICS codes) as additional input
- Feature improvements: SPY-relative strength, earnings surprise data
- Walk-forward with ensemble averaging

## Validation Plan

After all changes:
1. `uv run ruff check .` + `uv run ruff format .` — zero errors
2. `uv run mypy .` — baseline 36 errors (no new)
3. `uv run pytest` — 79/79 pass
4. `uv run python main.py --mode train --loss margin --seeds 1` — train a model
5. Backtest on test set — target Sharpe > 1.0 (vs current 0.87), score std > 0.05 (vs current 0.015), no tanh saturation

## Files Changed

| File | Task | Lines |
|------|------|-------|
| `models/stock_model.py` | clamp(-1,1), switch to encoder, drop stock_embed, fix gamma | ~20 changed, ~20 removed |
| `config.py` | Double model capacity (d_model 256, layers 4, heads 8, ff 512) | 4 lines |
| `training/train.py` | LR warmup (SequentialLR), dynamic threshold range for Sharpe scan | ~10 added, 1 changed |
| `main.py` | Default loss margin | 1 line |

## Revert Plan

If performance degrades:
- `git revert` the commit
- All changes are to 4 files, self-contained
- Feature caches (`data/features/*.npz`) are unaffected
- **Old model checkpoints NOT compatible** — architecture change (decoder→encoder) changes state_dict keys. Revert and retrain if needed.
