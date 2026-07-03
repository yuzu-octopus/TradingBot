# Plan 002: Fix docs — architecture claims, hyperparameters, CLI defaults

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step.
>
> **Drift check (run first)**: `git diff --stat 8473bcf..HEAD -- README.md AGENTS.md project.toml`
> If any in-scope file changed, compare excerpts against live code.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: docs
- **Planned at**: commit `8473bcf`, 2026-06-28

## Why this matters

The README, AGENTS.md, and project.toml all claim the model is "decoder-only with causal masking" — but the actual code at `models/stock_model.py:81` uses `nn.TransformerEncoder` with full bidirectional attention (no causal mask). The comment at line 102 explicitly says "Encoder-only with full bidirectional self-attention — every stock attends to every stock. No causal mask." The docs are wrong. Additionally, hyperparameters (d_model, layers, heads) and the `--loss` default are stale.

## Current state

`models/stock_model.py:72-81`:
```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
    dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
```

`models/stock_model.py:102-104` comment:
```python
# Encoder-only with full bidirectional self-attention — every stock
# attends to every stock. No causal mask, no permutation, no
# stock_embed (full attention captures cross-stock relationships
```

`config.py` default `--loss`:
```python
# main.py line where argparse sets default — check actual default
```

## Scope

**In scope:**
- `README.md` — update architecture, hyperparameters, and --loss default
- `AGENTS.md` — update architecture claims and --loss default
- `project.toml` — update architecture, hyperparameters, and model details sections

**Out of scope:**
- Code changes (the model architecture is correct; only docs are wrong)

## Steps

### Step 1: Fix README.md

Replace all occurrences of:
- "Decoder-only Transformer" → "Encoder-only Transformer (full bidirectional self-attention)"
- "causal masking" → "full bidirectional attention"
- "each stock can only attend to itself and preceding stocks" → "every stock attends to every other stock"
- "d_model: 128" → check `config.py` defaults and use actual values
- "3 layers, 4 heads" → check `config.py` defaults
- Update "How it works" section point 3
- Update terminology section "Decoder-Only" description
- Find `--loss` default in `main.py` argparse and fix if wrong

### Step 2: Fix AGENTS.md

Same architecture text updates. Also update the "Key architecture decisions" section.

### Step 3: Fix project.toml

Update the model architecture table and the How It Works steps. The project.toml generates the GitHub Pages site, so this is the public-facing documentation.

### Step 4: Full validation

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

Expected: all checks passed, 79 tests pass (docs-only changes shouldn't break anything).

## Done criteria

- [ ] README.md says "Encoder-only" everywhere, not "Decoder-only"
- [ ] Hyperparameter table matches config.py defaults
- [ ] --loss default matches main.py argparse
- [ ] project.toml architecture table is accurate
- [ ] No references to "causal masking" remain (the model uses full attention)
- [ ] `uv run pytest -q` exits 0

## STOP conditions

- If config.py defaults are unclear or changed frequently — use the current values from the Config dataclass
