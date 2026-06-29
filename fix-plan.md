# Fix Plan — All Items Resolved

**Last updated:** 2026-06-28 | HEAD: `8bcf234`

All items from the codebase audit have been fixed and committed. See `git log --oneline -20` for the full list of changes.

## Summary of fixes

| Item | Severity | Fix | Commit |
|------|----------|-----|--------|
| #1 Colab ImportError (trading deps) | CRITICAL | Lazy-import PaperTrader/trade in main.py; setup_logger → src/utils | `3ba25b7` |
| #2 Walk-forward threshold crash | CRITICAL | Fold model promotion to best.pt after loop | `3ba25b7` |
| #3 Pretrain chaining broken | CRITICAL | --mode pretrain then --mode train --pretrain | `0978ab2` |
| #4 Causal mask stock-order bias | HIGH | Random stock permutation per forward pass | `3ba25b7` |
| #5 No val Sharpe monitoring | MEDIUM | Compute val Sharpe every 10 epochs | `3ba25b7` |
| #6 Feature cache stale on code change | MEDIUM | Source-code hash in cache key | `3ba25b7` |
| #7 Duplicate trading loop | MEDIUM | Extracted `trade.run_trading_loop()`, both entry points call it | *(this PR)* |
| #8 pyperclip unused on Colab | MEDIUM | Removed from pip installs | `dae8232` |
| #9 Vestigial Console line | LOW | Removed dead Console(theme=...) + import | `3ba25b7` |
| N1 Auto-deps from pyproject.toml | NEW | tomllib-based pip string generation | *(this PR)* |
| N2 Dropped --asset-class --crypto-pairs | NEW | Forwarded to generated sys.argv | `8bcf234` |
| N3 Progress bars in Colab script | NEW | tqdm extraction + _phase() banners | *(this PR)* |
| Config forward reference crash | CRITICAL | from __future__ import annotations | `f4c0193` |
| flaglist NameError in pretrain block | CRITICAL | sv_orig = list(sys.argv) instead | `0978ab2` |
| Missing alpaca-py in Colab install | CRITICAL | Added to pip installs | `dae8232` |

## Architecture items (deferred)

| ID | Improvement |
|----|-------------|
| A2 | listnet_loss temperature (0.1) too aggressive |
| A3 | Survivorship bias in S&P 500 universe |
| A4 | No data augmentation |
| A5 | Walk-forward fold threshold ensemble |
| A6 | Pretrain drop_last=True on all loaders |
| B4 | margin_ranking_loss O(n²) memory |
