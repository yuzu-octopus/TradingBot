# Plan 005: Redact API keys from colab template source packing

> Drift check: `git diff --stat 672167a..HEAD -- src/colab_gen.py config.py`

## Status

- Priority: P2
- Effort: S
- Risk: LOW
- Depends on: none
- Category: security
- Planned at: commit `672167a`, 2026-06-23

## Why this matters

The `--colab-template` flag packs all source files (including `config.py`) into a base64 payload embedded in the generated script. If a user has set `alpaca_api_key` or `alpaca_secret_key` in the Config dataclass defaults (currently empty strings, but common during development), those keys are embedded in the script that gets copied to the clipboard and potentially shared/pasted into Colab/Kaggle notebooks.

## Current state

```python
# colab_gen.py:48-52
for p in [
    "config.py",
    "main.py",
    ...
]:
    files[p] = Path(p).read_text()  # reads raw content including any API keys
```

```python
# config.py:97-98 (current defaults are empty — safe today)
alpaca_api_key: str = ""
alpaca_secret_key: str = ""
```

## Scope

**In scope**: `src/colab_gen.py` (add redaction logic), `src/utils.py` (optionally add a redaction helper if preferred)

**Out of scope**: Changing config.py defaults, any other file

## Steps

### Step 1: Add key redaction to `_build_zip` or `generate_colab_script`

Before packing, redact known secret fields from source file contents. The simplest approach: post-process the packed content strings to replace known secret patterns:

```python
# In colab_gen.py, after reading each file
import re

def _redact_secrets(content: str, filename: str) -> str:
    """Replace API key / secret values with REDACTED in the generated script."""
    # Pattern: alpaca_api_key: str = "any_value" → alpaca_api_key: str = "REDACTED"
    content = re.sub(
        r'(alpaca_api_key|alpaca_secret_key)\s*[=:]\s*["\'].*?["\']',
        r'\1 = "REDACTED"',
        content,
    )
    return content
```

Apply in the file-reading loop:

```python
files[p] = _redact_secrets(Path(p).read_text(), p)
```

**Verify**: 

```bash
uv run python main.py --colab-template --show-script 2>/dev/null | grep -A2 "alpaca_api_key" | head -3
# Should show "REDACTED" or the default empty string
```

Confirm no `pk_` patterns in the generated script:

```bash
uv run python main.py --colab-template --show-script 2>/dev/null | grep "pk_" || echo "No API keys leaked"
```

## Done criteria

- [ ] `uv run ruff check .` exits 0
- [ ] `uv run pytest -q` → 60 passed
- [ ] Generated colab script contains `REDACTED` instead of actual API key values
- [ ] `grep "pk_"` on generated script output returns no matches

## STOP conditions

- If the regex misses edge cases (e.g., env-var based keys like `os.environ.get("ALPACA_API_KEY")`), report it — those are not in `config.py` defaults so they're lower risk, but worth noting.

## Maintenance notes

If new secret fields are added to `config.py`, they must be added to `_redact_secrets`. Review this function when adding any new configuration that accepts credentials.
