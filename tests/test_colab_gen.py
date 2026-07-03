"""Tests for src/colab_gen.py secret redaction."""

from src.colab_gen import _redact_secrets


def test_redact_secrets_alpaca_key():
    content = 'alpaca_api_key = "pk_abc123def"'
    result = _redact_secrets(content)
    assert "pk_abc123def" not in result
    assert "REDACTED" in result


def test_redact_secrets_alpaca_secret():
    content = "alpaca_secret_key = 'xyz_secret_789'"
    result = _redact_secrets(content)
    assert "xyz_secret_789" not in result
    assert "REDACTED" in result


def test_redact_secrets_preserves_non_secrets():
    content = "TICKER=AAPL"
    result = _redact_secrets(content)
    assert "AAPL" in result
