"""Tests for src/inference.py."""

from datetime import date

from src.inference import _last_business_day


def test_last_business_day_returns_weekday() -> None:
    d = date.fromisoformat(_last_business_day())
    assert d.weekday() < 5


def test_last_business_day_format() -> None:
    parts = _last_business_day().split("-")
    assert len(parts) == 3
    assert len(parts[0]) == 4
