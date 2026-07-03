"""Tests for src/inference.py utilities."""

from datetime import date

from src.inference import _is_nyse_holiday, _last_business_day


def test_is_nyse_holiday_new_years():
    assert _is_nyse_holiday(date(2026, 1, 1)) is True


def test_is_nyse_holiday_juneteenth():
    assert _is_nyse_holiday(date(2026, 6, 19)) is True


def test_is_nyse_holiday_not_weekend():
    assert _is_nyse_holiday(date(2026, 1, 3)) is False


def test_last_business_day_returns_string():
    result = _last_business_day()
    assert isinstance(result, str)
    assert len(result.split("-")) == 3
