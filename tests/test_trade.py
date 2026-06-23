"""Tests for trade.py — display functions only (no trading)."""

from rich.table import Table

from trade import build_layout, make_trade_table


def test_make_trade_table_returns_table() -> None:
    signals = {"AAPL": {"score": 0.8, "signal": "BUY"}}
    positions = {}
    trades = [("AAPL", 10, "BUY")]
    account = {"equity": 100000, "cash": 50000, "day_change": 100}
    table = make_trade_table(
        signals,
        positions,
        trades,
        account,
        cycle=1,
        interval=900,
        now_str="2026-01-01 10:00:00 ET",
    )
    assert isinstance(table, Table)


def test_build_layout_returns_layout() -> None:
    t = Table()
    layout = build_layout(t)
    assert layout is not None
