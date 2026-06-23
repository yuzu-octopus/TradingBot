"""Tests for src/utils.py."""

from pathlib import Path

from config import Config
from models.stock_model import StockTransformer
from src.utils import create_model, load_threshold, unwrap_model


def test_create_model_returns_transformer() -> None:
    cfg = Config()
    cfg.tickers = ["AAPL", "MSFT"]
    model = create_model(cfg)
    inner = unwrap_model(model)
    assert isinstance(inner, StockTransformer)


def test_unwrap_model_plain_module() -> None:
    import torch

    inner = torch.nn.Linear(4, 4)
    assert unwrap_model(inner) is inner


def test_load_threshold_default_when_no_file(tmp_path: Path) -> None:
    cfg = Config()
    cfg.features_path = str(tmp_path)
    buy, sell = load_threshold(cfg)
    assert buy == 0.5
    assert sell == 0.5


def test_load_threshold_parses_file(tmp_path: Path) -> None:
    cfg = Config()
    cfg.features_path = str(tmp_path)
    (tmp_path / "threshold.txt").write_text("0.3,0.4")
    buy, sell = load_threshold(cfg)
    assert buy == 0.3
    assert sell == 0.4
