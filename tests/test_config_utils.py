"""Tests for config.py utility functions."""

from config import Config, set_n_stocks


def test_set_n_stocks():
    cfg = Config()
    set_n_stocks(cfg, 10)
    assert len(cfg.tickers) == 10
    assert cfg.tickers[0] == "0"
    assert cfg.tickers[9] == "9"


def test_config_defaults():
    cfg = Config()
    assert cfg.batch_size > 0
    assert cfg.max_epochs > 0
    assert cfg.learning_rate > 0
    assert cfg.n_features > 0


def test_config_asset_class_default():
    cfg = Config()
    assert cfg.asset_class == "stocks"
