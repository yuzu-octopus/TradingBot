"""Tests for training/threshold.py optimize_threshold."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from config import Config, set_n_stocks
from src.utils import create_model


def test_optimize_threshold_returns_pair():
    """optimize_threshold returns (buy_t, sell_t) both floats >= 0."""
    from training.threshold import optimize_threshold

    config = Config()
    config.features_per_window = 3
    config.n_windows = 2
    n_feat = config.n_features
    set_n_stocks(config, 5)
    config.tickers = [str(i) for i in range(5)]

    # Need a valid scaler on disk for optimize_threshold
    with tempfile.TemporaryDirectory() as tmpdir:
        config.features_path = tmpdir
        scaler = StandardScaler()
        scaler.fit(np.random.randn(100, n_feat).astype(np.float32))
        scaler_path = Path(tmpdir) / "scaler.json"
        scaler_path.write_text(
            json.dumps({"mean": scaler.mean_.tolist(), "var": scaler.var_.tolist()})
        )

        model = create_model(config, torch.device("cpu"))
        val_features = np.random.randn(50, 5, n_feat).astype(np.float32)
        val_targets = np.random.randn(50, 5).astype(np.float32)

        buy_t, sell_t = optimize_threshold(config, model, val_features, val_targets)
        assert isinstance(buy_t, float)
        assert isinstance(sell_t, float)
        assert buy_t >= 0
        assert sell_t >= 0
