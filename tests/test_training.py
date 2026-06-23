import numpy as np
import torch

from config import Config
from models.stock_model import StockTransformer
from training.threshold import optimize_threshold
from training.train import listnet_loss, margin_ranking_loss, portfolio_mse_loss


def test_portfolio_mse_loss_shape() -> None:
    pred = torch.randn(4, 10)
    target = torch.randn(4, 10)
    loss = portfolio_mse_loss(pred, target)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_portfolio_mse_loss_perfect_prediction() -> None:
    pred = torch.tensor([[1.0, -1.0]])
    target = torch.tensor([[0.05, -0.03]])
    weights = pred / pred.sum(dim=1, keepdim=True)
    loss = portfolio_mse_loss(weights, target)
    assert loss.item() >= 0


def test_portfolio_mse_loss_zero() -> None:
    pred = torch.randn(4, 10)
    target = torch.zeros(4, 10)
    loss = portfolio_mse_loss(pred, target)
    assert loss.item() == 1.0


def test_margin_ranking_loss() -> None:
    pred = torch.randn(4, 10)
    target = torch.randn(4, 10)
    loss = margin_ranking_loss(pred, target)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_listnet_loss() -> None:
    pred = torch.randn(4, 10)
    target = torch.randn(4, 10)
    loss = listnet_loss(pred, target)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_optimize_threshold_runs() -> None:
    import json
    from pathlib import Path

    # Create a dummy scaler for optimize_threshold to load
    scaler_path = Path("data/features/scaler.json")
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_mean = np.zeros(120).tolist()
    dummy_var = np.ones(120).tolist()
    scaler_path.write_text(json.dumps({"mean": dummy_mean, "var": dummy_var}))

    model = StockTransformer(
        n_stocks=5, n_features=120, d_model=32, nhead=2, num_layers=1
    )
    val_features = np.random.randn(20, 5, 120)
    val_targets = np.random.randn(20, 5)
    buy_t, sell_t = optimize_threshold(Config(), model, val_features, val_targets)
    assert 0 <= buy_t < 0.5
    assert 0 <= sell_t < 0.5


def test_run_training_imports() -> None:
    from training.train import (
        load_checkpoint,
        run_training,
        save_checkpoint,
        train,
        train_seed,
    )

    assert callable(train)
    assert callable(train_seed)
    assert callable(run_training)
    assert callable(save_checkpoint)
    assert callable(load_checkpoint)
