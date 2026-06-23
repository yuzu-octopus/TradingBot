from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from config import Config
from src.utils import load_model, load_scaler, scale_features


def optimize_threshold(
    config: Config,
    model: torch.nn.Module,
    val_features: np.ndarray,
    val_targets: np.ndarray,
) -> tuple[float, float]:
    device = next(model.parameters()).device
    val_t = torch.tensor(
        scale_features(
            val_features, load_scaler(f"{config.features_path}/scaler.json")
        ),
        dtype=torch.float32,
    )

    model.eval()
    with torch.no_grad():
        scores = model(val_t.to(device)).cpu().numpy()

    flat_scores = scores.flatten()
    flat_targets = val_targets.flatten()
    mask = np.abs(flat_targets) > 1e-6
    cal_scores = flat_scores.copy()
    if mask.sum() > 10:
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
        lr.fit(flat_scores[mask].reshape(-1, 1), (flat_targets[mask] > 0).astype(int))
        cal_probs = lr.predict_proba(flat_scores.reshape(-1, 1))[:, 1]
        cal_scores = 2.0 * cal_probs - 1.0
    cal_scores = cal_scores.reshape(scores.shape)

    candidates = np.arange(0, 0.5, 0.01)
    best_buy = best_sell = 0.0
    best_sharpe = -float("inf")

    for buy_t in candidates:
        for sell_t in candidates:
            signals = np.where(
                cal_scores > buy_t, 1, np.where(cal_scores < -sell_t, -1, 0)
            )
            daily_ret = val_targets.mean(axis=1)
            port_ret = signals.mean(axis=1) * daily_ret
            sharpe = np.mean(port_ret) / (np.std(port_ret) + 1e-8) * np.sqrt(252)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_buy = buy_t
                best_sell = sell_t

    print(
        f"Best thresholds: buy > {best_buy:.2f}, sell < -{best_sell:.2f}, Sharpe={best_sharpe:.4f}"
    )
    return float(best_buy), float(best_sell)


def run_threshold_optimization(config: Config) -> tuple[float, float]:
    with np.load(Path(f"{config.features_path}/val.npz")) as data:
        val_features = data["features"]
        val_targets = data["targets"]

    model = load_model(config)
    buy_t, sell_t = optimize_threshold(config, model, val_features, val_targets)

    Path(f"{config.features_path}/threshold.txt").write_text(f"{buy_t},{sell_t}")
    return buy_t, sell_t
