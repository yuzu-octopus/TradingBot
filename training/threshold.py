from pathlib import Path

import numpy as np
import torch

from config import Config
from src.utils import load_model, load_scaler, scale_features


def optimize_threshold(
    config: Config,
    model: torch.nn.Module,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    market_state: np.ndarray | None = None,
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
        m_t = (
            torch.tensor(market_state, dtype=torch.float32).to(device)
            if market_state is not None
            else None
        )
        kwargs = {"market_state": m_t} if m_t is not None else {}
        scores = model(val_t.to(device), **kwargs).cpu().numpy()

    # Use raw model outputs directly. The previous algorithm fit a calibrator
    # (Isotonic -> Logistic / Platt) on the val set and then rescaled its output
    # back to [-1, 1] via `2*probs - 1`, which collapsed the dynamic range and
    # made historical thresholds pick up noise. Raw scores preserve the model's
    # intended confidence range, and Sharpe is scale-invariant for our signal-
    # weighting (mean of signals, not sum), so optimizing against raw scores
    # is comparable AND preserves interpretability.
    cal_scores = scores

    # Adapt the threshold scan to the actual score distribution. Raw model
    # outputs aren't bounded — with no Platt rescale, scores may span
    # [-k, +k] for some k > 1. Search at least up to the observed max-abs,
    # capped at 2.0 to keep the scan bounded on pathological score scales.
    max_abs = float(np.abs(scores).max())
    upper = max(0.5, min(max_abs, 2.0))
    if max_abs > 2.0:
        print(
            f"Note: raw scores hit {max_abs:.2f}; threshold scan capped at 2.0. "
            "Consider retraining or rescaling."
        )
    candidates = np.arange(0.0, upper + 0.05, 0.05)
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
        val_market = data.get("market_state")

    model = load_model(config)
    buy_t, sell_t = optimize_threshold(
        config, model, val_features, val_targets, market_state=val_market
    )

    tmp = Path(f"{config.features_path}/threshold.tmp")
    tmp.write_text(f"{buy_t},{sell_t}")
    tmp.rename(Path(f"{config.features_path}/threshold.txt"))
    return buy_t, sell_t
