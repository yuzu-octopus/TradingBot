import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from config import Config, get_device
from models.stock_model import StockTransformer


def create_model(
    config: Config, device: torch.device | None = None
) -> StockTransformer:
    if device is None:
        device = get_device()
    model = StockTransformer(
        n_stocks=config.n_stocks,
        n_features=config.n_features,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        rankglu_bottleneck=64,
        market_state_size=5,
    ).to(device)
    return model


def load_model(config: Config, device: torch.device | None = None) -> StockTransformer:
    if device is None:
        device = get_device()
    model = create_model(config, device)
    state = torch.load(config.model_save_path, weights_only=True, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def save_scaler(scaler: StandardScaler, path: str) -> None:
    Path(path).write_text(
        json.dumps({"mean": scaler.mean_.tolist(), "var": scaler.var_.tolist()})
    )


def load_scaler(path: str) -> StandardScaler:
    data = json.loads(Path(path).read_text())
    scaler = StandardScaler()
    scaler.mean_ = np.array(data["mean"])
    scaler.var_ = np.array(data["var"])
    scaler.scale_ = np.sqrt(scaler.var_)
    scaler.n_features_in_ = scaler.mean_.shape[0]
    scaler.n_samples_seen_ = 1
    return scaler


def scale_features(features: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    T, S, F = features.shape
    scaled = scaler.transform(features.reshape(-1, F))
    scaled = np.nan_to_num(scaled, nan=0.0)
    return scaled.reshape(T, S, F)


def load_threshold(config: Config) -> tuple[float, float]:
    path = Path(f"{config.features_path}/threshold.txt")
    if path.exists():
        parts = path.read_text().strip().split(",")
        if len(parts) > 1:
            return float(parts[0]), float(parts[1])
        return float(parts[0]), float(parts[0])
    return 0.5, 0.5
