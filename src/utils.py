import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from config import Config, get_device, is_distributed
from models.stock_model import StockTransformer


def wrap_ddp(module: nn.Module, device: torch.device) -> nn.Module:
    """Wrap in DistributedDataParallel if distributed; raise on non-CUDA DDP."""
    if not is_distributed():
        return module
    if device.type != "cuda":
        msg = f"DDP requires CUDA. Got device={device}."
        raise RuntimeError(msg)
    return DistributedDataParallel(module, device_ids=[device.index])


def create_model(config: Config, device: torch.device | None = None) -> nn.Module:
    if device is None:
        device = get_device()
    model: nn.Module = StockTransformer(
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
    if is_distributed():
        # PyTorch's DistributedDataParallel does NOT support MPS — fail-fast
        # with a clear message rather than crashing inside the constructor
        # with NCCL/Gloo complaints when someone runs `torchrun` on a Mac.
        if device.type != "cuda":
            msg = (
                f"DDP requires CUDA. Got device={device}. "
                "Run without torchrun on Apple Silicon, or use a CUDA host."
            )
            raise RuntimeError(msg)
        model = DistributedDataParallel(model, device_ids=[device.index])
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    """Recursively strip DistributedDataParallel wrappers, returning the inner module.

    Safe to call on plain nn.Modules (returns them as-is).
    """
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def load_model(
    config: Config, device: torch.device | None = None
) -> StockTransformer | DistributedDataParallel:
    if device is None:
        device = get_device()
    model = create_model(config, device)
    state = torch.load(config.model_save_path, weights_only=True, map_location=device)
    unwrap_model(model).load_state_dict(state)
    model.eval()
    return model


def save_scaler(scaler: StandardScaler, path: str) -> None:
    Path(path).write_text(
        json.dumps({"mean": scaler.mean_.tolist(), "var": scaler.var_.tolist()})
    )


def load_scaler(path: str) -> StandardScaler:
    data = json.loads(Path(path).read_text())
    scaler = StandardScaler()
    assert "mean" in data and "var" in data, f"Corrupted scaler at {path}: missing keys"
    assert len(data["mean"]) > 0, f"Corrupted scaler at {path}: empty mean"
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


def setup_logger(log_file: str = "data/trading_bot.log") -> None:
    """Configure root logger with a RotatingFileHandler (10 MB, 5 backups)."""
    p = Path(log_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        return
    handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def load_threshold(config: Config) -> tuple[float, float]:
    path = Path(f"{config.features_path}/threshold.txt")
    if path.exists():
        parts = path.read_text().strip().split(",")
        if len(parts) > 1:
            return float(parts[0]), float(parts[1])
        return float(parts[0]), float(parts[0])
    return 0.5, 0.5
