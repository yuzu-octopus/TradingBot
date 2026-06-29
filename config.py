import os
from dataclasses import dataclass, field

import pandas as pd
import requests
import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def set_n_stocks(cfg: Config, n: int) -> None:
    """Set universe size without mutating cfg.tickers (placeholder indices)."""
    cfg.tickers = [str(i) for i in range(n)]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device(f"cuda:{local_rank}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "TradingBot/1.0 (research project)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    import io

    tables = pd.read_html(io.StringIO(resp.text))
    return sorted(tables[0]["Symbol"].tolist())


@dataclass
class Config:
    tickers: list[str] = field(default_factory=list)

    train_start: str = "2015-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2025-06-01"

    features_per_window: int = 30
    n_windows: int = 4

    @property
    def n_stocks(self) -> int:
        return len(self.tickers)

    @property
    def n_features(self) -> int:
        return self.features_per_window * self.n_windows

    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1

    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    max_epochs: int = 300
    early_stop_patience: int = 25
    label_max_return: float = 0.05

    wf_window_size: int = 3
    wf_step_size: int = 1
    wf_val_size: int = 1
    wf_test_size: int = 1

    model_save_path: str = "data/models/best.pt"
    features_path: str = "data/features"
    raw_data_path: str = "data/stocks"

    pretrain_epochs: int = 100
    pretrain_lr: float = 1e-4
    pretrain_mask_ratio: float = 0.2
    pretrain_top_n_days: int = 3
    pretrain_weights_path: str = "data/models/pretrain/best.pt"

    tickers_file: str = ""
    alpaca_api_key: str = ""
    asset_class: str = "stocks"
    crypto_pairs: str = "top10"
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True
    trade_interval_minutes: int = 15
    trade_max_position_pct: float = 0.02
    trade_buy_qty: int = 10
    trade_sell_qty: int = 10
    max_portfolio_pct: float = 0.5
    no_amp: bool = False


CRYPTO_TOP10 = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "DOGE/USD",
    "XRP/USD",
    "ADA/USD",
    "AVAX/USD",
    "LINK/USD",
    "DOT/USD",
    "MATIC/USD",
]

CRYPTO_ALL = [
    *CRYPTO_TOP10,
    "BCH/USD",
    "FIL/USD",
    "LTC/USD",
    "NEAR/USD",
    "SHIB/USD",
    "TRX/USD",
    "UNI/USD",
]

CRYPTO_PAIR_MAP = {"top10": CRYPTO_TOP10, "all17": CRYPTO_ALL}
