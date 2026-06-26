from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import torch

from config import Config
from src.crypto_pipeline import fetch_crypto_data
from src.data_pipeline import fetch_stock_data
from src.features import compute_features_for_date, compute_market_state
from src.utils import load_model, load_scaler

# Cache keyed by (tickers, asof_date) so a new trading day forces a fresh fetch.
# Without date-keying, a Monday 9:30 AM cycle could cache OHLCV; a Tuesday cycle
# would silently return Monday's data. Same-day cycles still share the cache.
_raw_data_cache: dict[tuple, dict[str, pd.DataFrame]] = {}


def invalidate_inference_cache() -> None:
    """Drop the raw-data cache. Call when the asof date has changed externally."""
    _raw_data_cache.clear()


def _last_business_day() -> str:
    d = datetime.now(UTC).date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return str(d)


def run_inference(
    config: Config, buy_threshold: float = 0.5, sell_threshold: float = 0.5
) -> dict[str, dict]:
    target = _last_business_day()
    cache_key = (tuple(config.tickers), target)
    if cache_key not in _raw_data_cache:
        if len(_raw_data_cache) > 1:
            _raw_data_cache.clear()
        _raw_data_cache[cache_key] = (
            fetch_crypto_data(
                config.tickers,
                config.train_start,
                config.test_end,
                config.raw_data_path,
            )
            if config.asset_class == "crypto"
            else fetch_stock_data(
                config.tickers,
                config.train_start,
                config.test_end,
                config.raw_data_path,
            )
        )
    raw_data = _raw_data_cache[cache_key]
    all_dates = sorted(raw_data[next(iter(raw_data))].index)
    all_date_strs = {str(d.date()) for d in all_dates}
    latest_date = target if target in all_date_strs else str(all_dates[-1].date())
    features, tickers = compute_features_for_date(raw_data, latest_date)
    features = features[np.newaxis, :, :]

    market_ticker = "BTC/USD" if config.asset_class == "crypto" else "SPY"
    market = compute_market_state(raw_data, [latest_date], market_ticker=market_ticker)

    scaler = load_scaler(f"{config.features_path}/scaler.json")
    scaled = scaler.transform(features.reshape(-1, config.n_features)).reshape(
        1, -1, config.n_features
    )

    model = load_model(config)
    device = next(model.parameters()).device
    with torch.no_grad():
        inp = torch.tensor(scaled, dtype=torch.float32).to(device)
        market_t = torch.tensor(market, dtype=torch.float32).to(device)
        scores = model(inp, market_state=market_t).cpu().numpy()[0]

    results = {}
    for i, ticker in enumerate(tickers):
        score = float(scores[i])
        if score > buy_threshold:
            signal = "BUY"
        elif score < -sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"
        results[ticker] = {"score": score, "signal": signal}
    return results
