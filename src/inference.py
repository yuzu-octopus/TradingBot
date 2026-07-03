from datetime import UTC, date, datetime, timedelta

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


# ponytail: hardcoded NYSE holidays, update yearly
NYSE_HOLIDAYS: set[tuple[int, int]] = {
    (1, 1),  # New Year's Day
    (6, 19),  # Juneteenth
    (7, 4),  # Independence Day
    (12, 25),  # Christmas
}


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> int:
    """Return the day of the Nth weekday of a given month (weekday 0=Mon)."""
    first = datetime(year, month, 1, tzinfo=UTC)
    offset = (weekday - first.weekday()) % 7
    return 1 + offset + 7 * (n - 1)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> int:
    """Return the day of the last weekday of a given month."""
    last = (
        datetime(year, month + 1, 1, tzinfo=UTC)
        if month < 12
        else datetime(year + 1, 1, 1, tzinfo=UTC)
    )
    last -= timedelta(days=1)
    offset = (weekday - last.weekday()) % 7
    return last.day - offset


def _easter(year: int) -> date:
    """Compute Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _is_good_friday(d: date) -> bool:
    """Return True if d is Good Friday (2 days before Easter Sunday)."""
    easter = _easter(d.year)
    return d == easter - timedelta(days=2)


def _is_nyse_holiday(d: date) -> bool:
    """Return True if d is a NYSE holiday."""
    fixed = {
        (1, _nth_weekday_of_month(d.year, 1, 0, 3)),  # MLK Day: 3rd Mon of Jan
        (2, _nth_weekday_of_month(d.year, 2, 0, 3)),  # Presidents Day: 3rd Mon of Feb
        (5, _last_weekday_of_month(d.year, 5, 0)),  # Memorial Day: last Mon of May
        (9, _nth_weekday_of_month(d.year, 9, 0, 1)),  # Labor Day: 1st Mon of Sep
        (11, _nth_weekday_of_month(d.year, 11, 3, 4)),  # Thanksgiving: 4th Thu of Nov
    }
    return (d.month, d.day) in NYSE_HOLIDAYS | fixed or _is_good_friday(d)


def _last_business_day() -> str:
    _d = datetime.now(UTC).date() - timedelta(days=1)
    while _d.weekday() >= 5 or _is_nyse_holiday(_d):
        _d -= timedelta(days=1)
    return str(_d)


def run_inference(
    config: Config,
    buy_threshold: float = 0.5,
    sell_threshold: float = 0.5,
    model: torch.nn.Module | None = None,
) -> dict[str, dict]:
    # Crypto trades 24/7 — don't use NYSE business-day logic which freezes
    # the cache on Fridays. For stocks, roll back to the last business day.
    if config.asset_class == "crypto":
        _now = datetime.now(UTC)
        target = str((_now - timedelta(days=1)).date())
    else:
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

    if model is None:
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
