import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import unlockedpd
from tqdm import tqdm

WINDOW_1Y = 252
WINDOW_1M = 21
WINDOW_1W = 5
WINDOW_1D = 1
N_FEATURES = 30
N_WINDOWS = 4
LABEL_CLIP_PCT = 0.05


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series) -> pd.DataFrame:
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9).mean()
    hist = macd_line - signal
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal, "macd_hist": hist})


def compute_bollinger(series: pd.Series, period: int = 20) -> pd.DataFrame:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    bb_pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"bb_upper": upper, "bb_lower": lower, "bb_pct_b": bb_pct_b})


def compute_window_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    daily_return = close.pct_change()
    result = pd.DataFrame(index=df.index)

    for period in [5, 10, 20, 50, 200]:
        result[f"sma_{period}"] = close.rolling(period).mean()
        result[f"price_sma_{period}"] = close / result[f"sma_{period}"].replace(
            0, np.nan
        )

    for period in [5, 14]:
        result[f"rsi_{period}"] = compute_rsi(close, period)

    macd_df = compute_macd(close)
    for col in macd_df.columns:
        result[col] = macd_df[col]

    bb_df = compute_bollinger(close)
    for col in bb_df.columns:
        result[col] = bb_df[col]

    for period in [5, 21, 63, 252]:
        result[f"volatility_{period}"] = daily_return.rolling(period).std() * np.sqrt(
            period
        )

    result["volume_ratio"] = volume / volume.rolling(252).mean().replace(0, np.nan)
    result["intraday_range"] = (high - low) / close.replace(0, np.nan)

    for period in [1, 5, 21, 63, 252]:
        result[f"return_{period}d"] = close.pct_change(period)

    rolling_max = close.rolling(252).max()
    result["max_drawdown"] = (close - rolling_max) / rolling_max.replace(0, np.nan)

    return result


def normalize_targets_cross_sectional(
    targets: np.ndarray, winsorize_pct: float = 0.025
) -> np.ndarray:
    T, _S = targets.shape
    normalized = np.full_like(targets, 0.0)
    for i in range(T):
        day = targets[i]
        lower = np.quantile(day, winsorize_pct)
        upper = np.quantile(day, 1 - winsorize_pct)
        day = np.clip(day, lower, upper)
        mean = np.nanmean(day)
        std = np.nanstd(day)
        if std > 1e-8:
            normalized[i] = (day - mean) / std
    return np.nan_to_num(normalized, nan=0.0)


def build_targets(
    raw_data: dict[str, pd.DataFrame],
    tickers: list[str],
    dates: list[str],
    max_return: float,
    clip_extreme: bool = True,
    cross_sectional_norm: bool = False,
) -> np.ndarray:
    print("Computing training targets (next-day returns)...")
    n_dates, n_stocks = len(dates), len(tickers)
    targets = np.full((n_dates, n_stocks), np.nan)
    for i, date_str in enumerate(dates):
        date = pd.Timestamp(date_str)
        for j, ticker in enumerate(tickers):
            df = raw_data.get(ticker)
            if df is None or date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if idx + 1 >= len(df):
                continue
            next_close = df.iloc[idx + 1]["Close"]
            cur_close = df.loc[date, "Close"]
            if cur_close == 0:
                continue
            ret = (next_close - cur_close) / cur_close
            targets[i, j] = np.clip(ret / max_return, -1, 1)
    targets[np.isnan(targets)] = 0.0
    if clip_extreme:
        flat = targets.flatten()
        lower = np.quantile(flat, LABEL_CLIP_PCT)
        upper = np.quantile(flat, 1 - LABEL_CLIP_PCT)
        targets = np.clip(targets, lower, upper)
    if cross_sectional_norm:
        targets = normalize_targets_cross_sectional(targets)
    return targets


def compute_market_state(
    raw_data: dict[str, pd.DataFrame], dates: list[str]
) -> np.ndarray:
    spy = raw_data.get("SPY")
    if spy is None:
        return np.zeros((len(dates), 5))
    close = spy["Close"]
    volume = spy["Volume"]
    n = len(dates)
    state = np.zeros((n, 5))
    for i, date_str in enumerate(dates):
        date = pd.Timestamp(date_str)
        if date not in close.index:
            continue
        idx = close.index.get_loc(date)
        state[i, 0] = (
            close.iloc[idx] / close.iloc[max(0, idx - 1)] - 1 if idx >= 1 else 0
        )
        state[i, 1] = (
            close.iloc[idx] / close.iloc[max(0, idx - 5)] - 1 if idx >= 5 else 0
        )
        state[i, 2] = (
            close.iloc[idx] / close.iloc[max(0, idx - 21)] - 1 if idx >= 21 else 0
        )
        rets = close.iloc[max(0, idx - 21) : idx + 1].pct_change().dropna()
        state[i, 3] = float(rets.std()) if len(rets) > 1 else 0
        state[i, 4] = (
            volume.iloc[idx] / volume.iloc[max(0, idx - 21) : idx + 1].mean()
            if idx >= 21
            else 1
        )
    return np.nan_to_num(state, nan=0.0)


def build_feature_matrix(
    raw_data: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, list[str], list[str]]:
    all_features = {}
    for ticker, df in raw_data.items():
        if "Close" not in df.columns or len(df) < WINDOW_1Y:
            print(
                f"  Skipping {ticker}: insufficient data ({len(df) if 'Close' in df.columns else 0} rows)"
            )
            continue
        all_features[ticker] = compute_window_features(df)

    if not all_features:
        msg = "No stocks have enough data (need at least 252 days per stock)"
        raise ValueError(msg)

    valid_dates = [set(df.dropna().index) for df in all_features.values()]
    min_required = int(len(valid_dates) * 0.8)
    date_counts: dict = {}
    for s in valid_dates:
        for d in s:
            date_counts[d] = date_counts.get(d, 0) + 1

    common_idx = sorted(d for d, count in date_counts.items() if count >= min_required)
    dates = [str(d) for d in common_idx]
    tickers = list(all_features.keys())

    rolling_1y = {}
    rolling_1m = {}
    rolling_1w = {}
    for ticker, df in all_features.items():
        rolling_1y[ticker] = df.rolling(WINDOW_1Y).mean()
        rolling_1m[ticker] = df.rolling(WINDOW_1M).mean()
        rolling_1w[ticker] = df.rolling(WINDOW_1W).mean()

    feature_matrix = np.full((len(dates), len(tickers), N_FEATURES * N_WINDOWS), np.nan)

    def _extract_date(date_idx_pair):
        row_idx, date = date_idx_pair
        row = np.zeros((len(tickers), N_FEATURES * N_WINDOWS))
        for col_idx, ticker in enumerate(tickers):
            try:
                r1y = rolling_1y[ticker].loc[date]
                r1m = rolling_1m[ticker].loc[date]
                r1w = rolling_1w[ticker].loc[date]
                r1d = all_features[ticker].loc[date]
            except KeyError:
                continue
            cols = all_features[ticker].columns[:N_FEATURES]
            stock_vec = []
            for feat_series in [r1y, r1m, r1w, r1d]:
                stock_vec.extend(
                    float(feat_series[col]) if not pd.isna(feat_series[col]) else 0.0
                    for col in cols
                )
            row[col_idx] = np.array(stock_vec)
        return row_idx, row

    chunks = list(enumerate(common_idx))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_extract_date, c) for c in chunks]
        for f in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Building features",
            unit="date",
            file=sys.stderr,
        ):
            row_idx, row = f.result()
            feature_matrix[row_idx] = row

    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
    return feature_matrix, tickers, dates


def compute_features_for_date(
    raw_data: dict[str, pd.DataFrame],
    date_str: str,
) -> tuple[np.ndarray, list[str]]:
    date = pd.Timestamp(date_str)
    tickers = []
    features = {}
    for ticker, df in raw_data.items():
        if "Close" in df.columns and len(df) >= WINDOW_1Y:
            tickers.append(ticker)
            features[ticker] = compute_window_features(df)
    if not tickers:
        msg = "No stocks with sufficient data"
        raise ValueError(msg)
    feature_matrix = np.full((len(tickers), N_FEATURES * N_WINDOWS), 0.0)
    for col_idx, ticker in enumerate(tickers):
        try:
            feat = features[ticker]
            r1y = feat.rolling(WINDOW_1Y).mean().loc[date]
            r1m = feat.rolling(WINDOW_1M).mean().loc[date]
            r1w = feat.rolling(WINDOW_1W).mean().loc[date]
            r1d = feat.loc[date]
        except KeyError:
            continue
        except TypeError:
            continue
        cols = feat.columns[:N_FEATURES]
        stock_vec = []
        for feat_series in [r1y, r1m, r1w, r1d]:
            stock_vec.extend(
                float(feat_series[col]) if not pd.isna(feat_series[col]) else 0.0
                for col in cols
            )
        feature_matrix[col_idx] = np.array(stock_vec)
    return np.nan_to_num(feature_matrix, nan=0.0), tickers


FEATURE_CACHE_PATH = "data/features/matrix.npz"
HASH_CACHE_PATH = "data/features/matrix_hash.txt"


def _data_hash(raw_data_dir: str) -> str:
    hasher = hashlib.sha256()
    for f in sorted(Path(raw_data_dir).iterdir()):
        if f.suffix == ".csv":
            stat = f.stat()
            hasher.update(f"{f.name}:{stat.st_mtime}:{stat.st_size}".encode())
    return hasher.hexdigest()[:16]


def load_cached_features(
    raw_data_dir: str,
) -> tuple[np.ndarray, list[str], list[str]] | None:
    if not Path(FEATURE_CACHE_PATH).exists() or not Path(HASH_CACHE_PATH).exists():
        return None
    cached_hash = Path(HASH_CACHE_PATH).read_text().strip()
    if cached_hash != _data_hash(raw_data_dir):
        return None
    data = np.load(FEATURE_CACHE_PATH)
    return data["features"], data["tickers"].tolist(), data["dates"].tolist()


def save_cached_features(
    features: np.ndarray, tickers: list[str], dates: list[str], raw_data_dir: str
) -> None:
    Path(FEATURE_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(HASH_CACHE_PATH).write_text(_data_hash(raw_data_dir))
    np.savez_compressed(
        FEATURE_CACHE_PATH, features=features, tickers=tickers, dates=dates
    )
    print(f"  Cached feature matrix to {FEATURE_CACHE_PATH}")
