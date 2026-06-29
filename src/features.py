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
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


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


def build_targets(
    raw_data: dict[str, pd.DataFrame],
    tickers: list[str],
    dates: list[str],
    max_return: float,
    clip_extreme: bool = True,
    clip_per_ticker: bool = True,
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
    if clip_extreme:
        if clip_per_ticker:
            for j in range(n_stocks):
                col = targets[:, j]
                valid = col[~np.isnan(col)]
                if len(valid) == 0:
                    continue
                lower = np.quantile(valid, LABEL_CLIP_PCT)
                upper = np.quantile(valid, 1 - LABEL_CLIP_PCT)
                targets[:, j] = np.clip(col, lower, upper)
        else:
            flat = targets.flatten()
            valid = flat[~np.isnan(flat)]
            lower = np.quantile(valid, LABEL_CLIP_PCT)
            upper = np.quantile(valid, 1 - LABEL_CLIP_PCT)
            targets = np.clip(targets, lower, upper)
    targets[np.isnan(targets)] = 0.0
    return targets


def compute_market_state(
    raw_data: dict[str, pd.DataFrame],
    dates: list[str],
    market_ticker: str = "SPY",
) -> np.ndarray:
    spy = raw_data.get(market_ticker)
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
        stock_vec: list[float] = []
        for feat_series in [r1y, r1m, r1w, r1d]:
            stock_vec.extend(
                float(feat_series[col]) if not pd.isna(feat_series[col]) else 0.0
                for col in cols
            )
        feature_matrix[col_idx] = np.array(stock_vec)
    return np.nan_to_num(feature_matrix, nan=0.0), tickers


def _data_hash(raw_data_dir: str) -> str:
    import zlib

    parts = []
    for p in sorted(Path(raw_data_dir).glob("*.csv")):
        content = p.read_bytes()
        crc = f"{zlib.crc32(content[:4096]):08x}"
        parts.append(f"{p.name}|m={p.stat().st_mtime}|s={p.stat().st_size}|c={crc}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _features_code_hash() -> str:
    """Hash the feature computation source so cache invalidates on code changes."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def _cache_key(raw_data_dir: str) -> str:
    """Composite cache key: data fingerprint + feature-code fingerprint."""
    return f"{_data_hash(raw_data_dir)}:{_features_code_hash()}"


def load_cached_features(
    raw_data_dir: str,
    cache_dir: str = "data/features",
) -> tuple[np.ndarray, list[str], list[str]] | None:
    mat_path = f"{cache_dir}/matrix.npz"
    hash_path = f"{cache_dir}/matrix_hash.txt"
    if not Path(mat_path).exists() or not Path(hash_path).exists():
        return None
    cached_hash = Path(hash_path).read_text().strip()
    if cached_hash != _cache_key(raw_data_dir):
        return None
    data = np.load(mat_path)
    return data["features"], data["tickers"].tolist(), data["dates"].tolist()


def save_cached_features(
    features: np.ndarray,
    tickers: list[str],
    dates: list[str],
    raw_data_dir: str,
    cache_dir: str = "data/features",
) -> None:
    mat_path = f"{cache_dir}/matrix.npz"
    hash_path = f"{cache_dir}/matrix_hash.txt"
    Path(mat_path).parent.mkdir(parents=True, exist_ok=True)
    Path(hash_path).write_text(_cache_key(raw_data_dir))
    np.savez_compressed(mat_path, features=features, tickers=tickers, dates=dates)
    print(f"  Cached feature matrix to {mat_path}")
