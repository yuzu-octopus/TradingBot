"""Fetch crypto daily OHLCV data from yfinance (replaces Alpaca free tier).

yfinance provides 5+ years of daily crypto data for major pairs vs
Alpaca's free CryptoHistoricalDataClient which only returns ~9 months.
Ticker format: BTC/USD -> BTC-USD (yfinance uses hyphens).
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm


def _yf_symbol(alpaca_symbol: str) -> str:
    """Map Alpaca-style crypto pairs to yfinance format.

    Alpaca uses "BTC/USD"; yfinance uses "BTC-USD".
    """
    return alpaca_symbol.replace("/", "-")


def fetch_crypto_data(
    symbols: list[str],
    start: str,
    end: str,
    output_dir: str,
) -> dict[str, pd.DataFrame]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data: dict[str, pd.DataFrame] = {}
    cached = 0

    for symbol in tqdm(
        symbols, desc="Downloading crypto", unit="pair", file=sys.stderr
    ):
        safe_name = symbol.replace("/", "-")
        path = out / f"{safe_name}.csv"

        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            # Invalidate cached CSV if it doesn't cover the requested date
            # range — e.g. old Alpaca CSVs only go back to Sep 2024, but
            # yfinance can provide data back to 2020. Also re-download empty
            # CSVs left from failed previous fetches.
            if df.empty or pd.Timestamp(start) < df.index[0]:
                if not df.empty:
                    tqdm.write(f"  Cache too short for {symbol} — re-downloading")
                path.unlink()
            else:
                cached += 1
        if not path.exists():
            try:
                yf_sym = _yf_symbol(symbol)
                ticker = yf.Ticker(yf_sym)
                df = ticker.history(start=start, end=end, auto_adjust=False)
                if df.empty:
                    tqdm.write(f"  No data for {symbol}")
                    continue
                # yfinance returns a MultiIndex or DatetimeIndex — normalize
                # to a plain DatetimeIndex with tz-naive timestamps.
                if isinstance(df.index, pd.DatetimeIndex):
                    df.index = df.index.tz_localize(None)
                else:
                    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
                # Keep only the columns the feature pipeline expects.
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                df.to_csv(path)
            except Exception as e:
                tqdm.write(f"  Failed to fetch {symbol}: {e}")
                continue

        data[symbol] = df

    tqdm.write(f"  ({cached}/{len(symbols)} from cache)")
    return data
