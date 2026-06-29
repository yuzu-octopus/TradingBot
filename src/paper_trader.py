import logging
import math
import os
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TypeVar
from zoneinfo import ZoneInfo

from alpaca.data import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoLatestQuoteRequest, StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from config import Config

logger = logging.getLogger(__name__)

_F = TypeVar("_F")


def _retry[F](
    fn: Callable[..., _F], *args: object, max_tries: int = 3, **kwargs: object
) -> _F:
    for attempt in range(max_tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_tries - 1:
                raise
            wait = 2**attempt
            logger.warning("Retry %s/%d: %s", attempt + 1, max_tries, e)
            time.sleep(wait)
    return None


class PaperTrader:
    def __init__(self, config: Config) -> None:
        key = config.alpaca_api_key or os.environ.get("ALPACA_API_KEY", "")
        secret = config.alpaca_secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        if not key or not secret:
            msg = "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env"
            raise ValueError(msg)
        self.config = config
        self.trade_client = TradingClient(key, secret, paper=config.alpaca_paper)
        self._stock_client = StockHistoricalDataClient(key, secret)
        self._crypto_client = CryptoHistoricalDataClient()
        self.nyc = ZoneInfo("America/New_York")
        self._positions_cache: dict[str, dict] = {}
        self._account_cache: dict = {}
        self._audit_path = Path("data/paper_trades.csvl")

    def get_account(self) -> dict:
        acct = _retry(self.trade_client.get_account)
        self._account_cache = {
            "equity": float(acct.equity),  # type: ignore[attr-defined]
            "cash": float(acct.cash),  # type: ignore[attr-defined]
            "buying_power": float(acct.buying_power),  # type: ignore[attr-defined]
            "day_change": float(acct.equity) - float(acct.last_equity),  # type: ignore[attr-defined]
        }
        return self._account_cache

    def get_positions(self) -> dict[str, dict]:
        positions = _retry(self.trade_client.get_all_positions)
        self._positions_cache = {
            p.symbol: {
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "side": p.side,
            }
            for p in positions
        }
        return self._positions_cache

    def get_latest_quotes(self, symbols: list[str]) -> dict[str, dict]:
        if not symbols:
            return {}
        try:
            if self.config.asset_class == "crypto":
                req = CryptoLatestQuoteRequest(symbol_or_symbols=symbols)
                quotes = _retry(self._crypto_client.get_crypto_latest_quote, req)
            else:
                req = StockLatestQuoteRequest(symbol_or_symbols=symbols)
                quotes = _retry(self._stock_client.get_stock_latest_quote, req)
        except Exception as e:
            logger.warning("Quote fetch failed: %s", e)
            return {}
        result = {}
        for sym in symbols:
            q = quotes.get(sym)
            if q:
                result[sym] = {
                    "bid": float(q.bid_price) if q.bid_price else None,
                    "ask": float(q.ask_price) if q.ask_price else None,
                    "bid_size": float(q.bid_size) if q.bid_size else None,
                    "ask_size": float(q.ask_size) if q.ask_size else None,
                }
        return result

    def market_open(self) -> bool:
        if self.config.asset_class == "crypto":
            return True
        clock = _retry(self.trade_client.get_clock)
        return bool(clock.is_open)  # type: ignore[union-attr]

    def next_open(self) -> datetime:
        return _retry(self.trade_client.get_clock).next_open  # type: ignore[union-attr]

    def next_close(self) -> datetime:
        return _retry(self.trade_client.get_clock).next_close  # type: ignore[union-attr]

    def cancel_open_orders(self, symbol: str | None = None) -> None:
        if symbol is None:
            _retry(self.trade_client.cancel_orders)
            return
        try:
            orders = _retry(
                self.trade_client.get_orders,
                filter=GetOrdersRequest(symbols=[symbol]),
            )
        except Exception as e:
            logger.warning("Failed to fetch open orders: %s", e)
            return
        for o in orders:
            try:
                _retry(
                    self.trade_client.cancel_order_by_id,
                    order_id=o.id,  # type: ignore[union-attr]
                )
            except Exception as e:
                logger.warning("Cancel failed for %s: %s", o.symbol, e)  # type: ignore[union-attr]

    def close_all_positions(self) -> None:
        _retry(self.trade_client.close_all_positions)

    def submit_market_order(self, symbol: str, qty: float, side: OrderSide) -> dict:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.GTC
            if self.config.asset_class == "crypto"
            else TimeInForce.DAY,
        )
        return _retry(self.trade_client.submit_order, order_data=order)

    def reconcile(self, signals: dict[str, dict]) -> list:
        """Reconcile signals with the broker. Returns a list of trade tuples.

        Each tuple is `(ticker, qty, action)` where action is one of
        `BUY` / `SELL` / `NO_EQUITY` / `MAX_POS_CAP` / `PORTFOLIO_CAP` / `*_FAIL:<exc>`.

        Semantics note on the equity checks: when `account.equity <= 0`,
        `qty * ask` would always trip `MAX_POS_CAP` since `max_pos_value`
        collapses to 0. We emit `NO_EQUITY` before the cap so log output
        names the actual cause; the cap check still guards at low equity.
        """
        positions = self.get_positions()
        account = self.get_account()
        trades = []
        equity = float(account.get("equity") or 0.0)
        max_pos_value = max(0.0, equity * self.config.trade_max_position_pct)

        existing_notional = sum(p["market_value"] for p in positions.values())
        portfolio_capped = (
            equity > 0 and existing_notional > equity * self.config.max_portfolio_pct
        )

        # Bulk cancel all open orders — single API call instead of per-ticker.
        # Cancelling HOLD tickers' stale orders is harmless; they won't be re-filled
        # this cycle since reconcile only acts on BUY/SELL signals.
        self.cancel_open_orders()

        buy_tickers = [
            t
            for t, info in signals.items()
            if info["signal"] == "BUY" and t not in positions
        ]
        quotes = self.get_latest_quotes(buy_tickers) if buy_tickers else {}

        for ticker, info in signals.items():
            signal = info["signal"]
            pos = positions.get(ticker)
            has_pos = pos is not None

            if (
                signal == "BUY" and not has_pos
            ):  # no-pyramid: won't add to existing positions
                qty = self.config.trade_buy_qty
                if equity <= 0:
                    trades.append((ticker, 0, "NO_EQUITY"))
                    continue
                ask = quotes.get(ticker, {}).get("ask")
                if ask is None or ask <= 0:
                    logger.warning(
                        "No usable ask for %s; refusing buy without cap check", ticker
                    )
                    trades.append((ticker, 0, "NO_ASK"))
                    continue
                if portfolio_capped:
                    existing_pct = existing_notional / equity
                    trades.append((ticker, 0, f"PORTFOLIO_CAP:{existing_pct:.0%}"))
                    continue
                if qty * ask > max_pos_value:
                    trades.append((ticker, 0, "MAX_POS_CAP"))
                    continue
                try:
                    self.submit_market_order(ticker, qty, OrderSide.BUY)
                    trades.append((ticker, qty, "BUY"))
                except Exception as e:
                    trades.append((ticker, 0, f"BUY_FAIL:{e}"))

            elif signal == "SELL" and has_pos:
                # floor (not round): for fractional shares, rounding UP could
                # flip a short into a long (e.g., covering 3.9 of a -4 short
                # with round(3.9) = 4 would buy 4, leaving a spurious +0.1 long
                # position). floor leaves any fractional remainder in place.
                held = math.floor(abs(pos["qty"]))
                qty = min(held, self.config.trade_sell_qty)
                side = OrderSide.SELL if pos["side"] == "long" else OrderSide.BUY
                try:
                    self.submit_market_order(ticker, qty, side)
                    trades.append((ticker, qty, "SELL"))
                except Exception as e:
                    trades.append((ticker, 0, f"SELL_FAIL:{e}"))

        self._audit(trades, equity)
        return trades

    def _audit(self, trades: list, equity: float) -> None:
        if not trades:
            return
        ts = datetime.now(self.nyc).strftime("%H:%M:%S")
        header = not self._audit_path.exists()
        with self._audit_path.open("a") as f:
            if header:
                f.write("ts,ticker,action,qty,equity\n")
            for t in trades:
                f.write(f"{ts},{t[0]},{t[2]},{t[1]},{equity:.2f}\n")
