import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from config import Config

logger = logging.getLogger(__name__)


class PaperTrader:
    def __init__(self, config: Config):
        key = config.alpaca_api_key or os.environ.get("ALPACA_API_KEY", "")
        secret = config.alpaca_secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        if not key or not secret:
            msg = "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env"
            raise ValueError(msg)
        self.config = config
        self.trade_client = TradingClient(key, secret, paper=config.alpaca_paper)
        self.data_client = StockHistoricalDataClient(key, secret)
        self.nyc = ZoneInfo("America/New_York")
        self._positions_cache: dict[str, dict] = {}
        self._account_cache: dict = {}

    def get_account(self):
        acct = self.trade_client.get_account()
        self._account_cache = {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "day_change": float(acct.equity) - float(acct.last_equity),
        }
        return self._account_cache

    def get_positions(self):
        positions = self.trade_client.get_all_positions()
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
            req = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self.data_client.get_stock_latest_quote(req)
        except Exception:
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
        clock = self.trade_client.get_clock()
        return bool(clock.is_open)  # type: ignore[union-attr]

    def next_open(self) -> datetime:
        return self.trade_client.get_clock().next_open  # type: ignore[union-attr]

    def next_close(self) -> datetime:
        return self.trade_client.get_clock().next_close  # type: ignore[union-attr]

    def cancel_open_orders(self, symbol: str | None = None):
        if symbol:
            orders = self.trade_client.get_orders(
                filter=GetOrdersRequest(symbols=[symbol] if symbol else None)
            )
        else:
            orders = self.trade_client.get_orders()
        for o in orders:
            try:
                self.trade_client.cancel_order_by_id(order_id=o.id)  # type: ignore[union-attr]
            except Exception as e:
                logger.warning("Cancel failed for %s: %s", o.symbol, e)  # type: ignore[union-attr]

    def submit_market_order(self, symbol: str, qty: int, side: OrderSide):
        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        return self.trade_client.submit_order(order_data=order)

    def reconcile(self, signals: dict[str, dict]):
        """Reconcile signals with the broker. Returns a list of trade tuples.

        Each tuple is `(ticker, qty, action)` where action is one of
        `BUY` / `SELL` / `NO_EQUITY` / `MAX_POS_CAP` / `*_FAIL:<exc>`.

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

        # Only cancel orders for tickers we are about to act on. Skipping HOLD
        # tickers saves up to ~480 Alpaca calls per cycle on a full-universe
        # signal set.
        actionable = [
            t for t, info in signals.items() if info["signal"] in ("BUY", "SELL")
        ]
        for ticker in actionable:
            self.cancel_open_orders(symbol=ticker)

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

            if signal == "BUY" and not has_pos:
                qty = self.config.trade_buy_qty
                if equity <= 0:
                    trades.append((ticker, 0, "NO_EQUITY"))
                    continue
                ask = quotes.get(ticker, {}).get("ask")
                if ask is None or ask <= 0:
                    logger.warning(
                        "No usable ask for %s; skipping position-cap check", ticker
                    )
                elif qty * ask > max_pos_value:
                    trades.append((ticker, 0, "MAX_POS_CAP"))
                    continue
                try:
                    self.submit_market_order(ticker, qty, OrderSide.BUY)
                    trades.append((ticker, qty, "BUY"))
                except Exception as e:
                    trades.append((ticker, 0, f"BUY_FAIL:{e}"))

            elif signal == "SELL" and has_pos:
                held = round(abs(pos["qty"]))
                qty = (
                    min(held, self.config.trade_sell_qty)
                    if held > self.config.trade_sell_qty
                    else held
                )
                side = OrderSide.SELL if pos["side"] == "long" else OrderSide.BUY
                try:
                    self.submit_market_order(ticker, qty, side)
                    trades.append((ticker, qty, "SELL"))
                except Exception as e:
                    trades.append((ticker, 0, f"SELL_FAIL:{e}"))

        return trades
