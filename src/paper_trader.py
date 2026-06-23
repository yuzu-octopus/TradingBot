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
        positions = self.get_positions()
        account = self.get_account()
        trades = []
        max_pos_value = account["equity"] * self.config.trade_max_position_pct

        for ticker in signals:
            self.cancel_open_orders(symbol=ticker)

        for ticker, info in signals.items():
            signal = info["signal"]
            pos = positions.get(ticker)
            has_pos = pos is not None

            if signal == "BUY" and not has_pos:
                qty = self.config.trade_buy_qty
                notional = qty * 100
                if notional > max_pos_value and account["equity"] > 0:
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
