"""Textual TUI for paper trading."""

import asyncio
from argparse import ArgumentParser
from datetime import datetime
from typing import ClassVar
from zoneinfo import ZoneInfo

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Static
from textual.worker import Worker

from config import Config, get_sp500_tickers
from src.inference import run_inference
from src.paper_trader import PaperTrader
from src.utils import load_threshold


class MetricCard(Static):
    value = reactive("")

    def __init__(self, label: str, initial: str = "—") -> None:
        super().__init__()
        self.label = label
        self.value = initial

    def watch_value(self, new_value: str) -> None:
        self.update(f"{self.label}: {new_value}")


class TradingApp(App):
    CSS = """
    Screen { layout: vertical; }
    Header { background: $primary; }
    #metric-row {
        height: 3;
        padding: 0 1;
        background: $surface;
    }
    MetricCard {
        width: 1fr;
        content-align: center middle;
        color: $text;
    }
    DataTable { height: 1fr; }
    #status-bar {
        height: 1;
        padding: 0 1;
        background: $panel;
        color: $text-muted;
    }
    Footer { height: 1; }
    """

    BINDINGS: ClassVar[list] = [
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
        ("t", "toggle_dark", "Toggle theme"),
    ]

    def __init__(
        self,
        config: Config,
        trader: PaperTrader,
        buy_t: float,
        sell_t: float,
        interval: int,
    ) -> None:
        super().__init__()
        self._config = config
        self._trader = trader
        self._buy_t = buy_t
        self._sell_t = sell_t
        self._interval = interval
        self._nyc = ZoneInfo("America/New_York")
        self._cycle = 0
        self._equity_history: list[float] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="metric-row"):
            yield MetricCard("Equity")
            yield MetricCard("Cash")
            yield MetricCard("Day Δ")
            yield MetricCard("Positions")
            yield MetricCard("Cycle")
        yield DataTable(id="signals")
        yield Static("Ready", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#signals", DataTable)
        table.add_columns("Sym", "Pos", "%", "Score", "Trade", "P&L")
        table.cursor_type = "row"
        table.zebra_stripes = True
        self.run_worker(self._refresh_cycle(), name="init")
        self.set_interval(self._interval, self._on_timer)

    async def _on_timer(self) -> None:
        self.run_worker(self._refresh_cycle(), name="cycle")

    async def _refresh_cycle(self) -> None:
        table = self.query_one("#signals", DataTable)
        status = self.query_one("#status-bar", Static)

        self._cycle += 1
        status.update(f"Cycle #{self._cycle} — running inference...")

        try:
            if not self._trader.market_open():
                nxt = self._trader.next_open()
                wait = (
                    nxt.replace(tzinfo=None)
                    - datetime.now(self._nyc).replace(tzinfo=None)
                ).total_seconds()
                status.update(f"Market closed — next open ~{max(1, int(wait / 60))}m")
                return

            account = await self._run_in_thread(self._trader.get_account)
            self._equity_history.append(account.get("equity", 0))
            if len(self._equity_history) > 100:
                self._equity_history.pop(0)

            signals = await self._run_in_thread(
                run_inference,
                self._config,
                buy_threshold=self._buy_t,
                sell_threshold=self._sell_t,
            )
            positions = await self._run_in_thread(self._trader.get_positions)
            trades = await self._run_in_thread(self._trader.reconcile, signals)

            self._update_metrics(account, positions)
            self._update_table(table, signals, positions, trades, account)
            now_str = datetime.now(self._nyc).strftime("%H:%M:%S ET")
            spark = self._sparkline(self._equity_history)
            status.update(
                f"Cycle #{self._cycle} | {now_str} | Next: ~{self._interval}s | {spark}"
            )

        except Exception as e:
            status.update(f"Error: {e}")
            self.notify(str(e), severity="error")

    async def _run_in_thread(self, fn, *args: object, **kwargs: object):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    def _update_metrics(self, account: dict, positions: dict) -> None:
        self.query_one("#metric-row").children[
            0
        ].value = f"${account.get('equity', 0):,.0f}"
        self.query_one("#metric-row").children[
            1
        ].value = f"${account.get('cash', 0):,.0f}"
        dc = account.get("day_change", 0)
        style = "green" if dc >= 0 else "red"
        self.query_one("#metric-row").children[2].value = f"[{style}]${dc:+,.0f}[/]"
        self.query_one("#metric-row").children[3].value = f"{len(positions)}"
        self.query_one("#metric-row").children[4].value = f"#{self._cycle}"

    def _update_table(self, table, signals, positions, trades, account):
        table.clear()
        equity = account.get("equity", 0)
        trade_map = {t[0]: t for t in trades}
        for ticker, info in signals.items():
            score = info["score"]
            pos = positions.get(ticker)
            t = trade_map.get(ticker)

            pos_str = str(round(pos["qty"])) if pos else "—"
            alloc = (
                f"{pos['market_value'] / equity * 100:.1f}%"
                if pos and equity > 0
                else ""
            )
            pl = pos["unrealized_pl"] if pos else 0
            pl_str = (
                f"[green]${pl:+,.0f}[/]"
                if pl > 0
                else (f"[red]${pl:+,.0f}[/]" if pl < 0 else "—")
            )

            trade_str = "—"
            if t:
                act = t[2]
                trade_str = (
                    f"[green]BUY {int(t[1])}[/]"
                    if act == "BUY"
                    else (
                        f"[red]SELL {int(t[1])}[/]"
                        if act == "SELL"
                        else f"[yellow]{act}[/]"
                    )
                )

            table.add_row(ticker, pos_str, alloc, f"{score:+.4f}", trade_str, pl_str)

    def _sparkline(self, values, width=20):
        if not values:
            return ""
        mn, mx = min(values), max(values)
        rng = mx - mn or 1
        chars = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
        return "".join(chars[min(7, int((v - mn) / rng * 7))] for v in values[-width:])

    def action_refresh(self) -> None:
        self.notify("Refreshing...", severity="information")
        self.run_worker(self._refresh_cycle())

    def action_toggle_dark(self) -> None:
        self.dark = not self.dark


def main() -> None:
    parser = ArgumentParser(description="Textual TUI paper trader")
    parser.add_argument(
        "--interval", type=int, default=15, help="Minutes between cycles"
    )
    parser.add_argument("--buy-threshold", type=float, default=None)
    parser.add_argument("--sell-threshold", type=float, default=None)
    parser.add_argument("--asset-class", choices=["stocks", "crypto"], default="stocks")
    parser.add_argument("--crypto-pairs", choices=["top10", "all17"], default="top10")
    args = parser.parse_args()

    config = Config()
    config.asset_class = args.asset_class
    config.crypto_pairs = args.crypto_pairs
    config.trade_interval_minutes = args.interval
    if config.asset_class == "crypto":
        from config import CRYPTO_PAIR_MAP

        config.tickers = CRYPTO_PAIR_MAP[config.crypto_pairs]
        config.raw_data_path = "data/crypto/raw"
        config.features_path = "data/crypto/features"
        config.model_save_path = "data/models/crypto/best.pt"
    else:
        config.tickers = get_sp500_tickers()
    print(f"Loaded {len(config.tickers)} tickers")

    buy_t, sell_t = load_threshold(config)
    if args.buy_threshold is not None:
        buy_t = args.buy_threshold
    if args.sell_threshold is not None:
        sell_t = args.sell_threshold

    trader = PaperTrader(config)
    app = TradingApp(config, trader, buy_t, sell_t, args.interval * 60)
    app.run()


if __name__ == "__main__":
    main()
