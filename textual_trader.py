"""Textual TUI for paper trading --- stocks and crypto."""
# ruff: noqa: E402

import os
import sys

# Monkey-patch tqdm BEFORE any import touches it.
# tqdm.__new__ creates a multiprocessing RLock that triggers the resource
# tracker, which calls stderr.fileno(). Textual returns -1, causing
# "bad value in fds_to_keep" on Python 3.14.
import tqdm.std as _tqdm_std


class _NoopTqdm:
    def __init__(self, *a, **kw):  # noqa: ANN002,ANN003
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):  # noqa: ANN002
        pass

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

    def __contains__(self, _):
        return False

    def __getattr__(self, _):
        return lambda *a, **kw: None  # noqa: ARG005

    @staticmethod
    def write(s, file=None, end="\n", nolock=False) -> None:  # noqa: ARG004  # noqa: ARG004
        print(s, file=file or sys.stderr, end=end)


_tqdm_std.tqdm = _NoopTqdm

"""Textual TUI for paper trading — stocks and crypto."""

import asyncio
from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import ClassVar
from zoneinfo import ZoneInfo

from textual.app import App, ComposeResult
from textual.command import Hit, Hits, Provider
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Static

from config import Config, get_sp500_tickers
from src.inference import run_inference
from src.paper_trader import PaperTrader
from src.utils import load_threshold


class HelpScreen(ModalScreen[None]):
    """Keyboard shortcuts and usage help."""

    CSS = """
    HelpScreen {
        align: center middle;
    }
    #help-dialog {
        width: 50;
        height: auto;
        padding: 2;
        background: $surface;
        border: thick $primary;
    }
    #help-dialog Static {
        margin-bottom: 1;
    }
    #help-dialog .title {
        text-style: bold;
        color: $accent;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="help-dialog"):
            yield Static(
                "[bold]Trading Bot — Keyboard & Commands[/]\n\n"
                "[bold]Key[/]    [bold]Action[/]\n"
                "───     ──────\n"
                "R       Refresh data now\n"
                "S       Toggle asset class (stocks / crypto)\n"
                "C       Open theme picker\n"
                "Cmd+P   Open command palette\n"
                "H       Show this help\n"
                "Q       Quit\n\n"
                "[dim]Built with Textual · Alpaca Paper Trading[/]"
            )

    def on_key(self, event) -> None:
        if event.key in ("escape", "h", "q"):
            self.app.pop_screen()
            event.prevent_default()


class TradingCommands(Provider):
    """Custom command palette commands."""

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        app = self.app
        assert isinstance(app, TradingApp)

        commands = [
            ("Refresh data", "refresh", "Run inference + trade cycle now"),
            (
                "Toggle stocks/crypto",
                "toggle_asset",
                "Switch between S&P 500 and crypto",
            ),
            ("Open theme picker", "search_themes", "Browse and apply a theme"),
            ("Show help", "show_help", "View keyboard shortcuts"),
            ("Quit", "quit", "Exit the application"),
        ]

        for title, action, help_text in commands:
            score = matcher.match(title)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(title),
                    partial(self._run_action, app, action),
                    help=help_text,
                )

    @staticmethod
    def _run_action(app: TradingApp, action: str) -> None:
        getattr(app, f"action_{action}")()


class MetricCard(Static):
    value = reactive("")

    def __init__(self, label: str, initial: str = "—") -> None:
        super().__init__()
        self.label = label
        self.value = initial

    def watch_value(self, new_value: str) -> None:
        self.update(f"{self.label}: {new_value}")


def _load_dotenv() -> None:
    """Load .env file if present (uv run doesn't auto-load it)."""
    # os already imported at module scope

    env_path = Path(".env")
    if env_path.exists():
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key not in os.environ:
                os.environ[key] = val


class TradingApp(App):
    COMMANDS = App.COMMANDS | {TradingCommands}

    CSS = """
    Screen { layout: vertical; }
    Header { background: $primary; }
    #asset-row {
        height: 3;
        padding: 0 1;
        background: $surface;
        align: center middle;
    }
    #asset-label {
        width: auto;
        content-align: left middle;
        color: $text;
        padding: 0 1;
    }
    Button {
        width: 16;
        margin: 0 1;
    }
    Button.-active {
        background: $success;
        color: $text;
    }
    Button.-inactive {
        background: $surface;
        color: $text-muted;
    }
    #market-dot {
        width: 3;
        content-align: center middle;
    }
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
        ("s", "toggle_asset", "Stocks/Crypto"),
        ("+", "interval_up", "Faster"),
        ("-", "interval_down", "Slower"),
        ("[", "threshold_down", "Buy\u2193"),
        ("]", "threshold_up", "Buy\u2191"),
        ("{", "sell_threshold_down", "Sell\u2193"),
        ("}", "sell_threshold_up", "Sell\u2191"),
        ("c", "search_themes", "Theme"),
        ("h", "show_help", "Help"),
        ("q", "quit", "Quit"),
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
        self._asset_class = config.asset_class

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="asset-row"):
            yield Static("", id="market-dot")
            yield Static("Trading:", id="asset-label")
            yield Button("S&P 500", id="btn-stocks", variant="primary")
            yield Button("Crypto", id="btn-crypto", variant="primary")
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
        self._refresh_buttons()
        self.run_worker(self._refresh_cycle(), name="cycle", exclusive=True)
        self.set_interval(self._interval, self._on_timer)

    def _refresh_buttons(self) -> None:
        is_crypto = self._asset_class == "crypto"
        self.query_one("#btn-stocks", Button).classes = (
            "" if not is_crypto else "inactive"
        )
        self.query_one("#btn-crypto", Button).classes = "" if is_crypto else "inactive"
        dot = self.query_one("#market-dot", Static)
        if is_crypto:
            dot.update("[green]\u25cf[/] Crypto 24/7")
        else:
            try:
                is_open = self._trader.market_open()
                dot.update(
                    f"[{'green' if is_open else 'red'}]●[/] {'Open' if is_open else 'Closed'}"
                )
            except Exception:
                dot.update("[yellow]\u25cf[/] ?")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-stocks" and self._asset_class != "stocks":
            self._switch_asset("stocks")
        elif event.button.id == "btn-crypto" and self._asset_class != "crypto":
            self._switch_asset("crypto")

    def _switch_asset(self, target: str) -> None:
        self._asset_class = target
        self._config.asset_class = target
        if target == "crypto":
            from config import CRYPTO_PAIR_MAP

            self._config.tickers = CRYPTO_PAIR_MAP[self._config.crypto_pairs]
            self._config.raw_data_path = "data/crypto/raw"
            self._config.features_path = "data/crypto/features"
            self._config.model_save_path = "data/models/crypto/best.pt"
        else:
            self._config.tickers = get_sp500_tickers()
            self._config.raw_data_path = "data/stocks"
            self._config.features_path = "data/features"
            self._config.model_save_path = "data/models/best.pt"
        self._refresh_buttons()
        self.notify(f"Switched to {target}", severity="information")
        self.run_worker(self._refresh_cycle(), name="switch")

    async def _on_timer(self) -> None:
        self.run_worker(self._refresh_cycle(), name="cycle", exclusive=True)

    async def _refresh_cycle(self) -> None:
        table = self.query_one("#signals", DataTable)
        status = self.query_one("#status-bar", Static)
        self._cycle += 1
        status.update(f"Cycle #{self._cycle} — running inference...")

        try:
            if not self._trader.market_open() and self._asset_class != "crypto":
                nxt = self._trader.next_open()
                wait = (
                    nxt.replace(tzinfo=None)
                    - datetime.now(self._nyc).replace(tzinfo=None)
                ).total_seconds()
                status.update(f"Market closed — next open ~{max(1, int(wait / 60))}m")
                self._refresh_buttons()
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
                f"Cycle #{self._cycle} | {now_str} | Next: ~{self._interval // 60}m | "
                f"Buy\u2248{self._buy_t:.2f} Sell\u2248{self._sell_t:.2f} | {spark}"
            )
            self._refresh_buttons()
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
            pos_str = str(round(pos["qty"])) if pos else "\u2014"
            alloc = (
                f"{pos['market_value'] / equity * 100:.1f}%"
                if pos and equity > 0
                else ""
            )
            pl = pos["unrealized_pl"] if pos else 0
            pl_str = (
                f"[green]${pl:+,.0f}[/]"
                if pl > 0
                else (f"[red]${pl:+,.0f}[/]" if pl < 0 else "\u2014")
            )
            trade_str = "\u2014"
            if t:
                act = t[2]
                if act == "BUY":
                    trade_str = f"[green]BUY {int(t[1])}[/]"
                elif act == "SELL":
                    trade_str = f"[red]SELL {int(t[1])}[/]"
                else:
                    trade_str = f"[yellow]{act}[/]"
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

    def action_interval_up(self) -> None:
        self._interval = min(3600, self._interval + 60)
        self.notify(f"Interval: {self._interval // 60}m", severity="information")

    def action_interval_down(self) -> None:
        self._interval = max(60, self._interval - 60)
        self.notify(f"Interval: {self._interval // 60}m", severity="information")

    def action_threshold_up(self) -> None:
        self._buy_t = min(0.99, round(self._buy_t + 0.05, 2))
        self.notify(f"Buy threshold: {self._buy_t:.2f}", severity="information")

    def action_threshold_down(self) -> None:
        self._buy_t = max(0.01, round(self._buy_t - 0.05, 2))
        self.notify(f"Buy threshold: {self._buy_t:.2f}", severity="information")

    def action_sell_threshold_up(self) -> None:
        self._sell_t = min(0.99, round(self._sell_t + 0.05, 2))
        self.notify(f"Sell threshold: {self._sell_t:.2f}", severity="information")

    def action_sell_threshold_down(self) -> None:
        self._sell_t = max(0.01, round(self._sell_t - 0.05, 2))
        self.notify(f"Sell threshold: {self._sell_t:.2f}", severity="information")

    def action_toggle_asset(self) -> None:
        self._switch_asset("crypto" if self._asset_class == "stocks" else "stocks")

    def action_search_themes(self) -> None:
        self.search_themes()

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())


def main() -> None:
    _load_dotenv()
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
