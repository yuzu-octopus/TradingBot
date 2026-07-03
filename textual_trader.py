"""Textual TUI for paper trading --- stocks and crypto."""

import asyncio
import logging
import math
import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import ClassVar
from zoneinfo import ZoneInfo

import tqdm.std as _tqdm_std
from textual.app import App, ComposeResult
from textual.command import Hit, Hits, Provider
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import (
    Collapsible,
    DataTable,
    Digits,
    Footer,
    Header,
    Input,
    Label,
    MarkdownViewer,
    ProgressBar,
    RichLog,
    Sparkline,
    Static,
    Tab,
    TabbedContent,
)

from config import Config, get_sp500_tickers
from src.inference import run_inference
from src.paper_trader import PaperTrader
from src.utils import load_threshold

logger = logging.getLogger(__name__)


class _NoopTqdm:
    """Silent no-op replacement; never creates multiprocessing locks."""

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
    def write(s, file=None, end="\n", nolock=False) -> None:  # noqa: ARG004
        print(s, file=file or sys.stderr, end=end)


_ORIGINAL_TQDM = _tqdm_std.tqdm


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
                os.environ[key] = val.strip('"')


class ThresholdConfirm(ModalScreen[bool]):
    """Confirmation dialog for aggressive threshold values."""

    CSS = """
    ThresholdConfirm {
        align: center middle;
    }
    #threshold-dialog {
        width: 40;
        height: auto;
        padding: 2;
        background: $surface;
        border: thick $warning;
    }
    #threshold-dialog Label {
        margin-bottom: 1;
    }
    """

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="threshold-dialog"):
            yield Label("[bold yellow]Confirm Aggressive Threshold[/]")
            yield Label(self._message)
            yield Label("Press [bold]Y[/] to confirm, any other key to cancel.")

    def on_key(self, event) -> None:
        if event.key.lower() == "y":
            self.dismiss(result=True)
        elif event.key in ("escape", "n", "q"):
            self.dismiss(result=False)


class LiquidateConfirm(ModalScreen[bool]):
    """Modal to confirm liquidation of all positions."""

    CSS = """
    LiquidateConfirm {
        align: center middle;
    }
    #liquidate-dialog {
        width: 40;
        height: auto;
        padding: 2;
        background: $surface;
        border: thick $error;
    }
    #liquidate-dialog Label {
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="liquidate-dialog"):
            yield Label("[bold red]Liquidate ALL Positions?[/]")
            yield Label("This will close all open positions and cancel all orders.")
            yield Label("Type [bold]YES[/] to confirm:")
            yield Input(placeholder="YES", id="liquidate-input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.value.strip().upper() == "YES":
            self.dismiss(result=True)
        else:
            self.query_one("#liquidate-input", Input).value = ""
            self.notify("Type YES to confirm", severity="warning")


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
    TabbedContent {
        width: 1fr;
        height: auto;
    }
    TabbedContent > Tab {
        padding: 0 2;
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
    #equity-digits {
        width: 1fr;
        height: 3;
        content-align: center middle;
        color: $text;
    }
    .equity-digits {
        width: 1fr;
        height: 3;
    }
    #inference-progress {
        height: 1;
        margin: 0 1;
        display: none;
    }
    #inference-progress.-visible {
        display: block;
    }
    MetricCard {
        width: 1fr;
        content-align: center middle;
        color: $text;
    }
    #model-status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $surface;
    }
    DataTable { height: 1fr; }
    #signals .datatable--header:nth-child(2),
    #signals .datatable--header:nth-child(3),
    #signals .datatable--header:nth-child(4),
    #signals .datatable--header:nth-child(6),
    #signals .datatable--cell:nth-child(2),
    #signals .datatable--cell:nth-child(3),
    #signals .datatable--cell:nth-child(4),
    #signals .datatable--cell:nth-child(6) {
        text-align: right;
    }
    #strategy-notes {
        height: 8;
        margin: 0 1;
    }
    RichLog { height: 6; margin: 0 1; }
    Sparkline { height: 1; }
    .flash-up { background: $success 20%; }
    .flash-down { background: $error 20%; }
    #threshold-panel {
        padding: 0 1;
    }
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
        ("l", "liquidate", "Liquidate"),
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
        self._timer: Timer | None = None
        self._err_strikes = 0
        self._last_error: str | None = None
        self._error_count = 0
        self._last_refresh: float | None = None
        self._last_session_path = Path("data/last_session.json")
        self._load_session()
        self._asset_class = config.asset_class

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="asset-row"):
            yield Static("", id="market-dot")
            yield Static("Trading:", id="asset-label")
            with TabbedContent(initial="stocks", id="asset-tabs"):
                yield Tab("S&P 500", id="stocks")
                yield Tab("Crypto", id="crypto")
        with Horizontal(id="metric-row"):
            yield Digits(id="equity-digits", classes="equity-digits")
            yield MetricCard("Cash")
            yield MetricCard("Day Δ")
            yield MetricCard("Positions")
            yield MetricCard("Cycle")
        yield ProgressBar(id="inference-progress")
        yield Static(id="model-status")
        with Collapsible(title="Threshold Tuning", id="threshold-panel"):
            yield Static(
                f"Buy: {self._buy_t:.2f}  Sell: {self._sell_t:.2f}",
                id="threshold-info",
            )
        yield DataTable(id="signals")
        yield MarkdownViewer(
            "# Strategy Notes\n\n"
            "This bot uses a StockTransformer model with RankGLU output heads.\n"
            "Market-guided gating via SPY state rescales features each day.\n\n"
            "Create `data/strategy_notes.md` to customize this description.",
            id="strategy-notes",
            show_table_of_contents=False,
        )
        yield Sparkline(id="equity-spark")
        yield RichLog(id="trade-log", highlight=True, max_lines=10)
        yield Static("Ready", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        _tqdm_std.tqdm = _NoopTqdm  # type: ignore[misc,assignment]
        self.query_one(Header).sub_title = "PAPER TRADING"
        model_name = Path(self._config.model_save_path).name
        self.query_one("#model-status", Static).update(f"Model: {model_name}")
        self.query_one("#threshold-info", Static).update(
            f"Buy: {self._buy_t:.2f}  Sell: {self._sell_t:.2f}"
        )
        table = self.query_one("#signals", DataTable)
        table.add_columns("Sym", "Pos", "%", "Score", "Trade", "P&L")
        table.cursor_type = "row"
        table.zebra_stripes = True
        self._refresh_buttons()
        self.run_worker(self._refresh_cycle(), name="cycle", exclusive=True)
        self._timer = self.set_interval(self._interval, self._on_timer)

    def on_unmount(self) -> None:
        _tqdm_std.tqdm = _ORIGINAL_TQDM  # type: ignore[misc,assignment]

    def _refresh_buttons(self) -> None:
        is_crypto = self._asset_class == "crypto"
        tabs = self.query_one("#asset-tabs", TabbedContent)
        tabs.active = "crypto" if is_crypto else "stocks"
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
                dot.update("[yellow]\u25cf[/] ??")

    def _preview_threshold(self) -> None:
        """Run a quick inference preview and toast BUY/SELL/HOLD counts."""
        self.run_worker(self._preview_threshold_async(), exclusive=False)

    async def _preview_threshold_async(self) -> None:
        try:
            sigs = await self._run_in_thread(
                run_inference,
                self._config,
                buy_threshold=self._buy_t,
                sell_threshold=self._sell_t,
            )
            buys = sum(1 for s in sigs.values() if s["signal"] == "BUY")
            sells = sum(1 for s in sigs.values() if s["signal"] == "SELL")
            holds = sum(1 for s in sigs.values() if s["signal"] == "HOLD")
            self.notify(
                f"BUY: {buys}  SELL: {sells}  HOLD: {holds}",
                severity="information",
                timeout=3,
            )
        except Exception as e:
            logger.debug("Preview failed: %s", e)

    def on_tab_changed(self, event: TabbedContent.TabActivated) -> None:
        tab_id = event.tab.id
        if tab_id == "stocks" and self._asset_class != "stocks":
            self._switch_asset("stocks")
        elif tab_id == "crypto" and self._asset_class != "crypto":
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
        self._trader = PaperTrader(self._config)
        self._equity_history.clear()
        self._prev_equity = 0.0
        self.query_one("#equity-spark", Sparkline).data = []
        self._refresh_buttons()
        self.notify(f"Switched to {target}", severity="information", timeout=2)
        self.run_worker(self._refresh_cycle(), name="switch", exclusive=True)

    async def _on_timer(self) -> None:
        self.run_worker(self._refresh_cycle(), name="cycle", exclusive=True)

    async def _refresh_cycle(self) -> None:
        table = self.query_one("#signals", DataTable)
        status = self.query_one("#status-bar", Static)
        progress = self.query_one("#inference-progress", ProgressBar)
        self._cycle += 1
        progress.display = True
        progress.update(total=100, progress=0)
        status.update(f"Cycle #{self._cycle} — running inference...")

        try:
            if not self._trader.market_open() and self._asset_class != "crypto":
                nxt = self._trader.next_open()
                wait = (
                    nxt.replace(tzinfo=None)
                    - datetime.now(self._nyc).replace(tzinfo=None)
                ).total_seconds()
                status.update(f"Market closed — next open ~{max(1, int(wait / 60))}m")
                progress.display = False
                self._refresh_buttons()
                return

            account = await self._run_in_thread(self._trader.get_account)
            self._equity_history.append(account.get("equity", 0))
            if len(self._equity_history) > 100:
                self._equity_history.pop(0)

            progress.update(total=100, progress=30)
            signals = await self._run_in_thread(
                run_inference,
                self._config,
                buy_threshold=self._buy_t,
                sell_threshold=self._sell_t,
            )
            progress.update(total=100, progress=70)
            positions = await self._run_in_thread(self._trader.get_positions)
            trades = await self._run_in_thread(self._trader.reconcile, signals)

            self._update_metrics(account, positions)
            self._update_table(table, signals, positions, trades, account)
            log = self.query_one("#trade-log", RichLog)
            if trades:
                log.write("[dim]" + "-" * 40 + "[/]")
            for t in trades:
                ts = datetime.now(self._nyc).strftime("%H:%M:%S")
                act = t[2]
                sym = t[0]
                qty = t[1]
                if act == "BUY":
                    log.write(f"[green]{ts} BUY {int(qty)} {sym}[/]")
                elif act == "SELL":
                    log.write(f"[red]{ts} SELL {int(qty)} {sym}[/]")
                elif "FAIL" in act:
                    log.write(f"[red]{ts} {act} {sym}[/]")
                elif act in ("NO_ASK", "MAX_POS_CAP", "NO_EQUITY"):
                    log.write(f"[yellow]{ts} {act} {sym}[/]")

            now_str = datetime.now(self._nyc).strftime("%H:%M:%S ET")
            self.query_one("#equity-spark", Sparkline).data = self._equity_history[-50:]
            last_ref = ""
            if self._last_refresh is not None:
                secs = int(time.monotonic() - self._last_refresh)
                last_ref = f" | Last: {secs}s"
            self._last_refresh = time.monotonic()
            progress.display = False
            self.query_one("#threshold-info", Static).update(
                f"Buy: {self._buy_t:.2f}  Sell: {self._sell_t:.2f}"
            )
            status.update(
                f"Cycle #{self._cycle} | {now_str} | Next: ~{self._interval // 60}m | "
                f"Buy\u2248{self._buy_t:.2f} Sell\u2248{self._sell_t:.2f}{last_ref}"
            )
            self._refresh_buttons()
            self._err_strikes = 0
        except Exception as e:
            progress.display = False
            self._err_strikes += 1
            if self._err_strikes >= 3:
                self.query_one("#market-dot", Static).update(
                    "[red]\u25cf Disconnected[/]"
                )
            status.update(f"Error: {e}")
            err = str(e)
            if err == self._last_error:
                self._error_count += 1
                msg = (
                    f"{err} (\u00d7{self._error_count})"
                    if self._error_count > 1
                    else err
                )
            else:
                self._last_error = err
                self._error_count = 1
                msg = err
            self.notify(msg, severity="error", timeout=2)

    async def _run_in_thread(self, fn, *args: object, **kwargs: object):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    def _update_metrics(self, account: dict, positions: dict) -> None:
        prev = self._prev_equity if hasattr(self, "_prev_equity") else 0
        curr = account.get("equity", 0)
        equity_digits = self.query_one("#equity-digits", Digits)
        if prev and curr and curr != prev:
            cls = "flash-up" if curr > prev else "flash-down"
            equity_digits.add_class(cls)
            equity_digits.set_timer(0.3, lambda c=cls: equity_digits.remove_class(c))
        self._prev_equity = curr
        equity_digits.update(f"{curr:.2f}")
        self.query_one("#metric-row").children[
            1
        ].value = f"${account.get('cash', 0):,.0f}"  # type: ignore[attr-defined]
        dc = account.get("day_change", 0)
        style = "green" if dc >= 0 else "red"
        self.query_one("#metric-row").children[2].value = f"[{style}]${dc:+,.0f}[/]"  # type: ignore[attr-defined]
        self.query_one("#metric-row").children[3].value = f"{len(positions)}"  # type: ignore[attr-defined]
        self.query_one("#metric-row").children[4].value = f"#{self._cycle}"  # type: ignore[attr-defined]

    def _update_table(self, table, signals, positions, trades, account):
        table.clear()
        equity = account.get("equity", 0)
        trade_map = {t[0]: t for t in trades}
        for ticker, info in signals.items():
            score = info["score"]
            pos = positions.get(ticker)
            t = trade_map.get(ticker)
            held = math.floor(abs(pos["qty"])) if pos else 0
            pos_str = str(held) if pos else "\u2014"
            alloc = (
                f"{pos['market_value'] / equity * 100:.1f}%"
                if pos and equity > 0
                else ""
            )
            pl = pos["unrealized_pl"] if pos else 0
            pl_str = (
                f"[green]\u25b2${pl:+,.0f}[/]"
                if pl > 0
                else (f"[red]\u25bc-${abs(pl):,.0f}[/]" if pl < 0 else "\u2014")
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

    def action_refresh(self) -> None:
        self.notify("Refreshing...", severity="information", timeout=2)
        self.run_worker(self._refresh_cycle(), name="cycle", exclusive=True)

    def _save_session(self) -> None:
        import json

        Path("data").mkdir(exist_ok=True)
        with self._last_session_path.open("w") as f:
            json.dump(
                {
                    "interval": self._interval,
                    "buy_t": self._buy_t,
                    "sell_t": self._sell_t,
                },
                f,
            )

    def _load_session(self) -> None:
        import json

        if self._last_session_path.exists():
            try:
                d = json.loads(self._last_session_path.read_text())
                if not all(k in d for k in ("interval", "buy_t", "sell_t")):
                    logger.warning("Incomplete session data, skipping load")
                    return
                self._interval = d.get("interval", self._interval)
                self._buy_t = d.get("buy_t", self._buy_t)
                self._sell_t = d.get("sell_t", self._sell_t)
            except Exception as e:
                logger.warning("Failed to load session (using defaults): %s", e)

    def action_interval_up(self) -> None:
        self._interval = min(3600, self._interval + 60)
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_interval(self._interval, self._on_timer)
        self._save_session()
        self.notify(
            f"Interval: {self._interval // 60}m", severity="information", timeout=2
        )

    def action_interval_down(self) -> None:
        self._interval = max(60, self._interval - 60)
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_interval(self._interval, self._on_timer)
        self._save_session()
        self.notify(
            f"Interval: {self._interval // 60}m", severity="information", timeout=2
        )

    def _apply_threshold(self, side: str, change: float) -> None:
        """Apply a threshold change with UX16 confirmation for aggressive values."""
        if side == "buy":
            new_val = min(0.99, round(self._buy_t + change, 2))
            new_val = max(0.01, new_val)
            if new_val > 0.95 and new_val != self._buy_t:
                self.push_screen(
                    ThresholdConfirm(
                        f"Buy threshold {new_val:.2f} > 0.95 is very aggressive"
                    ),
                    lambda ok: self._commit_threshold("buy", new_val) if ok else None,
                )
            else:
                self._commit_threshold("buy", new_val)
        else:
            new_val = min(0.99, round(self._sell_t + change, 2))
            new_val = max(0.01, new_val)
            if new_val > 0.95 and new_val != self._sell_t:
                self.push_screen(
                    ThresholdConfirm(
                        f"Sell threshold {new_val:.2f} > 0.95 is very aggressive"
                    ),
                    lambda ok: self._commit_threshold("sell", new_val) if ok else None,
                )
            else:
                self._commit_threshold("sell", new_val)

    def _commit_threshold(self, side: str, new_val: float) -> None:
        if side == "buy":
            self._buy_t = new_val
        else:
            self._sell_t = new_val
        self._save_session()
        self.notify(
            f"{'Buy' if side == 'buy' else 'Sell'} threshold: {new_val:.2f}",
            severity="information",
            timeout=2,
        )
        self.query_one("#threshold-info", Static).update(
            f"Buy: {self._buy_t:.2f}  Sell: {self._sell_t:.2f}"
        )
        self._preview_threshold()

    def action_threshold_up(self) -> None:
        self._apply_threshold("buy", 0.05)

    def action_threshold_down(self) -> None:
        self._apply_threshold("buy", -0.05)

    def action_sell_threshold_up(self) -> None:
        self._apply_threshold("sell", 0.05)

    def action_sell_threshold_down(self) -> None:
        self._apply_threshold("sell", -0.05)

    def action_toggle_asset(self) -> None:
        self._switch_asset("crypto" if self._asset_class == "stocks" else "stocks")

    def action_search_themes(self) -> None:
        self.search_themes()

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_liquidate(self) -> None:
        def on_confirm(result: bool | None) -> None:
            if result:
                try:
                    self._trader.close_all_positions()
                    positions = self._trader.get_positions()
                    for ticker in positions:
                        self._trader.cancel_open_orders(ticker)
                    self.notify(
                        "All positions liquidated", severity="information", timeout=3
                    )
                except Exception as e:
                    self.notify(f"Liquidation failed: {e}", severity="error", timeout=3)

        self.push_screen(LiquidateConfirm(), on_confirm)


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
