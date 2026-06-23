import time
from argparse import ArgumentParser
from datetime import datetime
from zoneinfo import ZoneInfo

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from config import Config, get_sp500_tickers
from src.inference import run_inference
from src.paper_trader import PaperTrader
from src.utils import load_threshold

DRACULA = Theme(
    {
        "info": "#bd93f9",
        "success": "#50fa7b",
        "warning": "#ffb86c",
        "error": "#ff5555",
        "highlight": "#8be9fd",
        "dim": "#6272a4",
        "title": "#ff79c6",
    }
)

console = Console(theme=DRACULA)


def make_trade_table(
    results: dict,
    positions: dict,
    trades: list,
    account: dict,
    cycle: int,
    interval: int,
    now_str: str,
) -> Table:
    table = Table(
        title=f"[title]Paper Trading Bot[/title]  |  "
        f"Equity: [success]${account.get('equity', 0):,.0f}[/success]  "
        f"Cash: ${(account.get('cash') or 0):,.0f}  "
        f"Day Δ: [{'error' if account.get('day_change', 0) < 0 else 'success'}]"
        f"${account.get('day_change', 0):+,.0f}[/]",
        title_style="bold",
        border_style="purple",
        padding=(0, 1),
    )
    table.add_column("Symbol", style="highlight", width=8)
    table.add_column("Pos", justify="right", style="info", width=6)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Signal", width=7)
    table.add_column("Action", width=14)
    table.add_column("P&L", justify="right", width=12)

    trade_map = {t[0]: t for t in trades}
    for ticker, info in list(results.items())[:30]:
        score = info["score"]
        signal = info["signal"]
        pos = positions.get(ticker)
        t = trade_map.get(ticker)

        pos_str = f"{round(pos['qty'])}" if pos else "—"
        if pos:
            pl = pos["unrealized_pl"]
            pl_str = f"[success]${pl:+,.0f}[/]" if pl >= 0 else f"[error]${pl:+,.0f}[/]"
        else:
            pl_str = "—"

        if signal == "BUY":
            sig_style = "success"
        elif signal == "SELL":
            sig_style = "error"
        else:
            sig_style = "dim"

        action_str = "—"
        if t:
            if t[2] == "BUY":
                action_str = f"[success]BUY {int(t[1])}[/]"
            elif t[2] == "SELL":
                action_str = f"[error]SELL {int(t[1])}[/]"
            else:
                action_str = f"[warning]{t[2]}[/]"

        table.add_row(
            ticker,
            pos_str,
            f"{score:+.3f}",
            f"[{sig_style}]{signal}[/]",
            action_str,
            pl_str,
        )

    table.add_section()
    table.add_row(
        f"[dim]Cycle #{cycle} | {now_str} | Next: ~{interval}s[/]",
        "",
        "",
        "",
        "",
        "",
    )
    return table


def build_layout(table: Table) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(Panel(table, border_style="purple")),
        Layout(
            Panel(
                "[dim]Dracula Theme · Alpaca Paper Trading · Ctrl+C to stop[/dim]",
                border_style="dim",
            ),
            size=3,
        ),
    )
    return layout


def main():
    parser = ArgumentParser(description="Paper trading bot using Alpaca")
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Minutes between trading cycles (default: 15)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without Rich display (log only)",
    )
    parser.add_argument(
        "--buy-threshold",
        type=float,
        default=None,
        help="Override buy threshold (default: from training)",
    )
    parser.add_argument(
        "--sell-threshold",
        type=float,
        default=None,
        help="Override sell threshold (default: from training)",
    )
    args = parser.parse_args()

    config = Config()
    config.trade_interval_minutes = args.interval
    config.tickers = get_sp500_tickers()
    print(f"Loaded {len(config.tickers)} tickers")

    buy_t, sell_t = load_threshold(config)
    if args.buy_threshold is not None:
        buy_t = args.buy_threshold
    if args.sell_threshold is not None:
        sell_t = args.sell_threshold

    trader = PaperTrader(config)
    nyc = ZoneInfo("America/New_York")

    cycle = 0
    while True:
        try:
            cycle += 1
            now = datetime.now(nyc)
            now_str = now.strftime("%Y-%m-%d %H:%M:%S ET")

            if not trader.market_open():
                nxt = trader.next_open()
                wait = (
                    nxt.replace(tzinfo=None) - now.replace(tzinfo=None)
                ).total_seconds()
                wait_m = max(1, int(wait / 60))
                if not args.headless:
                    console.print(
                        f"[warning]Market closed. Next open ~{wait_m} min[/warning]"
                    )
                time.sleep(min(wait, 300))
                continue

            account = trader.get_account()
            signals = run_inference(config, buy_threshold=buy_t, sell_threshold=sell_t)
            positions = trader.get_positions()
            trades = trader.reconcile(signals)

            if not args.headless:
                table = make_trade_table(
                    signals,
                    positions,
                    trades,
                    account,
                    cycle,
                    args.interval * 60,
                    now_str,
                )
                console.clear()
                console.print(build_layout(table))
            else:
                n_trades = len([t for t in trades if "FAIL" not in str(t[2])])
                print(
                    f"[{now_str}] Cycle #{cycle} | "
                    f"Equity: ${account.get('equity', 0):,.0f} | "
                    f"Trades: {n_trades}"
                )

            time.sleep(args.interval * 60)

        except KeyboardInterrupt:
            console.print("\n[warning]Shutting down...[/warning]")
            break
        except Exception as e:
            console.print(f"[error]Cycle error: {e}[/error]")
            time.sleep(30)


if __name__ == "__main__":
    main()
