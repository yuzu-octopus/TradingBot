import time
from argparse import ArgumentParser
from datetime import datetime
from zoneinfo import ZoneInfo

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from config import Config, get_sp500_tickers
from src.inference import run_inference
from src.paper_trader import PaperTrader
from src.utils import load_threshold

_THEME = Theme(
    {
        "info": "bright_magenta",
        "success": "green",
        "warning": "yellow",
        "error": "red",
        "highlight": "cyan",
        "dim": "bright_black",
        "title": "bright_red",
        "border": "bright_magenta",
    }
)
console = None


def _score_color(score: float) -> str:
    if score > 0.3:
        return "bold success"
    if score > 0.1:
        return "success"
    if score > 0:
        return "dim success"
    if score > -0.1:
        return "dim error"
    if score > -0.3:
        return "error"
    return "bold error"


def _sparkline(values, width=20):
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn or 1
    chars = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    return "".join(chars[min(7, int((v - mn) / rng * 7))] for v in values[-width:])


def make_trade_table(
    results, positions, trades, account, cycle, interval, now_str, equity_history=None
):
    equity = account.get("equity", 0)
    day_change = account.get("day_change", 0)
    title = (
        "[title]Paper Trading Bot[/title]  |  "
        f"Equity: [success]${equity:,.0f}[/success]  "
        f"Cash: ${(account.get('cash') or 0):,.0f}  "
        f"Day \u0394: [{'error' if day_change < 0 else 'success'}]"
        f"${day_change:+,.0f}[/]"
    )
    table = Table(
        title=title, title_style="bold", border_style="border", padding=(0, 1)
    )
    table.add_column("Sym", style="highlight", width=7)
    table.add_column("Pos", justify="right", style="info", width=6)
    table.add_column("%", justify="right", style="dim", width=5)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Trade", width=12)
    table.add_column("P&L", justify="right", width=10)

    trade_map = {t[0]: t for t in trades}
    for ticker, info in results.items():
        score = info["score"]
        signal = info["signal"]
        pos = positions.get(ticker)
        t = trade_map.get(ticker)

        pos_str = f"{round(pos['qty'])}" if pos else "\u2014"
        alloc = (
            f"{pos['market_value'] / equity * 100:.1f}" if pos and equity > 0 else ""
        )
        pl = pos["unrealized_pl"] if pos else 0
        pl_str = (
            f"[success]${pl:+,.0f}[/]"
            if pl > 0
            else (f"[error]${pl:+,.0f}[/]" if pl < 0 else "\u2014")
        )

        trade_str = "\u2014"
        if t:
            act = t[2]
            if act == "BUY":
                trade_str = f"[success]BUY {int(t[1])}[/]"
            elif act == "SELL":
                trade_str = f"[error]SELL {int(t[1])}[/]"
            else:
                trade_str = f"[warning]{act}[/]"
        elif signal == "HOLD":
            trade_str = "[dim]HOLD[/]"

        table.add_row(
            ticker,
            pos_str,
            alloc,
            f"[{_score_color(score)}]{score:+.4f}[/]",
            trade_str,
            pl_str,
        )

    spark = _sparkline(equity_history or [])
    table.add_section()
    table.add_row(
        f"[dim]Cycle #{cycle} | {now_str} | Next: ~{interval}s[/]",
        "",
        "",
        "",
        f"[dim]{spark}[/]",
        "",
    )
    return table


def build_layout(table):
    layout = Layout()
    layout.split_column(
        Layout(Panel(table, border_style="border")),
        Layout(
            Panel(
                "[dim]Alpaca Paper Trading \u00b7 Ctrl+C to stop[/dim]",
                border_style="dim",
            ),
            size=3,
        ),
    )
    return layout


def main():
    parser = ArgumentParser(description="Paper trading bot using Alpaca")
    parser.add_argument(
        "--interval", type=int, default=15, help="Minutes between cycles"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without Rich display"
    )
    parser.add_argument(
        "--buy-threshold", type=float, default=None, help="Override buy threshold"
    )
    parser.add_argument(
        "--sell-threshold", type=float, default=None, help="Override sell threshold"
    )
    parser.add_argument(
        "--asset-class",
        choices=["stocks", "crypto"],
        default="stocks",
        help="Asset class",
    )
    parser.add_argument(
        "--crypto-pairs",
        choices=["top10", "all17"],
        default="top10",
        help="Crypto pairs",
    )
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

    if not args.headless:
        console = Console(theme=_THEME)

    trader = PaperTrader(config)
    nyc = ZoneInfo("America/New_York")
    equity_history = []
    t0 = make_trade_table({}, {}, {}, {"equity": 0}, 0, 0, "", [])
    live = (
        Live(build_layout(t0), screen=True, refresh_per_second=4)
        if not args.headless
        else None
    )

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
                print(f"Market closed. Next open ~{wait_m} min")
                time.sleep(min(wait, 300))
                continue

            account = trader.get_account()
            equity_history.append(account.get("equity", 0))
            if len(equity_history) > 100:
                equity_history.pop(0)

            signals = run_inference(config, buy_threshold=buy_t, sell_threshold=sell_t)
            positions = trader.get_positions()
            trades = trader.reconcile(signals)

            if not args.headless and live:
                table = make_trade_table(
                    signals,
                    positions,
                    trades,
                    account,
                    cycle,
                    args.interval * 60,
                    now_str,
                    equity_history,
                )
                live.update(build_layout(table))
            else:
                n = len([t for t in trades if "FAIL" not in str(t[2])])
                print(
                    f"[{now_str}] Cycle #{cycle} | Equity: ${account.get('equity', 0):,.0f} | Trades: {n}"
                )

            time.sleep(args.interval * 60)

        except KeyboardInterrupt:
            if live:
                live.stop()
            break
        except Exception as e:
            if console:
                console.print(f"[error]Cycle error: {e}[/error]")
            time.sleep(30)


if __name__ == "__main__":
    main()
