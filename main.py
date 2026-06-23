import argparse
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import Config, get_sp500_tickers
from src.data_pipeline import fetch_stock_data
from src.features import (
    build_feature_matrix,
    build_targets,
    compute_market_state,
    load_cached_features,
    save_cached_features,
)
from src.inference import run_inference
from src.paper_trader import PaperTrader
from src.utils import load_threshold
from trade import build_layout, make_trade_table
from training.threshold import run_threshold_optimization
from training.train import run_training


def prepare_walk_forward_splits(features, targets, market_state, dates, config):
    if any(
        v <= 0
        for v in (
            config.wf_window_size,
            config.wf_val_size,
            config.wf_test_size,
        )
    ):
        msg = "wf_window_size, wf_val_size, and wf_test_size must be positive"
        raise ValueError(msg)
    date_objs = [pd.Timestamp(d) for d in dates]
    start = pd.Timestamp(config.train_start)
    end = pd.Timestamp(config.test_end)
    folds = []
    current = start
    while (
        current
        + pd.DateOffset(
            years=config.wf_window_size + config.wf_val_size + config.wf_test_size
        )
        <= end
    ):
        train_end = current + pd.DateOffset(years=config.wf_window_size)
        val_end = train_end + pd.DateOffset(years=config.wf_val_size)
        test_end = val_end + pd.DateOffset(years=config.wf_test_size)
        train_idx = np.array([current <= d <= train_end for d in date_objs])
        val_idx = np.array([train_end < d <= val_end for d in date_objs])
        test_idx = np.array([val_end < d <= test_end for d in date_objs])
        folds.append((train_idx, val_idx, test_idx, f"{current.year}-{test_end.year}"))
        current += pd.DateOffset(years=config.wf_step_size)
    Path(config.features_path).mkdir(parents=True, exist_ok=True)
    for i, (tr, va, _te, _label) in enumerate(folds):
        np.savez(
            f"{config.features_path}/fold_{i}_train.npz",
            features=features[tr],
            targets=targets[tr],
            market_state=market_state[tr],
        )
        np.savez(
            f"{config.features_path}/fold_{i}_val.npz",
            features=features[va],
            targets=targets[va],
            market_state=market_state[va],
        )
    print(f"  Created {len(folds)} walk-forward folds")
    return len(folds)


def prepare_data(config: Config) -> int:
    print("\n=== Data Preparation ===")
    raw_data = fetch_stock_data(
        config.tickers, config.train_start, config.test_end, config.raw_data_path
    )
    cached = load_cached_features(config.raw_data_path)
    if cached is not None:
        features, tickers, dates = cached
        print(f"  Loaded cached feature matrix: {features.shape}")
    else:
        features, tickers, dates = build_feature_matrix(raw_data)
        save_cached_features(features, tickers, dates, config.raw_data_path)
    config.tickers = tickers
    print(
        f"  ({len(tickers)} stocks, {features.shape[2]} features, {features.shape[0]} dates)"
    )

    train_mask, val_mask, test_mask = _split_date_range(dates, config)
    targets = build_targets(raw_data, tickers, dates, config.label_max_return)

    market_state = compute_market_state(raw_data, dates)

    Path(config.features_path).mkdir(parents=True, exist_ok=True)
    np.savez(
        f"{config.features_path}/train.npz",
        features=features[train_mask],
        targets=targets[train_mask],
        market_state=market_state[train_mask],
    )
    np.savez(
        f"{config.features_path}/val.npz",
        features=features[val_mask],
        targets=targets[val_mask],
        market_state=market_state[val_mask],
    )
    np.savez(
        f"{config.features_path}/test.npz",
        features=features[test_mask],
        targets=targets[test_mask],
        market_state=market_state[test_mask],
    )

    n_train, n_val, n_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
    print(
        f"Split: {n_train} train + {n_val} val + {n_test} test = {n_train + n_val + n_test} dates"
    )

    n_folds = prepare_walk_forward_splits(
        features, targets, market_state, dates, config
    )
    return n_folds


def _split_date_range(
    dates: list[str], config: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    date_objs = [pd.Timestamp(d) for d in dates]

    def _in_range(d, start, end):
        return start <= str(d.date()) <= end

    return (
        np.array(
            [_in_range(d, config.train_start, config.train_end) for d in date_objs]
        ),
        np.array([_in_range(d, config.val_start, config.val_end) for d in date_objs]),
        np.array([_in_range(d, config.test_start, config.test_end) for d in date_objs]),
    )


def print_signals(results: dict[str, dict]) -> None:
    print(f"\n{'Ticker':<8} {'Score':<8} {'Signal':<8}")
    print("-" * 24)
    for ticker, info in results.items():
        print(f"{ticker:<8} {info['score']:<8.4f} {info['signal']:<8}")


def run_paper_trading(config: Config, args: argparse.Namespace) -> None:
    config.trade_interval_minutes = args.trade_interval
    config.trade_buy_qty = args.trade_buy_qty
    config.trade_sell_qty = args.trade_sell_qty

    trader = PaperTrader(config)
    nyc = ZoneInfo("America/New_York")
    buy_t, sell_t = load_threshold(config)
    if args.buy_threshold is not None:
        buy_t = args.buy_threshold
    if args.sell_threshold is not None:
        sell_t = args.sell_threshold

    console = None
    if not args.trade_headless:
        from rich.console import Console
        from rich.theme import Theme

        console = Console(
            theme=Theme(
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
            signals = run_inference(config, buy_threshold=buy_t, sell_threshold=sell_t)
            positions = trader.get_positions()
            trades = trader.reconcile(signals)

            if not args.trade_headless and console:
                table = make_trade_table(
                    signals,
                    positions,
                    trades,
                    account,
                    cycle,
                    args.trade_interval * 60,
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

            time.sleep(args.trade_interval * 60)

        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"Cycle error: {e}")
            time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Loss functions:\n"
            "  mse      Mean Squared Error — per-stock return prediction. Good baseline.\n"
            "  msrr     Max Sharpe Ratio Regression — directly optimizes portfolio Sharpe.\n"
            "           Noisier gradients; use --grad-accum >= 4. Avg SDF Sharpe 2.05.\n"
            "  margin   Pairwise ranking loss — encourages correct relative ordering\n"
            "           of stocks by return. Lower LR (50%% of base).\n"
            "  listnet  Listwise ranking loss — optimizes top-1 probability distribution.\n"
            "           Lower LR (30%% of base). Good risk-adjusted returns.\n"
            "\n"
            "Training:\n"
            "  --walk-forward: Splits data into multiple chronological windows (train/val/test),\n"
            "  trains on each, averages results. Gold standard for financial ML.\n"
            "\n"
            "  --seeds N: Trains N models with different random seeds, averages predictions.\n"
            "  Reduces variance. Recommended: 5-10 for MSRR, 1-3 for MSE.\n"
            "\n"
            "  --grad-accum N: Accumulate gradients over N batches before updating weights.\n"
            "  Simulates Nx larger batch without Nx memory. Stabilizes noisy gradients.\n"
            "  Recommended: 4 for MSRR/margin/listnet, 1 for MSE.\n"
            "\n"
            "  --resume: Load last checkpoint and continue training from where it stopped.\n"
            "\n"
            "Data:\n"
            "  First run auto-downloads data. Cached in data/stocks/.\n"
            "  Features cached in data/features/ after first build (~30 min).\n"
            "  Use --force-features to rebuild if you change tickers or date ranges.\n"
            "\n"
            "Colab:\n"
            "  --colab-template: Generate a complete Colab training script embedded with\n"
            "  all source code. Paste the output into a Colab GPU runtime to train there.\n"
            "  After training, download the model zip and place in data/models/top/.\n"
            "\n"
            "Examples:\n"
            "  uv run python main.py --mode train\n"
            "  uv run python main.py --mode train --loss msrr --seeds 5 --grad-accum 4\n"
            "  uv run python main.py --mode train --walk-forward\n"
            "  uv run python main.py --mode infer\n"
            "  uv run python main.py --mode train --resume\n"
            "  uv run python main.py --mode train --loss margin --grad-accum 4"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "pretrain", "trade"],
        default="train",
        help="train = train model + optimize threshold | infer = trading signals | pretrain = D6 pre-training | trade = Alpaca paper trading loop",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint (epoch, optimizer, scheduler restored)",
    )
    parser.add_argument(
        "--loss",
        choices=["mse", "msrr", "margin", "listnet"],
        default="mse",
        help="Loss function (see below for details)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of ensemble seeds (train N models with different random seeds, average predictions)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (accumulate N batches before optimizer step. Use 4 for MSRR/margin/listnet)",
    )
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Ignore cached features, rebuild from raw stock data",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Use walk-forward validation: sliding chronological train/val/test windows",
    )
    parser.add_argument(
        "--colab-template",
        action="store_true",
        help="Generate a self-contained Colab training script and copy to clipboard (does not train locally)",
    )
    parser.add_argument(
        "--show-script",
        action="store_true",
        help="When used with --colab-template, also print the full script to terminal",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Load model from data/models/<path>/best.pt (e.g. 'colab/run1' or 'top/run1')",
    )
    parser.add_argument(
        "--trade-interval",
        type=int,
        default=15,
        help="Minutes between trading cycles (default: 15)",
    )
    parser.add_argument(
        "--trade-headless",
        action="store_true",
        help="Run paper trading without Rich display",
    )
    parser.add_argument(
        "--trade-buy-qty",
        type=int,
        default=10,
        help="Shares to buy per long signal (default: 10)",
    )
    parser.add_argument(
        "--trade-sell-qty",
        type=int,
        default=10,
        help="Shares to sell per short signal (default: 10)",
    )
    parser.add_argument(
        "--buy-threshold",
        type=float,
        default=None,
        help="Override buy threshold",
    )
    parser.add_argument(
        "--sell-threshold",
        type=float,
        default=None,
        help="Override sell threshold",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Initialize training from pre-trained weights (data/models/pretrain/best.pt)",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=None,
        help="Override pretrain_epochs from config (default: 100)",
    )
    args = parser.parse_args()

    if args.colab_template:
        from src.colab_gen import generate_colab_script

        script = generate_colab_script(args)
        try:
            import pyperclip

            pyperclip.copy(script)
            print("Colab script copied to clipboard!")
        except Exception:
            pass
        if args.show_script:
            print("\n--- Colab Notebook Script ---")
            print(script)
            print("\n--- Paste into a Colab GPU runtime ---")
        return

    config = Config()
    if args.model:
        config.model_save_path = f"data/models/{args.model}/best.pt"
    if not config.tickers:
        config.tickers = get_sp500_tickers()
        print(f"Loaded {len(config.tickers)} tickers from S&P 500")

    has_features = Path(f"{config.features_path}/train.npz").exists()
    n_folds = 0
    if args.force_features or not has_features:
        if args.force_features:
            cache_dir = Path(config.features_path)
            if cache_dir.exists():
                for p in cache_dir.glob("*"):
                    p.unlink()
        n_folds = prepare_data(config)
    elif has_features:
        cached = load_cached_features(config.raw_data_path)
        if cached is not None:
            _, tickers, _ = cached
            config.tickers = tickers

    if args.mode == "train":
        if args.walk_forward and n_folds <= 1 and n_folds == 0:
            raw_data = fetch_stock_data(
                config.tickers,
                config.train_start,
                config.test_end,
                config.raw_data_path,
            )
            features, tickers, dates = build_feature_matrix(raw_data)
            config.tickers = tickers
            targets = build_targets(
                raw_data, config.tickers, dates, config.label_max_return
            )
            market_state = compute_market_state(raw_data, dates)
            n_folds = prepare_walk_forward_splits(
                features, targets, market_state, dates, config
            )

        pretrain_path = config.pretrain_weights_path if args.pretrain else None
        fold_count = n_folds if args.walk_forward else 1
        for fold in range(fold_count):
            if args.walk_forward:
                print(f"\n=== Walk-Forward Fold {fold + 1}/{fold_count} ===")
                rt_args = {
                    "config": config,
                    "resume": args.resume,
                    "loss_mode": args.loss,
                    "n_seeds": args.seeds,
                    "grad_accum_steps": args.grad_accum,
                    "train_path": f"{config.features_path}/fold_{fold}_train.npz",
                    "val_path": f"{config.features_path}/fold_{fold}_val.npz",
                    "pretrain_path": pretrain_path,
                }
                from training.train import run_training as rt

                rt(**rt_args)
            else:
                print(
                    f"\n=== Training (loss={args.loss}, seeds={args.seeds}, grad_accum={args.grad_accum}) ==="
                )
                run_training(
                    config,
                    resume=args.resume,
                    loss_mode=args.loss,
                    n_seeds=args.seeds,
                    grad_accum_steps=args.grad_accum,
                    pretrain_path=pretrain_path,
                )

        print("\n=== Threshold Optimization ===")
        buy_t, sell_t = run_threshold_optimization(config)
        print(f"Optimal thresholds: buy > {buy_t:.2f}, sell < -{sell_t:.2f}")

    elif args.mode == "pretrain":
        print("\n=== D6 Pre-Training ===")
        if args.pretrain_epochs is not None:
            config.pretrain_epochs = args.pretrain_epochs
        with np.load(f"{config.features_path}/train.npz") as data:
            pt_features = data["features"]
            pt_targets = data["targets"]
            pt_market = data.get("market_state")
        from training.pretrain import pretrain

        pretrain(
            config,
            pt_features,
            pt_targets,
            pt_market,
            loss_mode=args.loss,
            resume=args.resume,
            grad_accum_steps=args.grad_accum,
        )

    elif args.mode == "trade":
        print("\n=== Paper Trading ===")
        run_paper_trading(config, args)

    else:
        print("\n=== Inference ===")
        from src.inference import run_inference

        buy_t, sell_t = load_threshold(config)
        results = run_inference(config, buy_threshold=buy_t, sell_threshold=sell_t)
        print_signals(results)


if __name__ == "__main__":
    main()
