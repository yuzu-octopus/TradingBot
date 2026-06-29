import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config, get_sp500_tickers, is_distributed
from src.crypto_pipeline import fetch_crypto_data
from src.data_pipeline import fetch_stock_data
from src.features import (
    build_feature_matrix,
    build_targets,
    compute_market_state,
    load_cached_features,
    save_cached_features,
)
from src.utils import load_threshold, setup_logger
from training.threshold import run_threshold_optimization
from training.train import run_training


def _fetch_data(config: Config) -> dict:
    if config.asset_class == "crypto":
        return fetch_crypto_data(
            config.tickers,
            config.train_start,
            config.test_end,
            config.raw_data_path,
        )
    return fetch_stock_data(
        config.tickers,
        config.train_start,
        config.test_end,
        config.raw_data_path,
    )


def _fold_metadata(config: Config) -> dict:
    """Fingerprint the config knobs that influence fold slicing + feature shape.

    Used to validate cached fold npz files on load — stale folds produce
    silently-wrong training runs otherwise.
    """
    return {
        "wf_window_size": config.wf_window_size,
        "wf_val_size": config.wf_val_size,
        "wf_test_size": config.wf_test_size,
        "wf_step_size": config.wf_step_size,
        "train_start": config.train_start,
        "test_end": config.test_end,
        "label_max_return": config.label_max_return,
        "n_features": config.n_features,
        "asset_class": config.asset_class,
        "crypto_pairs": config.crypto_pairs,
        "n_stocks": config.n_stocks,
    }


def _folds_match_config(config: Config, fold_dir: Path) -> bool:
    meta_path = fold_dir / "folds_meta.json"
    if not meta_path.exists():
        return False
    expected = _fold_metadata(config)
    actual = json.loads(meta_path.read_text())
    return actual == expected


def prepare_walk_forward_splits(
    features: np.ndarray,
    targets: np.ndarray,
    market_state: np.ndarray,
    dates: list[str],
    config: Config,
) -> int:
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
    fold_dir = Path(config.features_path)
    fold_dir.mkdir(parents=True, exist_ok=True)
    for i, (tr, va, te, _label) in enumerate(folds):
        np.savez(
            f"{fold_dir}/fold_{i}_train.npz",
            features=features[tr],
            targets=targets[tr],
            market_state=market_state[tr],
        )
        np.savez(
            f"{fold_dir}/fold_{i}_val.npz",
            features=features[va],
            targets=targets[va],
            market_state=market_state[va],
        )
        np.savez(
            f"{fold_dir}/fold_{i}_test.npz",
            features=features[te],
            targets=targets[te],
            market_state=market_state[te],
        )
    # Sidecar fingerprint so a future run can detect config changes that
    # would invalidate the cached fold slices.
    (fold_dir / "folds_meta.json").write_text(
        json.dumps(_fold_metadata(config), indent=2)
    )
    print(f"  Created {len(folds)} walk-forward folds")
    return len(folds)


def prepare_data(config: Config) -> int:
    print("\n=== Data Preparation ===")
    raw_data = _fetch_data(config)
    cached = load_cached_features(config.raw_data_path, cache_dir=config.features_path)
    if cached is not None:
        features, tickers, dates = cached
        print(f"  Loaded cached feature matrix: {features.shape}")
    else:
        features, tickers, dates = build_feature_matrix(raw_data)
        save_cached_features(
            features,
            tickers,
            dates,
            config.raw_data_path,
            cache_dir=config.features_path,
        )
    config.tickers = tickers
    print(
        f"  ({len(tickers)} stocks, {features.shape[2]} features, {features.shape[0]} dates)"
    )

    # Dynamic date split for crypto — crypto data starts later than stocks
    # and varies significantly across pairs. A percentage-based split from
    # the actual available dates is more robust than hardcoded date ranges.
    if config.asset_class == "crypto":
        n = len(dates)
        if n < 100:
            msg = f"Not enough crypto data ({n} dates, need >= 100)"
            raise ValueError(msg)
        t_end = int(n * 0.6)
        v_end = int(n * 0.8)
        config.train_start = dates[0]
        config.train_end = dates[t_end - 1]
        config.val_start = dates[t_end]
        config.val_end = dates[v_end - 1]
        config.test_start = dates[v_end]
        config.test_end = dates[-1]
        print(
            f"  Crypto split: train={dates[0]}..{dates[t_end - 1]}, val={dates[t_end]}..{dates[v_end - 1]}, test={dates[v_end]}..{dates[-1]}"
        )

    train_mask, val_mask, test_mask = _split_date_range(dates, config)
    targets = build_targets(raw_data, tickers, dates, config.label_max_return)

    market_state = compute_market_state(
        raw_data,
        dates,
        market_ticker="BTC/USD" if config.asset_class == "crypto" else "SPY",
    )

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

    from trade import run_trading_loop

    run_trading_loop(
        config,
        interval=args.trade_interval,
        headless=args.trade_headless,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
    )


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
    parser.add_argument(
        "--asset-class",
        choices=["stocks", "crypto"],
        default="stocks",
        help="Asset class to trade (default: stocks)",
    )
    parser.add_argument(
        "--crypto-pairs",
        choices=["top10", "all17"],
        default="top10",
        help="Number of crypto pairs (default: top10)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--tickers-file",
        type=str,
        default="",
        help="Path to a file with one ticker per line (overrides Config.tickers)",
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
    config.asset_class = args.asset_class
    config.crypto_pairs = args.crypto_pairs
    config.no_amp = args.no_amp
    config.tickers_file = args.tickers_file
    setup_logger()
    if config.asset_class == "crypto":
        from config import CRYPTO_PAIR_MAP

        config.tickers = CRYPTO_PAIR_MAP[config.crypto_pairs]
        config.raw_data_path = "data/crypto/raw"
        config.features_path = "data/crypto/features"
        config.model_save_path = "data/models/crypto/best.pt"
        # Crypto data starts later than stocks — most pairs have data from
        # 2021 onwards via Alpaca. The default 2015-2025 stock date ranges
        # produce 0 train + 0 val splits when applied to crypto.
        config.train_start = "2021-01-01"
        config.train_end = "2023-06-30"
        config.val_start = "2023-07-01"
        config.val_end = "2023-12-31"
        config.test_start = "2024-01-01"
        config.test_end = "2025-06-01"
        print(f"Loaded {len(config.tickers)} crypto pairs ({config.crypto_pairs})")
    if args.tickers_file:
        with Path(args.tickers_file).open() as f:
            config.tickers = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(config.tickers)} tickers from {args.tickers_file}")
    if args.model:
        config.model_save_path = f"data/models/{args.model}/best.pt"
    if not config.tickers:
        config.tickers = get_sp500_tickers()
        print(f"Loaded {len(config.tickers)} tickers from S&P 500")

    if not config.alpaca_paper:
        confirm = os.environ.get("ALPACA_LIVE_CONFIRM", "")
        if confirm != "true":
            print(
                "\n⚠ LIVE TRADING DETECTED (alpaca_paper=False).\n"
                "Set ALPACA_LIVE_CONFIRM=true in .env to confirm.\n"
            )
            return

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
        cached = load_cached_features(
            config.raw_data_path, cache_dir=config.features_path
        )
        if cached is not None:
            _, tickers, _ = cached
            config.tickers = tickers

    if args.mode == "train":
        if args.walk_forward and n_folds == 0:
            # Reuse cached folds only if their config fingerprint still matches
            # the current config; otherwise rebuild so we don't silently train
            # on stale windows.
            fold_dir = Path(config.features_path)
            existing_folds = sorted(fold_dir.glob("fold_*_train.npz"))
            if existing_folds and _folds_match_config(config, fold_dir):
                n_folds = len(existing_folds)
                print(f"  Reusing {n_folds} cached walk-forward folds")
            else:
                raw_data = _fetch_data(config)
                features, tickers, dates = build_feature_matrix(raw_data)
                config.tickers = tickers
                targets = build_targets(
                    raw_data, config.tickers, dates, config.label_max_return
                )
                market_state = compute_market_state(
                    raw_data,
                    dates,
                    market_ticker="BTC/USD"
                    if config.asset_class == "crypto"
                    else "SPY",
                )
                n_folds = prepare_walk_forward_splits(
                    features, targets, market_state, dates, config
                )

        pretrain_path = config.pretrain_weights_path if args.pretrain else None
        fold_count = n_folds if args.walk_forward else 1
        orig_save_path = config.model_save_path
        for fold in range(fold_count):
            if args.walk_forward:
                print(f"\n=== Walk-Forward Fold {fold + 1}/{fold_count} ===")
                fold_model_path = config.model_save_path.replace(
                    ".pt", f"_fold{fold}.pt"
                )
                config.model_save_path = fold_model_path
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
                config.model_save_path = orig_save_path
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

        if is_distributed():
            import torch.distributed as dist

            dist.barrier()

        # Walk-forward: promote best fold model to best.pt so
        # run_threshold_optimization can load it (it was never written
        # during the fold loop since model_save_path pointed to _fold{N}.pt).
        if args.walk_forward:
            import shutil

            best_fold_path = None
            for fold in range(fold_count):
                fp = Path(orig_save_path).with_name(f"best_fold{fold}.pt")
                if fp.exists():
                    best_fold_path = fp
            if best_fold_path:
                shutil.copy2(best_fold_path, orig_save_path)
                print(f"  Promoted {best_fold_path.name} to {orig_save_path}")

        print("\n=== Threshold Optimization ===")
        buy_t, sell_t = run_threshold_optimization(config)
        print(f"Optimal thresholds: buy > {buy_t:.2f}, sell < -{sell_t:.2f}")

        # Walk-forward: evaluate each fold model on its test split using
        # thresholds derived from the fold's VALIDATION set (not test).
        if args.walk_forward:
            import torch

            from src.utils import load_model as _load_model
            from training.threshold import optimize_threshold

            print("\n=== Walk-Forward Test Evaluation ===")
            fold_sharpes = []
            for fold in range(fold_count):
                test_path = Path(f"{config.features_path}/fold_{fold}_test.npz")
                val_path = Path(f"{config.features_path}/fold_{fold}_val.npz")
                fold_model_path = Path(orig_save_path).with_name(f"best_fold{fold}.pt")
                if not test_path.exists() or not fold_model_path.exists():
                    continue
                saved = config.model_save_path
                try:
                    config.model_save_path = str(fold_model_path)
                    model = _load_model(config)
                finally:
                    config.model_save_path = saved
                # Optimize thresholds on the fold's validation set.
                bt, st = 0.5, 0.5
                if val_path.exists():
                    with np.load(val_path) as vd:
                        bt, st = optimize_threshold(
                            config,
                            model,
                            vd["features"],
                            vd["targets"],
                            market_state=vd.get("market_state"),
                        )
                else:
                    print(
                        f"  Fold {fold}: val split missing, "
                        f"using default thresholds (buy>{bt:.2f}, sell<-{st:.2f})"
                    )
                with np.load(test_path) as td:
                    te_feat = td["features"]
                    te_targ = td["targets"]
                    te_mkt = td.get("market_state")
                device = next(model.parameters()).device
                scores = (
                    model(
                        torch.tensor(te_feat, dtype=torch.float32).to(device),
                        market_state=torch.tensor(te_mkt, dtype=torch.float32).to(
                            device
                        )
                        if te_mkt is not None
                        else None,
                    )
                    .cpu()
                    .numpy()
                )
                sig = np.where(scores > bt, 1, np.where(scores < -st, -1, 0))
                port = sig.mean(axis=1) * te_targ.mean(axis=1)
                s = float(port.mean() / (port.std() + 1e-8) * np.sqrt(252))
                fold_sharpes.append(s)
                print(
                    f"  Fold {fold}: test Sharpe={s:.4f} (buy>{bt:.2f}, sell<-{st:.2f})"
                )
            if fold_sharpes:
                print(
                    f"  Mean test Sharpe: {np.mean(fold_sharpes):.4f} "
                    f"± {np.std(fold_sharpes):.4f}"
                )

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
