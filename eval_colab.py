#!/usr/bin/env python3
"""Evaluate Colab-trained model zips and promote the best to data/models/best.pt."""

import argparse
import csv
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

DOWNLOAD_DIR = Path.home() / "Downloads"
MODELS_DIR = Path("data/models")
COLAB_DIR = MODELS_DIR / "colab"
EVAL_LOG = MODELS_DIR / "eval_log.csv"


def find_latest_zip(directory: Path) -> Path | None:
    zips = sorted(
        directory.glob("tradingbot_model*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return zips[0] if zips else None


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    import zipfile

    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)
    print(f"  Extracted to {target_dir}")


def find_seed_models(model_dir: Path) -> list[Path]:
    return sorted(model_dir.glob("best_seed*.pt"))


def evaluate_model(seed_path: Path) -> float | None:
    import torch

    from config import Config
    from src.utils import create_model, load_scaler, scale_features, unwrap_model
    from training.threshold import optimize_threshold

    test_path = Path("data/features/test.npz")
    val_path = Path("data/features/val.npz")
    scaler_path = Path("data/features/scaler.json")
    if not test_path.exists() or not val_path.exists() or not scaler_path.exists():
        return None

    with np.load(val_path) as f:
        val_features = f["features"]
        val_targets = f["targets"]
    with np.load(test_path) as f:
        test_features = f["features"]
        test_targets = f["targets"]
        test_market = f.get("market_state")

    cfg = Config()
    cfg.model_save_path = str(seed_path)
    from config import set_n_stocks

    set_n_stocks(cfg, test_features.shape[1])

    try:
        from config import get_device

        device = get_device()
        model = create_model(cfg, device)
        state = torch.load(seed_path, weights_only=True, map_location=device)
        unwrap_model(model).load_state_dict(state)
        model.eval()
    except Exception as e:
        print(f"load fail: {e}", end="")
        return None

    scaler = load_scaler(str(scaler_path))
    scaled_val = scale_features(val_features, scaler)
    scaled_test = scale_features(test_features, scaler)
    test_t = torch.tensor(scaled_test, dtype=torch.float32)
    market_t = (
        torch.tensor(test_market, dtype=torch.float32)
        if test_market is not None
        else None
    )

    try:
        with torch.no_grad():
            if market_t is not None:
                scores = (
                    model(test_t.to(device), market_state=market_t.to(device))
                    .cpu()
                    .numpy()
                )
            else:
                scores = model(test_t.to(device)).cpu().numpy()
    except Exception as e:
        print(f"infer fail: {e}", end="")
        return None

    try:
        buy_t, sell_t = optimize_threshold(cfg, model, scaled_val, val_targets)
        signals = np.where(scores > buy_t, 1, np.where(scores < -sell_t, -1, 0))
        daily_ret = test_targets.mean(axis=1)
        port_ret = signals.mean(axis=1) * daily_ret
        sharpe = float(np.mean(port_ret) / (np.std(port_ret) + 1e-8) * np.sqrt(252))
        return sharpe  # noqa: TRY300
    except Exception as e:
        print(f"threshold fail: {e}", end="")
        return None


def log_result(
    timestamp: str, source: str, sharpe: float | None, promoted: bool, path: str
) -> None:
    header = not EVAL_LOG.exists()
    with EVAL_LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["timestamp", "source", "sharpe", "promoted", "path"])
        w.writerow(
            [
                timestamp,
                source,
                f"{sharpe:.4f}" if sharpe is not None else "FAIL",
                "PROMOTED" if promoted else "",
                path,
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and promote Colab-trained models"
    )
    parser.add_argument(
        "zip_path",
        nargs="?",
        type=str,
        default=None,
        help="Path to tradingbot_model.zip (default: auto-detect in ~/Downloads)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    args = parser.parse_args()

    if args.zip_path:
        zip_path = Path(args.zip_path)
        if not zip_path.exists():
            print(f"Error: {zip_path} not found")
            return
    else:
        z = find_latest_zip(DOWNLOAD_DIR)
        if z is None:
            print(f"No tradingbot_model*.zip found in {DOWNLOAD_DIR}")
            return
        zip_path = z

    print(f"Found: {zip_path}")
    if args.dry_run:
        print("[DRY RUN] Would extract, evaluate, and promote")
        return

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    extract_dir = COLAB_DIR / timestamp
    extract_zip(zip_path, extract_dir)
    zip_path.unlink()
    print("  Deleted zip")

    candidates = [extract_dir / "best.pt", *find_seed_models(extract_dir)]
    if not candidates:
        print("  No model files found")
        return

    print(f"  Evaluating {len(candidates)} model(s)...")
    results = []
    for seed_path in candidates:
        name = seed_path.name
        print(f"  {name}...", end=" ", flush=True)
        s = evaluate_model(seed_path)
        results.append((name, s))
        print(f"  {s:.4f}" if s is not None else "  FAIL")

    valid = [(n, s) for n, s in results if s is not None]
    if not valid:
        print("  No models evaluated successfully")
        return

    valid.sort(key=lambda x: x[1], reverse=True)
    best_name, best_sharpe = valid[0]
    best_path = extract_dir / best_name
    if best_path.exists():
        shutil.copy2(best_path, MODELS_DIR / "best.pt")
        print(f"\n  PROMOTED: {best_name} (Sharpe {best_sharpe:.4f})")

    ts = datetime.now(UTC).strftime("%Y-%m-%d_%H:%M:%S")
    for n, s in results:
        log_result(ts, n, s, promoted=(n == best_name), path=str(extract_dir / n))

    print(f"\n  {'Name':<20} {'Sharpe':<10} {'Status':<12}")
    print(f"  {'-' * 42}")
    for n, s in results:
        status = "PROMOTED" if n == best_name else ("FAIL" if s is None else "")
        ss = f"{s:.4f}" if s is not None else "FAIL"
        print(f"  {n:<20} {ss:<10} {status:<12}")


if __name__ == "__main__":
    main()
