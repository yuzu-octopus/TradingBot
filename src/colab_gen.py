import argparse
import base64
import io
import zipfile
from pathlib import Path


def _safe_str(data: str) -> str:
    return repr(data)


def _build_zip(files: dict[str, str]) -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for path, content in files.items():
            z.writestr(path, content)
    return base64.b64encode(buf.getvalue()).decode()


def _redact_secrets(content: str) -> str:
    """Replace known secret field values with REDACTED for the generated script."""
    import re

    return re.sub(
        r'(alpaca_api_key|alpaca_secret_key)\s*[=:].*?["\']([^"\']*)["\']',
        r'\1 = "REDACTED"',
        content,
    )


def generate_colab_script(args: argparse.Namespace) -> str:
    mode = args.mode
    loss = args.loss
    seeds = args.seeds
    grad_accum = args.grad_accum
    asset_class = args.asset_class
    crypto_pairs = args.crypto_pairs
    extra = ""
    if args.walk_forward:
        extra += " --walk-forward"
    if args.force_features:
        extra += " --force-features"
    if args.resume:
        extra += " --resume"
    if args.pretrain:
        extra += " --pretrain"
    if args.no_amp:
        extra += " --no-amp"
    if asset_class != "stocks":
        extra += f" --asset-class {asset_class}"
    if crypto_pairs != "top10":
        extra += f" --crypto-pairs {crypto_pairs}"

    files = {}
    for p in [
        "config.py",
        "main.py",
        "models/stock_model.py",
        "src/crypto_pipeline.py",
        "src/data_pipeline.py",
        "src/features.py",
        "src/paper_trader.py",
        "src/utils.py",
        "src/inference.py",
        "training/train.py",
        "training/threshold.py",
        "training/pretrain.py",
        "trade.py",
    ]:
        files[p] = _redact_secrets(Path(p).read_text())
    for p in ["models/__init__.py", "src/__init__.py", "training/__init__.py"]:
        files[p] = ""

    payload = _build_zip(files)

    flaglist = [
        "main.py",
        "--mode",
        mode,
        "--loss",
        loss,
        "--seeds",
        str(seeds),
        "--grad-accum",
        str(grad_accum),
    ] + (extra.strip().split() if extra else [])

    do_pretrain = " --pretrain" in extra

    return f"""# TradingBot — Remote Training
# Works on both Colab and Kaggle.
# Colab: paste into a GPU cell and run.
# Kaggle: create a Notebook (GPU accelerator), paste into a cell, run.

import os, sys, warnings, shutil, time, zipfile, base64, io
from pathlib import Path
warnings.filterwarnings("ignore")
start = time.time()

# --- Runtime environment detection ---
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

print("Installing dependencies...")
# Install ALL third-party deps from pyproject.toml. Even though main.py
# lazy-imports PaperTrader/trade, src.crypto_pipeline still imports
# alpaca-py at module load (it sits in the import chain). torch/pandas/etc.
# are pre-installed on Colab GPU runtimes but pinning them guarantees the
# versions we tested against.
if IS_KAGGLE:
    import subprocess as _sp
    _sp.run("pip install -q unlockedpd>=0.3.0 yfinance>=1.4.1 lxml>=6.1.1 alpaca-py>=0.43.0 torch numpy pandas scikit-learn tqdm rich".split())
else:
    get_ipython().system("pip install -q unlockedpd>=0.3.0 yfinance>=1.4.1 lxml>=6.1.1 alpaca-py>=0.43.0 torch numpy pandas scikit-learn tqdm rich")

import torch
device_count = torch.cuda.device_count()
device_name = torch.cuda.get_device_name(0) if device_count > 0 else "CPU"
print(f"CUDA: {{device_count > 0}} — {{device_count}} GPU(s) — {{device_name}}")

BASE = "/kaggle/working/tradingbot" if IS_KAGGLE else "/content/tradingbot"
ARCHIVE = "/kaggle/working/tradingbot_model" if IS_KAGGLE else "/content/tradingbot_model"
Path(BASE).mkdir(parents=True, exist_ok=True)

print("Extracting project files...")
with zipfile.ZipFile(io.BytesIO(base64.b64decode({_safe_str(payload)}))) as z:
    z.extractall(BASE)

os.chdir(BASE)
sys.path.insert(0, BASE)
sys.argv = {flaglist}

def _run():
    for m in list(sys.modules):
        if m.startswith(("config", "models", "src", "training")):
            sys.modules.pop(m, None)
    try:
        exec(open("main.py").read(), globals())
    except Exception:
        import traceback
        traceback.print_exc()
        print("\\n[FATAL] exec failed \\u2014 see traceback above")
        raise

{'sv_orig = list(sys.argv); pretrain_argv = list(sv_orig); pretrain_argv[pretrain_argv.index("--mode") + 1] = "pretrain"; sys.argv = pretrain_argv; print(f"[{time.time()-start:.0f}s] Pre-training..."); _run(); print(f"[{time.time()-start:.0f}s] Pre-training done. Fine-tuning..."); sys.argv = sv_orig' if do_pretrain else ""}
_run()

elapsed = time.time() - start
if Path(BASE, "data/models/best.pt").exists():
    shutil.make_archive(ARCHIVE, "zip", BASE + "/data/models")
    print(f"\\n[{{elapsed:.0f}}s] Training complete.")
    if IS_KAGGLE:
        print(f"Model saved to {{BASE}}/data/models/best.pt")
        print("Download via Kaggle UI: right sidebar \\u2192 Output")
    else:
        from google.colab import files
        files.download(ARCHIVE + ".zip")
    print("Extract best.pt to data/models/colab/<name>/ and run --model colab/<name>")
else:
    print("\\n[WARN] best.pt not found")
"""
