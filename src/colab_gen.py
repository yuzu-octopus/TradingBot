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

    # Build pip install string from pyproject.toml so deps never go stale.
    import tomllib

    with Path("pyproject.toml").open("rb") as f:
        _toml = tomllib.load(f)
    _all_deps = _toml["project"]["dependencies"]
    # Filter out textual — only used by the local TUI (textual_trader.py),
    # not included in the Colab/Kaggle zip. Saves ~15 MB of transitive deps.
    _pip_deps = " ".join(d for d in _all_deps if not d.startswith("textual"))

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

    # The generated script needs to know where the model will be saved.
    # For crypto, main.py sets model_save_path to "data/models/crypto/best.pt".
    model_save_path = (
        "data/models/crypto/best.pt"
        if asset_class == "crypto"
        else "data/models/best.pt"
    )

    return f"""# TradingBot — Remote Training
# Works on both Colab and Kaggle.
# Colab: paste into a GPU cell and run.
# Kaggle: create a Notebook (GPU accelerator), paste into a cell, run.

import os, sys, warnings, shutil, time, zipfile, base64, io
from pathlib import Path

from tqdm import tqdm

warnings.filterwarnings("ignore")
start = time.time()

# --- Runtime environment detection ---
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

def _phase(name: str) -> None:
    el = time.time() - start  # phase banner with elapsed time
    print(f"\\n{"=" * 60}")
    print(f"  [{{el:4.0f}}s] {{name}}")
    print(f"{"=" * 60}")

_phase("Installing dependencies")
# Install ALL third-party deps from pyproject.toml (auto-generated).
# torch/pandas/numpy/sklearn are pre-installed on Colab GPU runtimes
# but pinning guarantees the versions we tested against.
if IS_KAGGLE:
    import subprocess as _sp
    _sp.run("pip install -q {_pip_deps}".split())
else:
    get_ipython().system("pip install -q {_pip_deps}")

import torch
device_count = torch.cuda.device_count()
device_name = torch.cuda.get_device_name(0) if device_count > 0 else "CPU"
print(f"CUDA: {{device_count > 0}} \u2014 {{device_count}} GPU(s) \u2014 {{device_name}}")

BASE = "/kaggle/working/tradingbot" if IS_KAGGLE else "/content/tradingbot"
ARCHIVE = "/kaggle/working/tradingbot_model" if IS_KAGGLE else "/content/tradingbot_model"
Path(BASE).mkdir(parents=True, exist_ok=True)

_phase("Extracting project files")
with zipfile.ZipFile(io.BytesIO(base64.b64decode({_safe_str(payload)}))) as z:
    members = z.infolist()
    for m in tqdm(members, desc="Extracting", unit="file", file=sys.stderr):
        z.extract(m, BASE)

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

{'sv_orig = list(sys.argv); pretrain_argv = list(sv_orig); pretrain_argv[pretrain_argv.index("--mode") + 1] = "pretrain"; _phase("Pre-training"); sys.argv = pretrain_argv; _run(); _phase("Fine-tuning"); sys.argv = sv_orig' if do_pretrain else ""}
_phase("Training")
_run()

elapsed = time.time() - start
_model_path = Path(BASE, {_safe_str(model_save_path)})
if _model_path.exists():
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
