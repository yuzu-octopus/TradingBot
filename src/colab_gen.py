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


def generate_colab_script(args: argparse.Namespace) -> str:
    mode = args.mode
    loss = args.loss
    seeds = args.seeds
    grad_accum = args.grad_accum
    extra = ""
    if args.walk_forward:
        extra += " --walk-forward"
    if args.force_features:
        extra += " --force-features"
    if args.resume:
        extra += " --resume"
    if args.pretrain:
        extra += " --pretrain"

    files = {}
    for p in [
        "config.py",
        "main.py",
        "models/stock_model.py",
        "src/data_pipeline.py",
        "src/features.py",
        "src/utils.py",
        "src/inference.py",
        "training/train.py",
        "training/threshold.py",
        "training/pretrain.py",
    ]:
        files[p] = Path(p).read_text()
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

    return f"""# TradingBot — Colab Training
# Paste ALL of this into ONE Colab cell (GPU runtime) and run.

import os, sys, warnings, shutil, time, zipfile, base64, io
from pathlib import Path
warnings.filterwarnings("ignore")
start = time.time()

print("Installing dependencies...")
get_ipython().system("pip install -q unlockedpd>=0.3.0 yfinance>=1.4.1 pyperclip>=1.11.0 lxml>=6.1.1")

import torch
print(f"CUDA: {{torch.cuda.is_available()}} — Device: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}}")

BASE = "/content/tradingbot"
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
        print("\n[FATAL] exec failed — see traceback above")
        raise

{'print(f"[{time.time()-start:.0f}s] Pre-training step..."); sv = sys.argv; sys.argv = [a for a in sv if a != "--pretrain"]; _run(); print(f"[{time.time()-start:.0f}s] Pre-training done. Starting fine-tune..."); sys.argv = sv' if do_pretrain else ""}
_run()

elapsed = time.time() - start
if Path(BASE, "data/models/best.pt").exists():
    shutil.make_archive("/content/tradingbot_model", "zip", BASE + "/data/models")
    print(f"\\n[{{elapsed:.0f}}s] Downloading...")
    from google.colab import files
    files.download("/content/tradingbot_model.zip")
    print("Extract best.pt to data/models/colab/<name>/ and run --model colab/<name>")
else:
    print("\\n[WARN] best.pt not found")
"""
