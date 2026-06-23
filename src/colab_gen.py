import argparse
from pathlib import Path


def _safe_str(data: str) -> str:
    """Produce a safe Python string literal for embedding."""
    return repr(data)


def generate_colab_script(args: argparse.Namespace) -> str:
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

    lines = []
    for path, content in files.items():
        lines.append(f"_write({_safe_str(path)}, {_safe_str(content)})")
    writes = "\n".join(lines)

    flaglist = [
        "main.py",
        "--mode",
        "train",
        "--loss",
        loss,
        "--seeds",
        str(seeds),
        "--grad-accum",
        str(grad_accum),
    ] + (extra.strip().split() if extra else [])

    return f"""# TradingBot — Colab Training
# Paste ALL of this into ONE Colab cell (GPU runtime) and run.

import os, sys, warnings, shutil, time
from pathlib import Path
warnings.filterwarnings("ignore")
start = time.time()

import torch
print(f"CUDA: {{torch.cuda.is_available()}} — Device: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}}")

BASE = "/content/tradingbot"

def _write(path, content):
    full = Path(BASE, path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)

print("Writing project files...")
{writes}

os.chdir(BASE)
sys.path.insert(0, BASE)
sys.argv = {flaglist}

print(f"[{{time.time()-start:.0f}}s] Starting training...")
# Clear cached modules so updated files are loaded on re-run
for m in list(sys.modules):
    if m.startswith(("config", "models", "src", "training")):
        sys.modules.pop(m, None)
exec(open("main.py").read())

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
