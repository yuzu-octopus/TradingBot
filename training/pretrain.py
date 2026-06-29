import itertools
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import cross_entropy, mse_loss
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from tqdm import tqdm

from config import Config, get_device, get_rank, is_distributed
from src.utils import create_model, unwrap_model, wrap_ddp
from training.train import listnet_loss, margin_ranking_loss, portfolio_mse_loss

PRETRAIN_CHECKPOINT_PATH = "data/models/pretrain/checkpoint.pt"


def _csr_loss(pred, target, loss_mode):
    if loss_mode == "msrr":
        return portfolio_mse_loss(pred, target)
    if loss_mode == "margin":
        return margin_ranking_loss(pred, target)
    if loss_mode == "listnet":
        return listnet_loss(pred, target)
    return mse_loss(pred, target)


class TemporalOrderHead(nn.Module):
    def __init__(self, n_days: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(n_days, n_classes)

    def forward(self, pooled_scores: torch.Tensor) -> torch.Tensor:
        return self.fc(pooled_scores)


def prepare_mpp(features, targets, mask_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    T, S, _ = features.shape
    mask = rng.rand(T, S) < mask_ratio
    masked = features.copy()
    masked[mask] = 0.0
    return masked, targets, mask


def prepare_top(features, n_days=3, seed=42):
    rng = np.random.RandomState(seed)
    T, _S, _F = features.shape
    perms = list(itertools.permutations(range(n_days)))
    perm_to_label = {p: i for i, p in enumerate(perms)}
    windows, labels = [], []
    for t in range(T - n_days + 1):
        window = features[t : t + n_days].copy()
        perm = list(range(n_days))
        rng.shuffle(perm)
        shuffled = window[perm]
        windows.append(shuffled)
        inverse = tuple(np.argsort(perm))
        labels.append(perm_to_label[inverse])
    return np.stack(windows), np.array(labels), len(perms)


def mpp_loss(pred, target, mask):
    err = (pred - target) ** 2
    return (err * mask.float()).sum() / (mask.sum() + 1e-8)


def top_loss(logits, labels):
    return cross_entropy(logits, labels)


def pretrain(
    config: Config,
    features: np.ndarray,
    targets: np.ndarray,
    market_state: np.ndarray | None = None,
    loss_mode: str = "mse",
    resume: bool = False,
    grad_accum_steps: int = 1,
) -> nn.Module:
    device = get_device()
    model = create_model(config, device)
    top_head: nn.Module = TemporalOrderHead(
        config.pretrain_top_n_days, math.factorial(config.pretrain_top_n_days)
    ).to(device)
    top_head = wrap_ddp(top_head, device)

    mkt = market_state if market_state is not None else np.zeros((len(features), 5))

    # Guard against empty or insufficient training data BEFORE any tensor ops.
    # prepare_top requires at least pretrain_top_n_days rows; fewer rows
    # produce an empty iterable that crashes np.stack with "need at least
    # one array to stack".
    min_dates = max(2, config.pretrain_top_n_days)
    for name, arr in [("features", features), ("targets", targets)]:
        if arr.shape[0] == 0:
            msg = (
                f"No training data available — {name} array is empty. "
                "This usually means the configured date ranges don't overlap "
                "with the available data. Try a wider date range or fewer assets."
            )
            raise ValueError(msg)
        if arr.shape[0] < min_dates:
            msg = (
                f"Not enough training data — {name} has {arr.shape[0]} rows "
                f"but need at least {min_dates}. The data split may be too "
                "aggressive for this asset class."
            )
            raise ValueError(msg)

    mpp_x, mpp_y, mpp_mask = prepare_mpp(features, targets, config.pretrain_mask_ratio)
    top_x, top_y, _top_nc = prepare_top(features, config.pretrain_top_n_days)

    mkt_t = torch.tensor(mkt, dtype=torch.float32)

    # Wrap arrays as TensorDatasets so DistributedSampler accepts them (its signature
    # rejects raw tensors/ndarrays) and so save_/load_ is symmetric.
    mpp_dataset = TensorDataset(
        torch.tensor(mpp_x, dtype=torch.float32),
        torch.tensor(mpp_y, dtype=torch.float32),
        torch.tensor(mpp_mask, dtype=torch.bool),
        mkt_t,
    )
    top_dataset = TensorDataset(
        torch.tensor(top_x, dtype=torch.float32),
        torch.tensor(top_y, dtype=torch.long),
    )
    csr_dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        mkt_t,
    )

    mpp_sampler: DistributedSampler | None = (
        DistributedSampler(mpp_dataset, shuffle=True) if is_distributed() else None
    )
    top_sampler: DistributedSampler | None = (
        DistributedSampler(top_dataset, shuffle=True) if is_distributed() else None
    )
    csr_sampler: DistributedSampler | None = (
        DistributedSampler(csr_dataset, shuffle=True) if is_distributed() else None
    )

    mpp_loader = DataLoader(
        mpp_dataset,
        batch_size=config.batch_size,
        sampler=mpp_sampler,
        shuffle=mpp_sampler is None,
        drop_last=True,
    )
    top_loader = DataLoader(
        top_dataset,
        batch_size=config.batch_size,
        sampler=top_sampler,
        shuffle=top_sampler is None,
        drop_last=True,
    )
    csr_loader = DataLoader(
        csr_dataset,
        batch_size=config.batch_size,
        sampler=csr_sampler,
        shuffle=csr_sampler is None,
        drop_last=True,
    )

    all_params = list(unwrap_model(model).parameters()) + list(
        unwrap_model(top_head).parameters()
    )
    optimizer = optim.AdamW(
        all_params, lr=config.pretrain_lr, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.pretrain_epochs
    )
    use_amp = device.type in ("cuda", "mps") and not config.no_amp
    amp_scaler = (
        torch.amp.GradScaler(device.type) if use_amp and device.type == "cuda" else None
    )

    resume_epoch = 0
    best_loss = float("inf")
    patience_counter = 0
    if resume and Path(PRETRAIN_CHECKPOINT_PATH).exists():
        ckpt = torch.load(
            PRETRAIN_CHECKPOINT_PATH, weights_only=True, map_location=device
        )
        unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
        unwrap_model(top_head).load_state_dict(ckpt["top_head_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        resume_epoch = ckpt["epoch"]
        best_loss = ckpt["best_loss"]
        patience_counter = ckpt.get("patience_counter", 0)
        tqdm.write(f"  Resumed from epoch {resume_epoch} (best_loss={best_loss:.6f})")

    Path(config.pretrain_weights_path).parent.mkdir(parents=True, exist_ok=True)
    epoch_bar = tqdm(
        range(resume_epoch, config.pretrain_epochs),
        desc="Pretraining",
        unit="epoch",
        file=sys.stderr,
        initial=resume_epoch,
        total=config.pretrain_epochs,
    )

    for epoch in epoch_bar:
        if mpp_sampler is not None:
            mpp_sampler.set_epoch(epoch)
        if top_sampler is not None:
            top_sampler.set_epoch(epoch)
        if csr_sampler is not None:
            csr_sampler.set_epoch(epoch)
        model.train()
        top_head.train()
        total_loss = 0.0
        mpp_iter = iter(mpp_loader)
        top_iter = iter(top_loader)
        csr_iter = iter(csr_loader)
        n_batches = min(len(mpp_loader), len(top_loader), len(csr_loader))
        optimizer.zero_grad()

        for step in range(n_batches):
            x_mpp, y_mpp, mask_mpp, m_mpp = [t.to(device) for t in next(mpp_iter)]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred_mpp = model(x_mpp, market_state=m_mpp)
                l_mpp = mpp_loss(pred_mpp, y_mpp, mask_mpp)

            x_top, y_top = next(top_iter)
            x_top, y_top = x_top.to(device), y_top.to(device)
            B, N, S, F = x_top.shape
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred_top = model(x_top.reshape(B * N, S, F))
                pooled = pred_top.mean(dim=1).reshape(B, N)
                logits = top_head(pooled)
                l_top = top_loss(logits, y_top)

            x_csr, y_csr, m_csr = [t.to(device) for t in next(csr_iter)]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred_csr = model(x_csr, market_state=m_csr)
                l_csr = _csr_loss(pred_csr, y_csr, loss_mode)

            loss = (l_mpp + 0.5 * l_top + l_csr) / 3 / max(grad_accum_steps, 1)

            if use_amp and amp_scaler is not None:
                amp_scaler.scale(loss).backward()
            else:
                loss.backward()

            steps_since_update = (step + 1) % max(grad_accum_steps, 1)
            if steps_since_update == 0 or step == n_batches - 1:
                if use_amp and amp_scaler is not None:
                    amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, config.max_grad_norm)
                if use_amp and amp_scaler is not None:
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * max(grad_accum_steps, 1)

        scheduler.step()
        n_accum_steps = max(1, (n_batches + grad_accum_steps - 1) // grad_accum_steps)
        avg_loss = total_loss / n_accum_steps

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            if get_rank() == 0:
                torch.save(
                    unwrap_model(model).state_dict(), config.pretrain_weights_path
                )
        else:
            patience_counter += 1
            if patience_counter >= config.pretrain_early_stop_patience:
                tqdm.write(
                    f"  Pretrain early stopping at epoch {epoch + 1} "
                    f"(no improvement for {patience_counter} epochs)"
                )
                break

        if (epoch + 1) % 10 == 0 and get_rank() == 0:
            torch.save(
                {
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "top_head_state_dict": unwrap_model(top_head).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_loss": best_loss,
                    "patience_counter": patience_counter,
                },
                PRETRAIN_CHECKPOINT_PATH,
            )

        epoch_bar.set_postfix(loss=f"{avg_loss:.6f}")
        if not math.isfinite(avg_loss):
            tqdm.write(
                "  Pretrain loss diverged — NaN/inf detected. "
                "The dataset may be too small for effective pretraining. "
                "Fine-tuning proceeds from random init."
            )
            break

    return model
