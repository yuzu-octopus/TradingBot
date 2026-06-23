import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from tqdm import tqdm

from config import Config, get_device, get_rank, is_distributed
from src.utils import create_model, save_scaler, scale_features, unwrap_model

CHECKPOINT_PATH = "data/models/checkpoint.pt"


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_loss: float,
    patience_counter: int,
    path: str = CHECKPOINT_PATH,
) -> None:
    """Save training state. Under DDP, only rank 0 writes to avoid file races."""
    if is_distributed() and get_rank() != 0:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
    path: str = CHECKPOINT_PATH,
) -> tuple[int, float, int]:
    ckpt = torch.load(path, weights_only=True, map_location=device)
    unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["best_val_loss"], ckpt["patience_counter"]


def portfolio_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalizes portfolio return deviation from 1.0.

    Heuristic approximation: encourages prediction*return correlation.
    Does NOT directly optimize Sharpe ratio (variance is ignored).
    """
    portfolio_return = (pred * target).sum(dim=1)
    return ((1 - portfolio_return) ** 2).mean()


def margin_ranking_loss(
    pred: torch.Tensor, target: torch.Tensor, margin: float = 0.1
) -> torch.Tensor:
    n = pred.size(1)
    pred_i = pred.unsqueeze(2).expand(-1, n, n)
    pred_j = pred.unsqueeze(1).expand(-1, n, n)
    target_i = target.unsqueeze(2).expand(-1, n, n)
    target_j = target.unsqueeze(1).expand(-1, n, n)
    pairwise_diff = pred_i - pred_j
    target_order = torch.sign(target_i - target_j)
    loss = torch.clamp(margin - pairwise_diff * target_order, min=0)
    mask = target_order.abs() > 1e-6
    return loss[mask].mean()


def listnet_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_softmax = torch.softmax(pred, dim=1)
    target_softmax = torch.softmax(target / 0.1, dim=1)
    return -(target_softmax * torch.log(pred_softmax + 1e-8)).sum(dim=1).mean()


def train(
    config: Config,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    loss_mode: str = "mse",
    resume_epoch: int = 0,
    resume_best_loss: float = float("inf"),
    resume_patience: int = 0,
    grad_accum_steps: int = 1,
    train_market: np.ndarray | None = None,
    val_market: np.ndarray | None = None,
    pretrain_path: str | None = None,
    checkpoint_path: str = CHECKPOINT_PATH,
) -> tuple[nn.Module, StandardScaler]:
    scaler = StandardScaler()
    train_scaled = scale_features(
        train_features, scaler.fit(train_features.reshape(-1, config.n_features))
    )
    val_scaled = scale_features(val_features, scaler)

    if is_distributed():
        import torch.distributed as dist

        mean_t = torch.tensor(scaler.mean_, dtype=torch.float32)
        var_t = torch.tensor(scaler.var_, dtype=torch.float32)
        dist.broadcast(mean_t, src=0)
        dist.broadcast(var_t, src=0)
        scaler.mean_ = mean_t.numpy()
        scaler.var_ = var_t.numpy()
        scaler.scale_ = np.sqrt(scaler.var_)

    train_t = torch.tensor(train_scaled, dtype=torch.float32)
    train_y = torch.tensor(train_targets, dtype=torch.float32)
    val_t = torch.tensor(val_scaled, dtype=torch.float32)
    val_y = torch.tensor(val_targets, dtype=torch.float32)
    train_m_t = (
        torch.tensor(train_market, dtype=torch.float32)
        if train_market is not None
        else None
    )
    val_m_t = (
        torch.tensor(val_market, dtype=torch.float32)
        if val_market is not None
        else None
    )

    train_dataset = (
        TensorDataset(train_t, train_y, train_m_t)
        if train_m_t is not None
        else TensorDataset(train_t, train_y)
    )
    train_sampler: DistributedSampler | None = (
        DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
    )

    device = get_device()
    model = create_model(config, device)
    if pretrain_path and Path(pretrain_path).exists():
        unwrap_model(model).load_state_dict(
            torch.load(pretrain_path, weights_only=True, map_location=device)
        )
        if get_rank() == 0:
            print(f"  Loaded pre-trained weights from {pretrain_path}")
    Path(config.model_save_path).parent.mkdir(parents=True, exist_ok=True)
    use_amp = device.type in ("cuda", "mps")
    amp_scaler = torch.amp.GradScaler(device.type) if use_amp else None

    if loss_mode == "msrr":
        criterion = portfolio_mse_loss
        base_lr = config.learning_rate * 0.75
        body = [
            p
            for n, p in unwrap_model(model).named_parameters()
            if "output_head" not in n
        ]
        head = [
            p for n, p in unwrap_model(model).named_parameters() if "output_head" in n
        ]
        optimizer = optim.AdamW(
            [
                {"params": body, "weight_decay": 0.0},
                {"params": head, "weight_decay": 1e-3},
            ],
            lr=base_lr,
        )
    elif loss_mode == "margin":
        criterion = margin_ranking_loss
        optimizer = optim.AdamW(
            unwrap_model(model).parameters(),
            lr=config.learning_rate * 0.5,
            weight_decay=config.weight_decay,
        )
    elif loss_mode == "listnet":
        criterion = listnet_loss
        optimizer = optim.AdamW(
            unwrap_model(model).parameters(),
            lr=config.learning_rate * 0.3,
            weight_decay=config.weight_decay,
        )
    else:
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            unwrap_model(model).parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    best_val_loss = resume_best_loss
    patience_counter = resume_patience
    best_epoch = resume_epoch

    if resume_epoch > 0:
        loaded_epoch, best_val_loss, patience_counter = load_checkpoint(
            model, optimizer, scheduler, device, path=checkpoint_path
        )
        tqdm.write(
            f"  Resumed from epoch {loaded_epoch} (best_val_loss={best_val_loss:.6f})"
        )

    epoch_bar = tqdm(
        range(resume_epoch, config.max_epochs),
        desc=f"Training ({loss_mode.upper()})",
        unit="epoch",
        file=sys.stderr,
        initial=resume_epoch,
        total=config.max_epochs,
    )

    for epoch in epoch_bar:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            if train_m_t is not None:
                batch_x, batch_y, batch_m = batch
                batch_x, batch_y, batch_m = (
                    batch_x.to(device),
                    batch_y.to(device),
                    batch_m.to(device),
                )
            else:
                batch_x, batch_y = batch
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if train_m_t is not None:
                    pred = model(batch_x, market_state=batch_m)
                else:
                    pred = model(batch_x)
                loss = criterion(pred, batch_y) / grad_accum_steps
            if use_amp and amp_scaler is not None:
                amp_scaler.scale(loss).backward()
            else:
                loss.backward()
            if (step + 1) % grad_accum_steps == 0 or step == len(train_loader) - 1:
                if use_amp and amp_scaler is not None:
                    amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    unwrap_model(model).parameters(), config.max_grad_norm
                )
                if use_amp and amp_scaler is not None:
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            train_loss += loss.item() * grad_accum_steps

        model.eval()
        with torch.no_grad():
            if val_m_t is not None:
                val_pred = unwrap_model(model)(
                    val_t.to(device), market_state=val_m_t.to(device)
                )
            else:
                val_pred = unwrap_model(model)(val_t.to(device))
            val_loss = criterion(val_pred, val_y.to(device)).item()
        scheduler.step()
        epoch_bar.set_postfix(
            train_loss=f"{train_loss / len(train_loader):.6f}",
            val_loss=f"{val_loss:.6f}",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            if get_rank() == 0:
                torch.save(unwrap_model(model).state_dict(), config.model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                tqdm.write(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_val_loss,
                patience_counter,
                path=checkpoint_path,
            )

    if get_rank() == 0:
        unwrap_model(model).load_state_dict(
            torch.load(config.model_save_path, weights_only=True, map_location="cpu")
        )
    tqdm.write(f"  Best val_loss: {best_val_loss:.6f} at epoch {best_epoch}")
    return model, scaler


def train_seed(
    config: Config,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    seed: int,
    loss_mode: str = "mse",
    resume: bool = False,
    grad_accum_steps: int = 1,
    train_market: np.ndarray | None = None,
    val_market: np.ndarray | None = None,
    pretrain_path: str | None = None,
    checkpoint_path: str | None = None,
) -> tuple[nn.Module, StandardScaler]:
    # Per-seed checkpoint path so --seeds N --resume doesn't leak a previous seed's
    # checkpoint into the next seed (and to keep DDP ranks from racing on the same file).
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH.replace(".pt", f"_seed{seed}.pt")

    torch.manual_seed(seed)
    np.random.seed(seed)

    resume_epoch = resume_patience = 0
    resume_best_loss: float = float("inf")
    if resume and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        resume_epoch = ckpt["epoch"]
        resume_best_loss = ckpt["best_val_loss"]
        resume_patience = ckpt["patience_counter"]
        del ckpt

    return train(
        config,
        train_features,
        train_targets,
        val_features,
        val_targets,
        loss_mode=loss_mode,
        grad_accum_steps=grad_accum_steps,
        resume_epoch=resume_epoch,
        resume_best_loss=resume_best_loss,
        resume_patience=resume_patience,
        train_market=train_market,
        val_market=val_market,
        pretrain_path=pretrain_path,
        checkpoint_path=checkpoint_path,
    )


def run_training(
    config: Config,
    resume: bool = False,
    loss_mode: str = "mse",
    n_seeds: int = 1,
    grad_accum_steps: int = 1,
    train_path: str | None = None,
    val_path: str | None = None,
    pretrain_path: str | None = None,
) -> tuple[nn.Module, StandardScaler]:
    train_path = train_path or f"{config.features_path}/train.npz"
    val_path = val_path or f"{config.features_path}/val.npz"
    train_path_obj: Path = Path(train_path)  # type: ignore[arg-type]
    val_path_obj: Path = Path(val_path)  # type: ignore[arg-type]

    with np.load(train_path_obj) as data:
        train_features = data["features"]
        train_targets = data["targets"]
        train_market = data.get("market_state")
    with np.load(val_path_obj) as data:
        val_features = data["features"]
        val_targets = data["targets"]
        val_market = data.get("market_state")

    print(f"Train: {train_features.shape[0]} dates, Val: {val_features.shape[0]} dates")

    if n_seeds > 1:
        models = []
        scalers = []
        for seed_num in range(1, n_seeds + 1):
            print(f"\n  --- Seed {seed_num}/{n_seeds} ---")
            m, s = train_seed(
                config,
                train_features,
                train_targets,
                val_features,
                val_targets,
                seed=seed_num,
                loss_mode=loss_mode,
                resume=resume,
                grad_accum_steps=grad_accum_steps,
                train_market=train_market,
                val_market=val_market,
                pretrain_path=pretrain_path,
            )
            models.append(m)
            scalers.append(s)
            seed_path = config.model_save_path.replace(".pt", f"_seed{seed_num}.pt")
            # Gate seed-weight save to rank 0: every rank hitting torch.save
            # simultaneously to the same `seed_path` would corrupt the file
            # under DDP (same class of bug as save_checkpoint periodic save).
            if get_rank() == 0:
                torch.save(unwrap_model(m).state_dict(), seed_path)
                print(f"  Saved seed {seed_num} to {seed_path}")
        model, scaler = models[0], scalers[0]
    else:
        model, scaler = train_seed(
            config,
            train_features,
            train_targets,
            val_features,
            val_targets,
            seed=42,
            loss_mode=loss_mode,
            resume=resume,
            grad_accum_steps=grad_accum_steps,
            train_market=train_market,
            val_market=val_market,
            pretrain_path=pretrain_path,
        )

    save_scaler(scaler, f"{config.features_path}/scaler.json")
    if get_rank() == 0:
        print(f"Training complete. Best model saved to {config.model_save_path}")
    return model, scaler
