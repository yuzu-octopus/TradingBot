# Plan 002: DDP correctness — barrier, scaler broadcast, CUDA guard

> Drift check: `git diff --stat 672167a..HEAD -- training/train.py training/pretrain.py src/utils.py main.py`

## Status

- Priority: P1
- Effort: M
- Risk: MED
- Depends on: none
- Category: bug
- Planned at: commit `672167a`, 2026-06-23

## Why this matters

Three DDP correctness bugs that only surface under `torchrun` (multi-GPU): (1) no `dist.barrier()` means non-rank-0 ranks read partial checkpoints, (2) `StandardScaler` is fitted per-rank on sharded data producing different scalers on each rank, and (3) the `device.type == "cuda"` guard on DDP wrapping means CPU-based DDP trains without gradient sync.

## Current state

**F16/F32** — No `dist.barrier()` anywhere. After training, all ranks immediately proceed to threshold optimization. Rank 0 may still be writing `best.pt`:

```python
# train.py:270-271  (last improvement)
if get_rank() == 0:
    torch.save(unwrap_model(model).state_dict(), config.model_save_path)

# main.py:473  (immediately after training loop)
buy_t, sell_t = run_threshold_optimization(config)  # all ranks load best.pt
```

**F17** — `run_training` returns scalers from every rank. Under DDP, each rank fits its own scaler on the shard:

```python
# train.py:94-95
train_scaled = scale_features(
    train_features, scaler.fit(train_features.reshape(-1, config.n_features))
)
# Under DistributedSampler, train_features is 1/world_size rows on each rank
```

**F18** — DDP wrapping is guarded by `device.type == "cuda"`:

```python
# utils.py:28-29
if is_distributed() and device.type == "cuda":
    model = DistributedDataParallel(model, device_ids=[device.index])
```

## Scope

**In scope**: `training/train.py`, `training/pretrain.py`, `src/utils.py`, `main.py`

**Out of scope**: Any test changes, any other file.

## Steps

### Step 1: Add `dist.barrier()` after training, before threshold optimization

In `main.py`, in the training mode branch, after `run_training(...)` returns and before `run_threshold_optimization(config)`:

```python
from config import is_distributed
...
if is_distributed():
    import torch.distributed as dist
    dist.barrier()
```

### Step 2: Broadcast scaler from rank 0 after fitting

In `training/train.py`, after the scaler is fitted (line 94-95), broadcast it from rank 0:

```python
from config import is_distributed, get_rank

# After scaler is fitted
if is_distributed():
    import torch.distributed as dist
    if get_rank() == 0:
        mean_t = torch.tensor(scaler.mean_, dtype=torch.float32)
        var_t = torch.tensor(scaler.var_, dtype=torch.float32)
    else:
        mean_t = torch.zeros(scaler.n_features_in_, dtype=torch.float32)
        var_t = torch.zeros(scaler.n_features_in_, dtype=torch.float32)
    dist.broadcast(mean_t, src=0)
    dist.broadcast(var_t, src=0)
    scaler.mean_ = mean_t.numpy()
    scaler.var_ = var_t.numpy()
    scaler.scale_ = np.sqrt(scaler.var_)
```

### Step 3: Remove the `device.type == "cuda"` guard

In `src/utils.py:28`:

```python
if is_distributed():
    model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
```

And in `training/pretrain.py:88-91` for `top_head`:

```python
if is_distributed():
    top_head = nn.parallel.DistributedDataParallel(
        top_head, device_ids=[device.index] if device.type == "cuda" else None
    )
```

**Verify**: `uv run ruff check . && uv run ruff format . && uv run pytest -q` → all pass

Install torch deps check: `uv run python -c "from config import is_distributed; print(f'ok dist={is_distributed()}')"` → `ok dist=False`

## Test plan

Existing tests (`test_ddp.py`) cover the non-DDP path. The barrier/scaler broadcast changes are only testable with actual `torchrun` (requires multi-GPU). Rely on code review for correctness.

## Done criteria

- [ ] `uv run ruff check .` exits 0
- [ ] `uv run ruff format --check .` exits 0
- [ ] `uv run pytest -q` → 60 passed
- [ ] `dist.barrier()` exists between training and threshold optimization in `main.py`
- [ ] Scaler broadcast logic exists in `training/train.py`
- [ ] DDP wrapping no longer requires `device.type == "cuda"` in `utils.py` and `pretrain.py`

## STOP conditions

- Cannot test DDP paths without `torchrun` — that's expected. Do not add fake process groups.
- If `dist.broadcast` is not available in the installed PyTorch version, stop and report.

## Maintenance notes

The scaler broadcast assumes the scaler is fitted on the full dataset in a non-sharded manner. If `DistributedSampler` options change, the broadcast logic must be reviewed.
