"""Config-driven training loop for step-up.

The loop is deliberately minimal: it owns dataset construction, the ReBind
model, AdamW with linear warmup and cosine decay, periodic validation,
best-checkpoint saving, and TensorBoard logging. No HuggingFace Trainer, no
accelerate — keeping the control flow legible while we're still iterating on
the model.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data.csv_dataset import CSVMoleculeDataset
from .data.splits import stable_split
from .models.rebind import build_rebind, get_collator

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    dataset_path: str
    dataset_source: str  # "smiles" | "mol2"
    subset_size: int | None = None
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int = 0
    # Column with a stable per-molecule ID (e.g. `refcode`). Each molecule's split
    # is a hash of this key and `split_seed` (see `stable_split`); without it the
    # key is the CSV row number.
    id_column: str | None = None
    # Optional CSV column filter (e.g. `filter_column: spinmult, filter_value: 1`
    # to restrict BOSTMC to singlets).
    filter_column: str | None = None
    filter_value: Any = None
    # Featurize every row up front and drop the ones that fail. Disable only for
    # datasets known to be clean; a bad row then raises mid-training instead.
    validate_dataset: bool = True
    # Keep featurized graph dicts in memory instead of re-featurizing every epoch.
    # Trades memory for speed; recommended for smoke runs.
    cache_dataset: bool = False
    # Fail validation if more than this fraction of rows are dropped.
    max_drop_fraction: float = 0.5

    # Model
    n_layers: int = 8
    d_model: int = 512
    d_ffn: int = 1024
    n_head: int = 8
    dropout: float = 0.0

    # Optimization
    epochs: int = 20
    batch_size: int = 100
    eval_batch_size: int = 100
    lr: float = 9e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    # Global gradient-norm clip. ReBind's published training inherited
    # HuggingFace Trainer's default of 1.0; without it the 8-layer / d=512
    # model NaNs out in the first epoch on QM9.
    grad_clip: float = 1.0
    # Skip batches whose loss comes back non-finite (the LJ block in ReBind's
    # head can divide by predicted distances that are ~0 early in training).
    # Abort if more than this many batches are skipped over the whole run.
    nan_skip_max: int = 50
    num_workers: int = 0

    # Runtime
    device: str = "cpu"
    output_dir: str = "outputs/run"
    seed: int = 0
    log_interval: int = 10

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if "split_ratios" in raw:
            raw["split_ratios"] = tuple(raw["split_ratios"])
        return cls(**raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _nonfinite_keys(batch: dict[str, Any]) -> list[str]:
    """Return floating-point tensors in ``batch`` that contain non-finite values."""
    bad: list[str] = []
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dtype.is_floating_point and not torch.isfinite(v).all():
            bad.append(k)
    return bad


def _all_params_finite(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return False
    return True


def _warmup_cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _mean_or_nan(total: float, count: int) -> float:
    # NaN rather than 0.0 when nothing was averaged: a 0.0 D-MAE would read as a
    # perfect score and win best-checkpoint selection.
    return total / count if count else math.nan


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    scheduler_state: dict[str, int],
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    grad_clip: float = 0.0,
    nan_skip_max: int = 0,
) -> tuple[float, float]:
    """Run one pass over ``loader``. Returns (mean_loss, mean_dmae).

    During training, batches whose loss comes back non-finite are skipped
    (no backward, no optimizer step). The first occurrence prints whether the
    badness is in the input batch or appeared inside the forward pass — useful
    to distinguish data corruption from model-side numerical instability.
    During evaluation, non-finite batches are excluded from the means with a
    warning. If no batch contributes, both means are NaN.
    """
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum, dmae_sum, n = 0.0, 0.0, 0
    pbar = tqdm(loader, leave=False, desc="train" if is_train else "val", dynamic_ncols=True)
    for batch in pbar:
        batch = _move_batch_to_device(batch, device)
        if is_train:
            lr_now = _warmup_cosine_lr(scheduler_state["step"], total_steps, warmup_steps, base_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr_now
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            loss_val = out.loss
            if not torch.isfinite(loss_val):
                scheduler_state["nan_skipped"] = scheduler_state.get("nan_skipped", 0) + 1
                skipped = scheduler_state["nan_skipped"]
                bad_inputs = _nonfinite_keys(batch)
                n_atoms = batch["node_mask"].sum(dim=-1).long().tolist()
                origin = (
                    f"input nonfinite in {bad_inputs}"
                    if bad_inputs
                    else "input is clean -> NaN originated in the model forward"
                )
                print(
                    f"WARN: non-finite loss at step {scheduler_state['step']} "
                    f"(skipped {skipped}/{nan_skip_max}); "
                    f"batch n_atoms={n_atoms}; {origin}",
                    flush=True,
                )
                if skipped > nan_skip_max:
                    raise RuntimeError(
                        f"Exceeded nan_skip_max={nan_skip_max} non-finite batches. "
                        "Lower lr, raise grad_clip stringency, or filter the dataset."
                    )
                scheduler_state["step"] += 1
                continue
            loss_val.backward()
            # clip_grad_norm_ returns the *unclipped* total gradient norm. If
            # it's non-finite, the gradients themselves contain NaN/Inf — a
            # well-known failure mode of torch.svd on rank-deficient inputs,
            # which the Kabsch alignment inside ReBind's conformer head can
            # produce. Calling optimizer.step() with NaN gradients would
            # poison every parameter with NaN, making every subsequent
            # forward pass NaN. Skip the step in that case.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip if grad_clip > 0 else float("inf")
            )
            if not torch.isfinite(grad_norm):
                scheduler_state["nan_skipped"] = scheduler_state.get("nan_skipped", 0) + 1
                skipped = scheduler_state["nan_skipped"]
                print(
                    f"WARN: non-finite gradient norm ({grad_norm.item()}) at step "
                    f"{scheduler_state['step']} (skipped {skipped}/{nan_skip_max}). "
                    f"Likely a degenerate batch through Kabsch SVD.",
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                if skipped > nan_skip_max:
                    raise RuntimeError(
                        f"Exceeded nan_skip_max={nan_skip_max} non-finite batches. "
                        "Lower lr or filter the dataset."
                    )
                scheduler_state["step"] += 1
                continue
            optimizer.step()
            # Cheap sanity: any parameter going NaN means we have a bug
            # upstream (e.g. another path that bypassed the grad-norm check).
            # Fail loudly instead of producing meaningless NaN for thousands
            # of steps.
            if not _all_params_finite(model):
                raise RuntimeError(
                    f"Model parameters went non-finite after optimizer step at "
                    f"step {scheduler_state['step']}. This indicates a code "
                    "bug bypassing the gradient-NaN check."
                )
            scheduler_state["step"] += 1
        else:
            with torch.no_grad():
                out = model(**batch)
            if not torch.isfinite(out.loss):
                print("WARN: non-finite loss on an evaluation batch; excluded from metrics.")
                continue
        loss_sum += float(out.loss.detach())
        dmae_sum += float(out.cdist_mae.detach())
        n += 1
        pbar.set_postfix(loss=f"{loss_sum / n:.4f}", dmae=f"{dmae_sum / n:.4f}")
    return _mean_or_nan(loss_sum, n), _mean_or_nan(dmae_sum, n)


def train(config: TrainConfig) -> dict[str, Any]:
    _set_seed(config.seed)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    # Data
    dataset = CSVMoleculeDataset(
        path=config.dataset_path,
        source=config.dataset_source,  # type: ignore[arg-type]
        subset_size=config.subset_size,
        validate=config.validate_dataset,
        max_drop_fraction=config.max_drop_fraction,
        filter_column=config.filter_column,
        filter_value=config.filter_value,
        cache=config.cache_dataset,
        id_column=config.id_column,
    )
    train_set, val_set, test_set = stable_split(
        dataset, dataset.split_keys(), ratios=config.split_ratios, seed=config.split_seed
    )
    print(
        f"[split] train={len(train_set)} val={len(val_set)} test={len(test_set)} "
        f"(keyed on {config.id_column or 'CSV row number'}, seed={config.split_seed})",
        flush=True,
    )
    if len(train_set) == 0 or len(val_set) == 0:
        raise ValueError(
            f"Empty train or val split from {len(dataset)} molecules with "
            f"split_ratios={config.split_ratios}. Use more data or larger ratios."
        )
    collator = get_collator()()

    def _make_loader(subset: Subset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            collate_fn=collator,
            persistent_workers=config.num_workers > 0,
        )

    train_loader = _make_loader(train_set, config.batch_size, shuffle=True)
    val_loader = _make_loader(val_set, config.eval_batch_size, shuffle=False)

    # Model + optimizer
    model = build_rebind(
        n_layers=config.n_layers,
        d_model=config.d_model,
        d_ffn=config.d_ffn,
        n_head=config.n_head,
        dropout=config.dropout,
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    writer = SummaryWriter(out_dir / "tb")
    scheduler_state: dict[str, int] = {"step": 0}
    history: list[dict[str, float]] = []
    best_val = math.inf
    best_epoch: int | None = None
    t0 = time.time()
    for epoch in range(1, config.epochs + 1):
        train_loss, train_dmae = _run_epoch(
            model,
            train_loader,
            optimizer,
            config.device,
            scheduler_state,
            config.lr,
            total_steps,
            warmup_steps,
            grad_clip=config.grad_clip,
            nan_skip_max=config.nan_skip_max,
        )
        val_loss, val_dmae = _run_epoch(
            model,
            val_loader,
            None,
            config.device,
            scheduler_state,
            config.lr,
            total_steps,
            warmup_steps,
        )
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/dmae", train_dmae, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/dmae", val_dmae, epoch)
        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:>3}/{config.epochs}] "
            f"train_loss={train_loss:.4f} train_dmae={train_dmae:.4f} "
            f"val_loss={val_loss:.4f} val_dmae={val_dmae:.4f} "
            f"elapsed={elapsed:.1f}s"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dmae": train_dmae,
                "val_loss": val_loss,
                "val_dmae": val_dmae,
            }
        )
        if math.isfinite(val_dmae) and val_dmae < best_val:
            best_val, best_epoch = val_dmae, epoch
            torch.save(model.state_dict(), out_dir / "best.pt")
    writer.close()

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    if best_epoch is None:
        raise RuntimeError(
            "No epoch produced a finite validation D-MAE, so no checkpoint was saved. "
            f"Per-epoch metrics are in {out_dir / 'history.json'}."
        )

    # ------------------------------------------------------------------
    # Test-set evaluation on this run's best checkpoint (selected by val D-MAE).
    # ------------------------------------------------------------------
    test_metrics: dict[str, float] = {}
    if len(test_set) == 0:
        print("WARN: test split is empty; skipping test evaluation.", flush=True)
    else:
        model.load_state_dict(torch.load(out_dir / "best.pt", map_location=config.device))
        test_loader = _make_loader(test_set, config.eval_batch_size, shuffle=False)
        test_loss, test_dmae = _run_epoch(
            model,
            test_loader,
            None,
            config.device,
            scheduler_state,
            config.lr,
            total_steps,
            warmup_steps,
        )
        test_metrics = {"test_loss": test_loss, "test_dmae": test_dmae}
        with open(out_dir / "test_metrics.json", "w") as f:
            json.dump({**test_metrics, "best_epoch": best_epoch}, f, indent=2)
        print(
            f"[test] test_loss={test_loss:.4f} test_dmae={test_dmae:.4f} "
            f"(checkpoint from epoch {best_epoch})"
        )

    return {"history": history, "best_val_dmae": best_val, "best_epoch": best_epoch, **test_metrics}
