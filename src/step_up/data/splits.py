"""Random and reproducible split helpers."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, Subset


def random_split(
    dataset: Dataset,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[Subset, Subset, Subset]:
    """Random train/val/test split. Ratios must sum to 1."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1, got {ratios}")
    n = len(dataset)  # type: ignore[arg-type]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
