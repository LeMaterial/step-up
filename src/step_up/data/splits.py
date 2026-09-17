"""Deterministic train/val/test split helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from torch.utils.data import Dataset, Subset


def _unit_interval(key: str, seed: int) -> float:
    """Map ``(seed, key)`` to a float in [0, 1), identical on every platform and run."""
    digest = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def stable_split(
    dataset: Dataset,
    keys: Sequence[str],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[Subset, Subset, Subset]:
    """Split ``dataset`` into train/val/test by hashing each item's key.

    ``keys[i]`` identifies item ``i``, e.g. a molecule ID or its row number in the
    source CSV. An item's split depends only on its own key, ``seed`` and ``ratios``,
    so it never moves when other items are added, removed, filtered out, or fail
    featurization, and items that share a key always share a split. Split sizes
    follow ``ratios`` only approximately, which matters just for tiny datasets.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1, got {ratios}")
    n = len(dataset)  # type: ignore[arg-type]
    if len(keys) != n:
        raise ValueError(f"Got {len(keys)} keys for a dataset of length {n}")
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for i, key in enumerate(keys):
        u = _unit_interval(key, seed)
        if u < ratios[0]:
            train_idx.append(i)
        elif u < ratios[0] + ratios[1]:
            val_idx.append(i)
        else:
            test_idx.append(i)
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
