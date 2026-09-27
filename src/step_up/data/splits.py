"""Deterministic train/val/test split helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from torch.utils.data import Dataset, Subset

SPLIT_NAMES = ("train", "val", "test")
# Names other projects use for the same three splits.
_SPLIT_ALIASES = {"valid": "val", "validation": "val", "dev": "val", "eval": "val"}


def split_by_labels(dataset: Dataset, labels: Sequence[str]) -> tuple[Subset, Subset, Subset]:
    """Partition ``dataset`` by an explicit per-row split name.

    Use this to adopt a published split (``labels[i]`` is item ``i``'s split)
    instead of deriving one. ``valid``/``validation``/``dev``/``eval`` are
    accepted as spellings of ``val``.
    """
    n = len(dataset)  # type: ignore[arg-type]
    if len(labels) != n:
        raise ValueError(f"Got {len(labels)} labels for a dataset of length {n}")
    buckets: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    for i, raw in enumerate(labels):
        name = str(raw).strip().lower()
        name = _SPLIT_ALIASES.get(name, name)
        if name not in buckets:
            raise ValueError(
                f"Unknown split name {raw!r} at row {i}; expected one of {SPLIT_NAMES}"
            )
        buckets[name].append(i)
    return tuple(Subset(dataset, buckets[name]) for name in SPLIT_NAMES)  # type: ignore[return-value]


_ID_LIST_HEADERS = {"refcode", "id", "mol_id", "identifier", "name", "complex"}


def read_id_list(path: str | Path) -> list[str]:
    """Read IDs from a one-per-line text file or a single-column CSV.

    A leading line that looks like a column header (``refcode``, ``id``, ...) is
    skipped, and only the first comma-separated field of each line is used.
    """
    ids: list[str] = []
    for line in Path(path).read_text().splitlines():
        entry = line.split(",")[0].strip()
        if entry:
            ids.append(entry)
    if ids and ids[0].lower() in _ID_LIST_HEADERS:
        ids = ids[1:]
    return ids


def split_by_id_files(
    dataset: Dataset, keys: Sequence[str], files: Mapping[str, str | Path]
) -> tuple[Subset, Subset, Subset]:
    """Partition ``dataset`` using published ID lists, one file per split.

    ``files`` maps a split name to a file of IDs, and ``keys[i]`` is item ``i``'s
    ID. Items listed in none of the files are dropped, which is how a split
    published against a slightly different snapshot of a dataset still applies
    to ours.
    """
    assignment: dict[str, str] = {}
    for raw_name, path in files.items():
        name = str(raw_name).strip().lower()
        name = _SPLIT_ALIASES.get(name, name)
        if name not in SPLIT_NAMES:
            raise ValueError(f"Unknown split name {raw_name!r}; expected one of {SPLIT_NAMES}")
        for identifier in read_id_list(path):
            previous = assignment.get(identifier)
            if previous is not None and previous != name:
                raise ValueError(f"ID {identifier!r} is listed in both {previous} and {name}")
            assignment[identifier] = name

    buckets: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    unlisted = 0
    for i, key in enumerate(keys):
        name = assignment.get(str(key))
        if name is None:
            unlisted += 1
            continue
        buckets[name].append(i)
    if unlisted:
        print(
            f"[split] {unlisted} rows are in none of the split files and were dropped", flush=True
        )
    return tuple(Subset(dataset, buckets[name]) for name in SPLIT_NAMES)  # type: ignore[return-value]


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
