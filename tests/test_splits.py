"""Stable, hash-based train/val/test splits."""

from __future__ import annotations

import random

import pytest

from step_up.data.splits import stable_split

RATIOS = (0.8, 0.1, 0.1)


def _membership(keys: list[str], seed: int = 0) -> dict[str, str]:
    """Map each key to the name of the split it lands in."""
    subsets = stable_split(list(range(len(keys))), keys, ratios=RATIOS, seed=seed)
    return {
        keys[i]: name
        for name, subset in zip(("train", "val", "test"), subsets, strict=True)
        for i in subset.indices
    }


def test_membership_is_independent_of_other_rows() -> None:
    keys = [f"mol{i}" for i in range(2000)]
    full = _membership(keys)
    # Drop a third of the rows (as filtering or failed featurization would) and shuffle.
    kept = [key for i, key in enumerate(keys) if i % 3]
    random.Random(0).shuffle(kept)
    assert _membership(kept) == {key: full[key] for key in kept}


def test_proportions_seed_and_shared_keys() -> None:
    keys = [f"mol{i}" for i in range(20000)]
    membership = _membership(keys)
    for name, ratio in zip(("train", "val", "test"), RATIOS, strict=True):
        fraction = sum(split == name for split in membership.values()) / len(keys)
        assert abs(fraction - ratio) < 0.01, (name, fraction)
    assert _membership(keys, seed=1) != membership
    # Rows sharing a key (e.g. duplicated molecules) always share a split.
    subsets = stable_split(list(range(4)), ["a", "b", "a", "b"], ratios=RATIOS)
    split_of = {i: s for s, subset in enumerate(subsets) for i in subset.indices}
    assert split_of[0] == split_of[2]
    assert split_of[1] == split_of[3]


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        stable_split([0, 1], ["a", "b"], ratios=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="keys"):
        stable_split([0, 1], ["a"], ratios=RATIOS)
