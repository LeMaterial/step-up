"""Verify D-MAE / D-RMSE / coord-RMSD have expected mathematical behavior."""

from __future__ import annotations

import pytest
import torch

from step_up.data.csv_dataset import CSVMoleculeDataset
from step_up.eval.metrics import cdist_mae, cdist_rmse, coord_rmsd, per_element_dmae
from step_up.models.rebind import get_collator


def _toy_batch() -> tuple[torch.Tensor, torch.Tensor]:
    # Two molecules: first has 3 valid atoms, second has 4.
    coords = torch.zeros(2, 4, 3, dtype=torch.float32)
    coords[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 1] = torch.tensor([1.0, 0.0, 0.0])
    coords[0, 2] = torch.tensor([0.0, 1.0, 0.0])
    coords[1, 0] = torch.tensor([0.0, 0.0, 0.0])
    coords[1, 1] = torch.tensor([2.0, 0.0, 0.0])
    coords[1, 2] = torch.tensor([0.0, 2.0, 0.0])
    coords[1, 3] = torch.tensor([0.0, 0.0, 2.0])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.float32)
    return coords, mask


def test_zero_when_identical() -> None:
    coords, mask = _toy_batch()
    assert cdist_mae(coords, coords, mask).item() == 0.0
    assert cdist_rmse(coords, coords, mask).item() == 0.0
    assert coord_rmsd(coords, coords, mask).item() == 0.0


def test_dmae_positive_on_perturbation() -> None:
    coords, mask = _toy_batch()
    noisy = coords + 0.1 * torch.randn_like(coords) * mask.unsqueeze(-1)
    assert cdist_mae(noisy, coords, mask).item() > 0.0
    assert cdist_rmse(noisy, coords, mask).item() > 0.0


def test_per_element_dmae_groups_by_z() -> None:
    coords, mask = _toy_batch()
    pred = coords + 0.5  # uniform shift; cdist is translation-invariant -> 0
    # True atomic numbers: first mol = C, N, O (Z = 6, 7, 8); padding atom is
    # whatever — the function uses node_mask, not a sentinel Z. Second mol =
    # H, H, H, H (Z = 1) — H must NOT be silently dropped just because Z=1
    # is small.
    z = torch.tensor([[6, 7, 8, 0], [1, 1, 1, 1]], dtype=torch.long)
    per = per_element_dmae(pred, coords, mask, z)
    # Uniform translation leaves pairwise distances unchanged -> 0 per element.
    assert set(per.keys()) == {6, 7, 8, 1}, f"hydrogen must appear in keys: {per.keys()}"
    for z_val, v in per.items():
        assert v == 0.0, f"element {z_val} should have 0 D-MAE under translation"


def test_per_element_dmae_nonzero_on_perturbation() -> None:
    coords, mask = _toy_batch()
    z = torch.tensor([[6, 7, 8, 0], [1, 1, 1, 1]], dtype=torch.long)
    noisy = coords + 0.1 * torch.randn_like(coords) * mask.unsqueeze(-1)
    per = per_element_dmae(noisy, coords, mask, z)
    # All real elements should pick up some error.
    for z_val in (1, 6, 7, 8):
        assert per[z_val] > 0.0, f"element {z_val} should have nonzero D-MAE"


def test_per_element_dmae_takes_collated_node_type(qm9_path) -> None:
    """A collated batch's node_type is already Z (H = 1) and can be passed directly."""
    ds = CSVMoleculeDataset(qm9_path, source="smiles", subset_size=4)
    batch = get_collator()()([ds[i] for i in range(len(ds))])
    coords = batch["conformer"]
    per = per_element_dmae(coords, coords, batch["node_mask"], batch["node_type"])
    assert 1 in per, f"hydrogen missing, so node_type wasn't read as true Z: {sorted(per)}"
    assert set(per) <= {1, 6, 7, 8, 9}  # QM9 elements: H, C, N, O, F


def test_per_element_dmae_rejects_z_minus_1_indices() -> None:
    coords, mask = _toy_batch()
    # Graph-dict style Z - 1 indices: C, N, O (+ padding), then four hydrogens as 0.
    z_minus_1 = torch.tensor([[5, 6, 7, 0], [0, 0, 0, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="true atomic numbers"):
        per_element_dmae(coords, coords, mask, z_minus_1)
