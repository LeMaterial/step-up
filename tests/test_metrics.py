"""Verify D-MAE / D-RMSE / coord-RMSD have expected mathematical behavior."""

from __future__ import annotations

import torch

from step_up.eval.metrics import cdist_mae, cdist_rmse, coord_rmsd, per_element_dmae


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
    # Atomic numbers: first mol = C,N,O (5,6,7); second mol = H,H,H,H (0).
    z = torch.tensor([[5, 6, 7, 0], [0, 0, 0, 0]], dtype=torch.long)
    per = per_element_dmae(pred, coords, mask, z)
    # Uniform translation leaves pairwise distances unchanged -> 0 per element.
    for z_val, v in per.items():
        assert v == 0.0, f"element {z_val} should have 0 D-MAE under translation"
