"""Conformer-quality metrics matching ReBind's evaluate.py.

All metrics operate on padded coordinate tensors of shape ``(B, N, 3)`` with a
boolean ``node_mask`` of shape ``(B, N)`` marking valid (non-padding) atoms.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _pair_mask(node_mask: torch.Tensor) -> torch.Tensor:
    m = node_mask.to(torch.bool).unsqueeze(-1)
    return (m & m.transpose(-1, -2)).to(torch.float32)


def cdist_mae(pred: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error on pairwise distance matrices (ReBind ``D-MAE``)."""
    mask = _pair_mask(node_mask)
    d_pred = torch.cdist(pred, pred) * mask
    d_true = torch.cdist(target, target) * mask
    return F.l1_loss(d_pred, d_true, reduction="sum") / mask.sum().clamp_min(1)


def cdist_rmse(pred: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Root mean squared error on pairwise distance matrices (ReBind ``D-RMSE``)."""
    mask = _pair_mask(node_mask)
    d_pred = torch.cdist(pred, pred) * mask
    d_true = torch.cdist(target, target) * mask
    mse = F.mse_loss(d_pred, d_true, reduction="sum") / mask.sum().clamp_min(1)
    return torch.sqrt(mse)


def coord_rmsd(pred: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Per-molecule coordinate RMSD averaged over the batch (no Kabsch alignment).

    ReBind aligns predictions to targets *inside* the model head before the loss,
    so by the time tensors leave the model, this raw RMSD is already
    rotation-aware. For unaligned eval (e.g., a baseline that does no
    alignment), use ``kabsch_rmsd_per_mol`` below.
    """
    delta = (pred - target).to(torch.float32)
    sq = (delta * delta).sum(dim=-1) * node_mask.to(torch.float32)
    msd = sq.sum(dim=-1) / node_mask.sum(dim=-1).clamp_min(1)
    return torch.sqrt(msd).mean()


def per_element_dmae(
    pred: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
    atomic_numbers: torch.Tensor,
) -> dict[int, float]:
    """Per-element D-MAE: |d_ij_pred - d_ij_true| averaged over pairs touching Z.

    ``atomic_numbers`` is ``(B, N)`` of integer Z values (use 0 for padding).
    Returns a dict mapping Z -> mean |Δd| over pairs (i,j) with i or j of that
    element (each pair contributes to both endpoint elements).
    """
    mask = _pair_mask(node_mask)
    diff = (torch.cdist(pred, pred) - torch.cdist(target, target)).abs() * mask

    zs = atomic_numbers.to(torch.long)
    per_z: dict[int, float] = {}
    for z in torch.unique(zs):
        if int(z.item()) == 0:
            continue
        # Pairs where either endpoint has atomic number z.
        endpoint_mask = (zs == z).unsqueeze(-1) | (zs == z).unsqueeze(-2)
        pair_mask = mask * endpoint_mask.to(torch.float32)
        denom = pair_mask.sum().clamp_min(1)
        per_z[int(z.item())] = float((diff * endpoint_mask.to(torch.float32)).sum() / denom)
    return per_z
