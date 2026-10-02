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

    ``atomic_numbers`` is ``(B, N)`` of **true atomic numbers** (1 for H, 6 for
    C, etc.). Padding positions are identified via ``node_mask`` (1 = valid,
    0 = padding) — *not* via a sentinel atomic-number value.

    ReBind uses two encodings, so check which one you have:

    - A collated batch's ``node_type`` is already true atomic numbers (the
      collator adds 1, with 0 at padding). Pass it directly.
    - A single graph dict's ``node_type`` is ``Z - 1`` (H is 0). Add 1 first.

    Raises ``ValueError`` if a non-padding atom has atomic number < 1, which
    catches ``Z - 1`` indices passed by mistake whenever hydrogen is present.

    Returns a dict mapping Z -> mean |Δd| over pairs (i, j) with i or j of
    that element (each pair contributes to both endpoint elements).
    """
    mask = _pair_mask(node_mask)
    diff = (torch.cdist(pred, pred) - torch.cdist(target, target)).abs() * mask
    valid = node_mask.to(torch.bool)

    zs = atomic_numbers.to(torch.long)
    if (zs[valid] < 1).any():
        raise ValueError(
            "atomic_numbers must be true atomic numbers (H = 1), but a non-padding atom "
            "has Z < 1. Graph-dict node_type is Z - 1; add 1 before calling."
        )
    per_z: dict[int, float] = {}
    for z in torch.unique(zs[valid]):
        z_int = int(z.item())
        # Pairs where either endpoint is a valid atom of element z.
        endpoint = ((zs == z) & valid).to(torch.float32)
        endpoint_mask = (endpoint.unsqueeze(-1) + endpoint.unsqueeze(-2)).clamp_max(1.0)
        pair_mask = mask * endpoint_mask
        denom = pair_mask.sum().clamp_min(1)
        per_z[z_int] = float((diff * endpoint_mask).sum() / denom)
    return per_z
