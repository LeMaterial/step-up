"""Forward pass on a 4-molecule QM9 batch through a tiny ReBind."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset

from step_up.data.csv_dataset import CSVMoleculeDataset
from step_up.models.rebind import Collator, build_rebind


def test_forward_tiny_qm9(qm9_path) -> None:
    ds = CSVMoleculeDataset(qm9_path, source="smiles", subset_size=8)
    subset = Subset(ds, [0, 1, 2, 3])
    loader = DataLoader(subset, batch_size=4, shuffle=False, collate_fn=Collator())
    batch = next(iter(loader))

    model = build_rebind(n_layers=2, d_model=32, d_ffn=64, n_head=4)
    model.eval()
    with torch.no_grad():
        out = model(**batch)

    assert out.loss.dim() == 0
    assert torch.isfinite(out.loss)
    assert out.conformer_hat.shape == out.conformer.shape
    b, n, _ = out.conformer_hat.shape
    assert b == 4
    assert n > 0
    assert torch.isfinite(out.conformer_hat).all()


def test_lj_patch_supports_heavy_z(tmqmg_path) -> None:
    """Verify that the LJ patch lets tmQMg (Z up to 80) pass through Collator."""
    ds = CSVMoleculeDataset(tmqmg_path, source="mol2", subset_size=2)
    subset = Subset(ds, [0, 1])
    loader = DataLoader(subset, batch_size=2, shuffle=False, collate_fn=Collator())
    # This would KeyError pre-patch on transition-metal indices.
    batch = next(iter(loader))
    assert "sigma" in batch and "epsilon" in batch
    assert torch.isfinite(batch["sigma"]).all()
    assert torch.isfinite(batch["epsilon"]).all()
