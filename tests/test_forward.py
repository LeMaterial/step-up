"""Forward pass on a 4-molecule QM9 batch through a tiny ReBind."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from step_up.data.csv_dataset import CSVMoleculeDataset
from step_up.models import rebind
from step_up.models.lj_params import UFF_LJ_PARAMETERS
from step_up.models.rebind import build_rebind, get_collator


def test_forward_tiny_qm9(qm9_path) -> None:
    ds = CSVMoleculeDataset(qm9_path, source="smiles", subset_size=8)
    subset = Subset(ds, [0, 1, 2, 3])
    loader = DataLoader(subset, batch_size=4, shuffle=False, collate_fn=get_collator()())
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


def test_forward_with_backward(qm9_path) -> None:
    """End-to-end train-mode forward+backward through the patched REBIND.forward.

    Specifically exercises the residual_head path that uses
    ``inputs["pred_conformation"] = node_embedding`` as ``conformer_base`` —
    catches future regressions where someone might be tempted to switch this
    to ``conformer_cache`` (which would crash on a shape mismatch).
    """
    ds = CSVMoleculeDataset(qm9_path, source="smiles", subset_size=8)
    subset = Subset(ds, [0, 1, 2, 3])
    loader = DataLoader(subset, batch_size=4, shuffle=False, collate_fn=get_collator()())
    batch = next(iter(loader))

    model = build_rebind(n_layers=2, d_model=32, d_ffn=64, n_head=4)
    model.train()
    out = model(**batch)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    # At least one parameter must have a finite, nonzero gradient.
    finite_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and (p.grad != 0).any()
        for p in model.parameters()
    )
    assert finite_grad, "no parameter received a finite, nonzero gradient"


def test_patched_forward_matches_upstream(qm9_path, monkeypatch) -> None:
    """The patched model must reproduce upstream ReBind's outputs exactly.

    Upstream's in-place writes only break autograd, so under ``no_grad`` the
    original methods run and the two forwards can be compared directly. This
    guards the residual head's ``conformer_base``, which upstream builds through
    an in-place write in its decoder. The LJ distance clamp is the one intended
    numerical difference, so it is disabled here.
    """
    ds = CSVMoleculeDataset(qm9_path, source="smiles", subset_size=8)
    subset = Subset(ds, [0, 1, 2, 3])
    loader = DataLoader(subset, batch_size=4, shuffle=False, collate_fn=get_collator()())
    batch = next(iter(loader))

    torch.manual_seed(0)
    model = build_rebind(n_layers=2, d_model=32, d_ffn=64, n_head=4)
    model.eval()
    monkeypatch.setattr(rebind, "_LJ_D_MIN", 0.0)
    with torch.no_grad():
        patched = model(**batch)
        for cls, upstream_forward in rebind._UPSTREAM_FORWARDS.items():
            monkeypatch.setattr(cls, "forward", upstream_forward)
        upstream = model(**batch)

    torch.testing.assert_close(patched.loss, upstream.loss)
    torch.testing.assert_close(patched.conformer_hat, upstream.conformer_hat)


def test_lj_patch_supports_heavy_z(tmqmg_path) -> None:
    """Verify that the LJ patch lets tmQMg (Z up to 80) pass through Collator."""
    ds = CSVMoleculeDataset(tmqmg_path, source="mol2", subset_size=2)
    subset = Subset(ds, [0, 1])
    loader = DataLoader(subset, batch_size=2, shuffle=False, collate_fn=get_collator()())
    # This would KeyError pre-patch on transition-metal indices.
    batch = next(iter(loader))
    assert "sigma" in batch and "epsilon" in batch
    assert torch.isfinite(batch["sigma"]).all()
    assert torch.isfinite(batch["epsilon"]).all()

    # Element-specific, not a flat fallback. The collator combines pairs
    # (Lorentz-Berthelot), so each atom's own parameters sit on the diagonal.
    node_mask = batch["node_mask"].bool()
    seen = set()
    for row in range(node_mask.shape[0]):
        keep = node_mask[row]
        node_types = batch["node_type"][row][keep].tolist()
        sigmas = batch["sigma"][row][keep][:, keep].diagonal().tolist()
        epsilons = batch["epsilon"][row][keep][:, keep].diagonal().tolist()
        for node_type, sigma, epsilon in zip(node_types, sigmas, epsilons, strict=True):
            # The collator shifts node_type by 1 so 0 can mean padding.
            expected = UFF_LJ_PARAMETERS[int(node_type) - 1]
            assert sigma == pytest.approx(expected[0], rel=1e-5)
            assert epsilon == pytest.approx(expected[1], rel=1e-5)
            seen.add(int(node_type))
    assert max(seen) - 1 > 35, "the fixture should contain a metal beyond ReBind's own table"


def test_uff_table_agrees_with_rebinds_own_values() -> None:
    """Our table must extend upstream's, not restate it differently.

    ReBind defines its parameters inside the function body, so we read them out
    of the vendored source: over Z=1..36 the two tables have to match exactly, or
    swapping ours in would quietly change the published QM9 setup.
    """
    source = (
        Path(__file__).resolve().parents[1]
        / "external"
        / "ReBIND"
        / "models"
        / "modules"
        / "utils.py"
    ).read_text()
    start = source.index("lj_parameters = {")
    end = source.index("\n    }", start) + len("\n    }")
    upstream = ast.literal_eval(source[start + len("lj_parameters = ") : end])

    assert upstream, "failed to parse ReBind's LJ table"
    for node_type, params in upstream.items():
        assert UFF_LJ_PARAMETERS[node_type] == (params["sigma"], params["epsilon"])
    assert max(UFF_LJ_PARAMETERS) > max(upstream)
