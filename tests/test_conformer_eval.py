"""Conformer metrics in ReBind's published protocol."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Subset

from step_up.data.csv_dataset import CSVMoleculeDataset
from step_up.eval.conformer_eval import evaluate_split, kabsch_rmsd
from step_up.models.rebind import build_rebind, get_collator


class _PerfectModel(torch.nn.Module):
    """Predicts the target conformer exactly."""

    def forward(self, **batch):
        conformer = batch["conformer"]
        return SimpleNamespace(conformer_hat=conformer, conformer=conformer)


def test_perfect_prediction_scores_zero(qm9_sdf_path) -> None:
    ds = CSVMoleculeDataset(qm9_sdf_path, "sdf")
    subset = Subset(ds, list(range(len(ds))))
    metrics = evaluate_split(_PerfectModel(), ds, subset, get_collator()(), batch_size=2)
    assert metrics["n_molecules"] == len(ds)
    # Every fixture molecule reaches the RDKit C-RMSD path.
    assert metrics["n_rmsd_molecules"] == len(ds)
    assert metrics["n_rmsd_failures"] == 0
    assert metrics["d_mae"] == 0.0
    assert metrics["d_rmse"] == 0.0
    assert metrics["c_rmsd"] < 1e-6


def test_untrained_model_gives_finite_metrics(qm9_sdf_path) -> None:
    ds = CSVMoleculeDataset(qm9_sdf_path, "sdf")
    subset = Subset(ds, list(range(len(ds))))
    torch.manual_seed(0)
    model = build_rebind(n_layers=1, d_model=32, d_ffn=64, n_head=4)
    metrics = evaluate_split(model, ds, subset, get_collator()(), batch_size=2)
    assert metrics["d_mae"] > 0.0
    assert all(math.isfinite(metrics[key]) for key in ("d_mae", "d_rmse", "c_rmsd"))


def test_mol2_rows_fall_back_to_aligned_coordinate_rmsd(bostmc_path) -> None:
    """Organometallic rows have no RDKit molecule, so C-RMSD uses aligned coordinates."""
    ds = CSVMoleculeDataset(bostmc_path, "mol2")
    subset = Subset(ds, list(range(len(ds))))
    metrics = evaluate_split(_PerfectModel(), ds, subset, get_collator()(), batch_size=2)
    assert metrics["c_rmsd_method"] == "aligned_coords"
    assert metrics["n_rmsd_molecules"] == len(ds)
    assert metrics["d_mae"] == 0.0
    assert metrics["c_rmsd"] < 1e-6


def test_forcing_aligned_coords_skips_rdkit(qm9_sdf_path) -> None:
    """QM9 rows do build RDKit molecules, so the option has to override that.

    Forcing the plain metric is what makes RMSD comparable with the MOL2 sets.
    """
    ds = CSVMoleculeDataset(qm9_sdf_path, "sdf")
    subset = Subset(ds, list(range(len(ds))))
    auto = evaluate_split(_PerfectModel(), ds, subset, get_collator()(), batch_size=2)
    forced = evaluate_split(
        _PerfectModel(), ds, subset, get_collator()(), batch_size=2, rmsd_method="aligned_coords"
    )
    assert auto["c_rmsd_method"] == "rdkit_bestrms"
    assert forced["c_rmsd_method"] == "aligned_coords"
    assert forced["n_rmsd_molecules"] == len(ds)

    with pytest.raises(ValueError, match="Unknown rmsd_method"):
        evaluate_split(_PerfectModel(), ds, subset, get_collator()(), rmsd_method="kabsch")


def test_aligned_coords_rmsd_can_drop_hydrogens(bostmc_path) -> None:
    """The MOL2 path has no RDKit molecule, so it must drop Hs by atom type.

    Organometallic RMSD was being reported over every atom while the QM9 C-RMSD
    excluded hydrogen, which is not a comparison.
    """
    ds = CSVMoleculeDataset(bostmc_path, "mol2")
    subset = Subset(ds, list(range(len(ds))))
    torch.manual_seed(0)
    model = build_rebind(n_layers=1, d_model=32, d_ffn=64, n_head=4)
    heavy = evaluate_split(model, ds, subset, get_collator()(), batch_size=2, remove_hs=True)
    everything = evaluate_split(model, ds, subset, get_collator()(), batch_size=2, remove_hs=False)

    assert heavy["c_rmsd_method"] == everything["c_rmsd_method"] == "aligned_coords"
    assert heavy["c_rmsd"] != everything["c_rmsd"], "the fixture molecules have hydrogens"
    # Dropping atoms changes only the RMSD; the distance metrics pool over all
    # atoms either way, as ReBind's evaluate.py does.
    assert heavy["d_mae"] == everything["d_mae"]
    assert heavy["d_rmse"] == everything["d_rmse"]


def test_kabsch_rmsd_is_invariant_to_rigid_motion() -> None:
    """A rotated and translated copy of a structure has zero RMSD."""
    torch.manual_seed(0)
    target = torch.randn(12, 3, dtype=torch.float64)
    angle = torch.tensor(0.7, dtype=torch.float64)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    moved = (rotation @ target.T).T + torch.tensor([3.0, -1.0, 2.0], dtype=torch.float64)
    assert kabsch_rmsd(moved, target) < 1e-10
    # A reflection is not a rotation, so it must not be aligned away.
    mirrored = target * torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64)
    assert kabsch_rmsd(mirrored, target) > 0.1
    # Displacing every atom by a fixed amount along one axis shows up in full.
    assert kabsch_rmsd(target + torch.tensor([0.0, 0.0, 0.0]), target) < 1e-10
