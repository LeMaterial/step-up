"""Molecule-level charge / spin conditioning."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from step_up.data.csv_dataset import CSVMoleculeDataset
from step_up.models.rebind import N_GLOBAL_FEATURES, build_collator, build_rebind


def _dataset(path) -> CSVMoleculeDataset:
    return CSVMoleculeDataset(path, "mol2", charge_column="charge", spin_column="spinmult")


def _batch(path, n: int) -> dict:
    loader = DataLoader(
        Subset(_dataset(path), list(range(n))),
        batch_size=n,
        collate_fn=build_collator(conditioned=True),
    )
    return next(iter(loader))


def test_dataset_attaches_charge_and_spin(bostmc_path) -> None:
    df = pd.read_csv(bostmc_path)
    ds = _dataset(bostmc_path)
    assert [ds[i]["charge"] for i in range(len(ds))] == df["charge"].astype(float).tolist()
    assert [ds[i]["spin_multiplicity"] for i in range(len(ds))] == (
        df["spinmult"].astype(float).tolist()
    )


def test_collator_emits_charge_and_unpaired_electron_count(bostmc_path) -> None:
    df = pd.read_csv(bostmc_path)
    batch = _batch(bostmc_path, len(df))
    # Column 1 is unpaired electrons, i.e. multiplicity - 1: 0 for a singlet.
    expected = torch.tensor(
        [[float(c), float(s) - 1.0] for c, s in zip(df["charge"], df["spinmult"], strict=True)]
    )
    assert batch["global_features"].shape == (len(df), N_GLOBAL_FEATURES)
    torch.testing.assert_close(batch["global_features"], expected)


def test_conditioning_is_inert_at_init_then_changes_predictions(bostmc_path) -> None:
    """Zero-initialised output layer means a fresh model matches the unconditioned one."""
    batch = _batch(bostmc_path, 2)
    torch.manual_seed(0)
    model = build_rebind(
        n_layers=1, d_model=32, d_ffn=64, n_head=4, n_global_features=N_GLOBAL_FEATURES
    )
    model.eval()
    without_conditioning = {k: v for k, v in batch.items() if k != "global_features"}
    with torch.no_grad():
        conditioned = model(**batch)
        plain = model(**without_conditioning)
    torch.testing.assert_close(conditioned.conformer_hat, plain.conformer_hat)

    # Once the layer is non-zero, the molecule-level features reach the output.
    torch.nn.init.normal_(model.global_cond[-1].weight, std=0.5)
    with torch.no_grad():
        neutral = model(**{**batch, "global_features": torch.zeros_like(batch["global_features"])})
        charged = model(
            **{**batch, "global_features": torch.full_like(batch["global_features"], 2.0)}
        )
    assert not torch.allclose(neutral.conformer_hat, charged.conformer_hat)
