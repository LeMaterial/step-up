"""The per-molecule structure dump: one CSV row per test molecule."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from step_up.models import N_GLOBAL_FEATURES, build_model

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dump_structures.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("_dump_structures", str(_SCRIPT))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dump_structures"] = module
    spec.loader.exec_module(module)
    return module


def _write_config(tmp_path: Path, dataset_path: Path, **overrides) -> Path:
    config = {
        "dataset_path": str(dataset_path),
        "dataset_source": "mol2",
        "id_column": "refcode",
        "charge_column": "charge",
        "spin_column": "spinmult",
        "split_ratios": [0.4, 0.3, 0.3],
        "split_seed": 0,
        "n_layers": 1,
        "d_model": 32,
        "d_ffn": 64,
        "n_head": 4,
        "epochs": 1,
        "batch_size": 2,
        "eval_batch_size": 2,
        "num_workers": 0,
        "device": "cpu",
        "output_dir": str(tmp_path / "run"),
        "seed": 0,
    }
    config.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _untrained_checkpoint(tmp_path: Path, conditioned: bool) -> Path:
    torch.manual_seed(0)
    model = build_model(
        "rebind",
        n_layers=1,
        d_model=32,
        d_ffn=64,
        n_head=4,
        n_global_features=N_GLOBAL_FEATURES if conditioned else 0,
    )
    path = tmp_path / "run" / "best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def _parse_xyz(block: str) -> tuple[list[str], list[list[float]]]:
    lines = block.splitlines()
    count = int(lines[0])
    symbols, coords = [], []
    for line in lines[2 : 2 + count]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(v) for v in parts[1:4]])
    assert len(symbols) == count, "count line disagrees with the number of atom lines"
    return symbols, coords


def test_dump_writes_one_row_per_molecule_with_both_structures(bostmc_path, tmp_path) -> None:
    module = _load_script()
    config_path = _write_config(tmp_path, bostmc_path)
    _untrained_checkpoint(tmp_path, conditioned=True)

    assert module.main(["-c", str(config_path), "--split", "test"]) == 0
    out = tmp_path / "run" / "structures_test.csv"
    frame = pd.read_csv(out)
    assert list(frame.columns) == list(module._COLUMNS)
    assert len(frame) > 0

    for _, row in frame.iterrows():
        true_symbols, true_coords = _parse_xyz(row["xyz_true"])
        pred_symbols, pred_coords = _parse_xyz(row["xyz_pred"])
        # Same atoms, same order, so the two blocks overlay directly.
        assert true_symbols == pred_symbols
        assert len(true_coords) == len(pred_coords) == row["n_atoms"]
        assert row["elements"].split() == true_symbols
        # Real coordinates, not padding zeros.
        assert any(any(v != 0.0 for v in xyz) for xyz in true_coords)
        assert all(row[key] >= 0.0 for key in ("d_mae", "d_rmse", "rmsd"))
        # BOSTMC states both, so neither may be blank.
        assert not pd.isna(row["charge"])
        assert not pd.isna(row["spin_multiplicity"])


def test_dump_records_the_true_charge_and_spin(bostmc_path, tmp_path) -> None:
    """The point of the file is downstream analysis, so these must be the data's."""
    module = _load_script()
    config_path = _write_config(tmp_path, bostmc_path)
    _untrained_checkpoint(tmp_path, conditioned=True)
    module.main(["-c", str(config_path), "--split", "test"])

    dumped = pd.read_csv(tmp_path / "run" / "structures_test.csv").set_index("id")
    source = pd.read_csv(bostmc_path).set_index("refcode")
    for key, row in dumped.iterrows():
        assert row["charge"] == pytest.approx(float(source.loc[key, "charge"]))
        assert row["spin_multiplicity"] == pytest.approx(float(source.loc[key, "spinmult"]))


def test_dump_leaves_charge_and_spin_blank_when_the_data_is_silent(qm9_sdf_path, tmp_path) -> None:
    """An unconditioned dataset must not report a charge it was never given."""
    module = _load_script()
    config_path = _write_config(
        tmp_path,
        qm9_sdf_path,
        dataset_source="sdf",
        id_column="mol_id",
        charge_column=None,
        spin_column=None,
    )
    _untrained_checkpoint(tmp_path, conditioned=False)
    module.main(["-c", str(config_path), "--split", "test"])

    with open(tmp_path / "run" / "structures_test.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        assert row["charge"] == ""
        assert row["spin_multiplicity"] == ""


def test_predicted_block_is_rigidly_aligned_not_rebuilt(bostmc_path, tmp_path) -> None:
    """Alignment may rotate and translate the prediction, never reshape it."""
    module = _load_script()
    torch.manual_seed(0)
    pred = torch.randn(10, 3, dtype=torch.float64)
    target = torch.randn(10, 3, dtype=torch.float64)
    moved = module._aligned(pred, target)
    torch.testing.assert_close(torch.cdist(moved, moved), torch.cdist(pred, pred))
    # And it lands on the target's centroid rather than the origin.
    torch.testing.assert_close(moved.mean(dim=0), target.mean(dim=0))
