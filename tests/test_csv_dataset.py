"""Verify each CSV path produces valid ReBind graph dicts on a small subset."""

from __future__ import annotations

import pandas as pd
import pytest

from step_up.data import featurize
from step_up.data.csv_dataset import CSVMoleculeDataset


def _check_graph_dict(g: dict) -> None:
    required = {"node_attr", "edge_attr", "edge_index", "node_type", "conformer"}
    missing = required - g.keys()
    assert not missing, f"graph dict missing keys: {missing}"
    n = g["num_nodes"]
    assert n > 0
    assert len(g["node_type"]) == n
    assert len(g["node_attr"]) == n
    assert len(g["conformer"]) == n
    # edge_index shape: 2 x num_edges (lists)
    assert len(g["edge_index"]) == 2
    assert len(g["edge_index"][0]) == len(g["edge_index"][1])


def test_qm9_smoke(qm9_path) -> None:
    ds = CSVMoleculeDataset(qm9_path, source="smiles", subset_size=5)
    assert len(ds) == 5
    for i in range(len(ds)):
        _check_graph_dict(ds[i])


def test_tmqmg_smoke(tmqmg_path) -> None:
    ds = CSVMoleculeDataset(tmqmg_path, source="mol2", subset_size=5)
    assert len(ds) == 5
    elements_seen: set[int] = set()
    for i in range(len(ds)):
        g = ds[i]
        _check_graph_dict(g)
        elements_seen.update(g["node_type"])
    # tmQMg always includes a transition metal. node_type is Z-1, so 3d/4d/5d
    # metals show up at indices >= 20. Confirm at least one heavy element
    # made it through the direct MOL2 parser without being silently dropped.
    assert max(elements_seen) >= 20, f"no heavy elements seen: {sorted(elements_seen)}"


def test_bostmc_smoke(bostmc_path) -> None:
    ds = CSVMoleculeDataset(bostmc_path, source="mol2")
    elements_seen: set[int] = set()
    for i in range(len(ds)):
        g = ds[i]
        _check_graph_dict(g)
        elements_seen.update(g["node_type"])
    # Every complex in the fixture contains a d-block metal.
    assert max(elements_seen) >= 20, f"no transition metals seen: {sorted(elements_seen)}"


def test_bostmc_spin_filter(bostmc_path) -> None:
    """The filter_column knob should leave only matching rows in the dataset."""
    unfiltered = CSVMoleculeDataset(bostmc_path, source="mol2", validate=False)
    assert len(unfiltered) > 0
    singlets = CSVMoleculeDataset(
        bostmc_path, source="mol2", filter_column="spinmult", filter_value=1
    )
    doublets = CSVMoleculeDataset(
        bostmc_path, source="mol2", filter_column="spinmult", filter_value=2
    )
    # The mini fixture is built to contain both — filter must split them.
    assert len(singlets) > 0
    assert len(doublets) > 0
    assert len(singlets) + len(doublets) <= len(unfiltered)


def test_missing_submodule_error_is_not_swallowed(qm9_path, monkeypatch, tmp_path) -> None:
    """Without the ReBind checkout, validation must surface the submodule error.

    Otherwise every row is counted as a featurization failure and the user sees
    a misleading "Validation dropped all rows" error instead.
    """
    monkeypatch.setattr(featurize, "_rebind_utils", None)
    monkeypatch.setattr(featurize, "_REBIND_UTILS_PATH", tmp_path / "missing" / "utils.py")
    with pytest.raises(FileNotFoundError, match="git submodule update --init --recursive"):
        CSVMoleculeDataset(qm9_path, source="smiles", subset_size=2)


def test_split_keys_come_from_the_csv_not_dataset_positions(bostmc_path) -> None:
    """Keys are CSV row numbers (or id_column values), unchanged by filtering."""
    df = pd.read_csv(bostmc_path)
    doublet_rows = df.index[df["spinmult"] == 2]
    assert doublet_rows[0] > 0, "fixture should have doublets after the first row"
    by_row = CSVMoleculeDataset(bostmc_path, "mol2", filter_column="spinmult", filter_value=2)
    assert by_row.split_keys() == [str(i) for i in doublet_rows]
    by_id = CSVMoleculeDataset(
        bostmc_path, "mol2", filter_column="spinmult", filter_value=2, id_column="refcode"
    )
    assert by_id.split_keys() == df.loc[doublet_rows, "refcode"].tolist()
