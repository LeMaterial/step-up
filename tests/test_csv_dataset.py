"""Verify each CSV path produces valid ReBind graph dicts on a small subset."""

from __future__ import annotations

import pandas as pd
import pytest

from step_up.data import featurize
from step_up.data.csv_dataset import CSVMoleculeDataset
from step_up.data.splits import split_by_labels


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


def test_sdf_source_uses_published_bonds_and_split(qm9_sdf_path) -> None:
    """The sdf source featurizes molblocks and can read a published split column."""
    ds = CSVMoleculeDataset(qm9_sdf_path, "sdf", id_column="mol_id", split_column="split")
    df = pd.read_csv(qm9_sdf_path)
    assert len(ds) == len(df)
    for i in range(len(ds)):
        _check_graph_dict(ds[i])
    assert ds.split_keys() == df["mol_id"].tolist()
    assert ds.split_labels() == df["split"].tolist()

    train, val, test = split_by_labels(ds, ds.split_labels())
    assert (len(train), len(val), len(test)) == (2, 2, 1)
    # Methane from gdb9.sdf: 4 C-H bonds, stored as two directed edges each.
    methane = ds[0]
    assert methane["num_nodes"] == 5
    assert methane["num_edges"] == 8


# The numH atom feature is column 4 of ReBind's 9-column atom feature vector,
# and the degree is column 2.
_NUMH_COLUMN = 4
_DEGREE_COLUMN = 2


@pytest.mark.parametrize(
    ("fixture_name", "source"),
    [("qm9_sdf_path", "sdf"), ("tmqmg_path", "mol2"), ("bostmc_path", "mol2")],
)
def test_remove_hs_drops_hydrogens_from_every_source(fixture_name, source, request) -> None:
    """Training without hydrogens has to work the same on both featurization paths."""
    path = request.getfixturevalue(fixture_name)
    full = CSVMoleculeDataset(path, source=source)
    heavy = CSVMoleculeDataset(path, source=source, remove_hs=True)
    assert len(full) == len(heavy)

    saw_hydrogen = False
    for i in range(len(full)):
        before, after = full[i], heavy[i]
        _check_graph_dict(after)
        n_h = sum(1 for z in before["node_type"] if z == 0)  # node_type is Z - 1
        saw_hydrogen = saw_hydrogen or n_h > 0
        assert after["num_nodes"] == before["num_nodes"] - n_h
        assert 0 not in after["node_type"], "a hydrogen survived the strip"
        # Bonds to a dropped atom go too, and the survivors stay in range.
        assert all(idx < after["num_nodes"] for side in after["edge_index"] for idx in side)
        # Hydrogen count survives as an atom feature, so the graph still knows
        # how many there were — only where they were is gone. (RDKit reports 0
        # while the hydrogens are explicit atoms and fills the count in once they
        # are removed, so this is checked against the bond graph, not against the
        # feature in `before`.)
        attached = [0] * before["num_nodes"]
        for a, b in zip(*before["edge_index"], strict=True):
            if before["node_type"][b] == 0:
                attached[a] += 1
        assert [row[_NUMH_COLUMN] for row in after["node_attr"]] == [
            count for count, z in zip(attached, before["node_type"], strict=True) if z != 0
        ]
    assert saw_hydrogen, "fixture has no hydrogens, so this proves nothing"


def test_remove_hs_reports_the_heavy_atom_degree(qm9_sdf_path) -> None:
    """Degree must count surviving bonds, or it would disagree with the edges."""
    heavy = CSVMoleculeDataset(qm9_sdf_path, source="sdf", remove_hs=True)
    for i in range(len(heavy)):
        graph = heavy[i]
        counted = [0] * graph["num_nodes"]
        for node in graph["edge_index"][0]:
            counted[node] += 1
        assert [row[_DEGREE_COLUMN] for row in graph["node_attr"]] == counted


def test_remove_hs_keeps_rdkit_mol_aligned_with_the_graph(qm9_sdf_path) -> None:
    """C-RMSD writes predicted coordinates onto this molecule by atom index.

    If it still carried hydrogens while the graph did not, every coordinate
    after the first hydrogen would land on the wrong atom.
    """
    heavy = CSVMoleculeDataset(qm9_sdf_path, source="sdf", id_column="mol_id", remove_hs=True)
    for i in range(len(heavy)):
        assert heavy.rdkit_mol(i).GetNumAtoms() == heavy[i]["num_nodes"]

    full = CSVMoleculeDataset(qm9_sdf_path, source="sdf", id_column="mol_id")
    assert full.rdkit_mol(0).GetNumAtoms() == full[0]["num_nodes"]
