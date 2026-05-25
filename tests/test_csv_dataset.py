"""Verify each CSV path produces valid ReBind graph dicts on a small subset."""

from __future__ import annotations

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
    ds = CSVMoleculeDataset(tmqmg_path, source="mol2", subset_size=20)
    assert len(ds) == 20
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
    ds = CSVMoleculeDataset(bostmc_path, source="mol2", subset_size=5)
    assert len(ds) == 5
    elements_seen: set[int] = set()
    for i in range(len(ds)):
        g = ds[i]
        _check_graph_dict(g)
        elements_seen.update(g["node_type"])
    # BOSTMC is d-block-only by construction.
    assert max(elements_seen) >= 20, f"no transition metals seen: {sorted(elements_seen)}"
