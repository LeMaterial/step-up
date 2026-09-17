"""Ring and aromaticity features from the direct MOL2 parser."""

from __future__ import annotations

from step_up.data.mol2 import mol2_to_graph_dict

# Column positions in ``node_attr`` (see the ``step_up.data.mol2`` docstring).
IS_AROMATIC = 7
IS_IN_RING = 8

# Synthetic complex: Co bound to a pyridine N, a chloride, and both carbons of an
# ethylene (eta-2). It has two independent cycles, the aromatic pyridine ring and
# the Co-C=C metallacycle, while the Co-N and Co-Cl bonds are bridges. Atom
# order: Co, N, C x5 (pyridine), Cl, C x2 (ethylene).
SYNTHETIC_COMPLEX = """\
@<TRIPOS>MOLECULE
synthetic
10 11 1
SMALL
NO_CHARGES

@<TRIPOS>ATOM
1 Co1 0.000 0.000 0.000 Co.oh
2 N1 2.000 0.000 0.000 N.ar
3 C1 2.700 1.200 0.000 C.ar
4 C2 4.100 1.200 0.000 C.ar
5 C3 4.800 0.000 0.000 C.ar
6 C4 4.100 -1.200 0.000 C.ar
7 C5 2.700 -1.200 0.000 C.ar
8 Cl1 -2.200 0.000 0.000 Cl
9 C6 0.000 1.400 1.400 C.2
10 C7 0.000 0.000 2.000 C.2
@<TRIPOS>BOND
1 1 2 1
2 2 3 ar
3 3 4 ar
4 4 5 ar
5 5 6 ar
6 6 7 ar
7 7 2 ar
8 1 8 1
9 1 9 1
10 1 10 1
11 9 10 2
"""


def test_ring_and_aromatic_flags() -> None:
    g = mol2_to_graph_dict(SYNTHETIC_COMPLEX)
    # "Co.oh" carries a geometry suffix; the element must still parse as Co (Z=27).
    assert g["node_type"][0] == 26
    assert [atom[IS_AROMATIC] for atom in g["node_attr"]] == [0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    # Ring membership comes from bridge detection, so the metallacycle counts as a
    # ring and the chloride (a bridge) does not.
    assert [atom[IS_IN_RING] for atom in g["node_attr"]] == [1, 1, 1, 1, 1, 1, 1, 0, 1, 1]

    # Bond features are [type, dir, stereo, is_conjugated], stored per direction.
    bonds = dict(zip(zip(*g["edge_index"], strict=True), g["edge_attr"], strict=True))
    assert bonds[(1, 2)] == bonds[(2, 1)] == [3, 0, 0, 1]  # aromatic N-C, conjugated
    assert bonds[(8, 9)] == [1, 0, 0, 0]  # C=C double
    assert bonds[(0, 7)] == [0, 0, 0, 0]  # Co-Cl single
