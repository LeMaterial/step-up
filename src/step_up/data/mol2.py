"""Direct Tripos MOL2 parser — no RDKit perception.

For organometallic data we already have the connectivity and bond types in the
MOL2 file (from molSimplify), and we have SYBYL atom types that encode
hybridization and aromaticity. Going through ``Chem.MolFromMol2Block`` with
``sanitize=False`` would give us nothing beyond what's already in the file
while throwing the SYBYL information away — so we parse MOL2 ourselves and
emit a graph dict in the same shape ReBind's collator expects.

Atom feature semantics match ``ALLOWABLE_FEATURES`` in
``external/ReBIND/data/utils.py``:

    [atomic_num_idx, chirality_idx, degree_idx, formal_charge_idx,
     numH_idx, num_radical_idx, hybridization_idx, is_aromatic_idx,
     is_in_ring_idx]

Bond features (4 per bond, undirected expanded to two directed entries):

    [bond_type_idx, bond_dir_idx, bond_stereo_idx, is_conjugated_idx]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Element table (Z=1..103). Kept inline to avoid a periodic-table dep.
# ---------------------------------------------------------------------------
_ELEMENTS: tuple[str, ...] = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr",
)  # fmt: skip
_SYMBOL_TO_Z: dict[str, int] = {sym: i + 1 for i, sym in enumerate(_ELEMENTS)}

# ---------------------------------------------------------------------------
# ReBind vocab indices (mirroring ALLOWABLE_FEATURES). Keeping these inline so
# the file is independent — if upstream changes, the smoke tests will catch
# the mismatch in feature dimensions.
# ---------------------------------------------------------------------------
_POSSIBLE_ATOMIC_NUM_LEN = 118  # list(range(1, 119))
_POSSIBLE_DEGREE = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)  # plus "misc" -> idx 11
_POSSIBLE_FORMAL_CHARGE = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)  # +"misc"->11
_POSSIBLE_NUMH = (0, 1, 2, 3, 4, 5, 6, 7, 8)  # +"misc"->9

# Chirality, hybridization, bond_type, bond_dir, bond_stereo indices line up
# with ReBind's ALLOWABLE_FEATURES tuples. We only use specific entries:
_CHI_UNSPECIFIED = 0
_HYB = {"SP": 0, "SP2": 1, "SP3": 2, "SP3D": 3, "SP3D2": 4, "misc": 5}
_BOND_TYPE = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
_BOND_DIR_NONE = 0
_BOND_STEREO_NONE = 0

# SYBYL atom type -> (hybridization name, is_aromatic)
_SYBYL_HYB: dict[str, tuple[str, bool]] = {
    "1": ("SP", False),
    "2": ("SP2", False),
    "3": ("SP3", False),
    "4": ("SP3", False),  # quaternary N -> SP3
    "ar": ("SP2", True),  # aromatic
    "am": ("SP2", False),  # amide N
    "pl3": ("SP2", False),  # planar N
    "cat": ("SP2", False),  # guanidinium C
    "co2": ("SP2", False),  # carboxylate O
    "spc": ("SP3", False),  # SPC water O
    "t3p": ("SP3", False),  # TIP3P water O
    "o": ("SP3", False),  # generic
}

# MOL2 bond-type string -> (canonical bond_type_name, is_conjugated)
_MOL2_BOND_TYPE: dict[str, tuple[str, bool]] = {
    "1": ("SINGLE", False),
    "2": ("DOUBLE", False),
    "3": ("TRIPLE", False),
    "ar": ("AROMATIC", True),
    "am": ("SINGLE", True),  # amide C-N -> single but conjugated
    "du": ("SINGLE", False),  # dummy
    "un": ("SINGLE", False),  # unknown -> treat as single
    "nc": ("SINGLE", False),  # not connected (shouldn't appear, but)
}


@dataclass
class Mol2Atom:
    """One row from the ``@<TRIPOS>ATOM`` block."""

    idx: int  # 0-indexed
    element: str  # parsed from SYBYL type prefix
    sybyl: str  # full SYBYL string (e.g., "C.3", "Ni")
    coords: np.ndarray  # (3,)


@dataclass
class Mol2Bond:
    """One row from the ``@<TRIPOS>BOND`` block."""

    a: int  # 0-indexed
    b: int  # 0-indexed
    mol2_type: str  # raw MOL2 type string


@dataclass
class Mol2:
    atoms: list[Mol2Atom]
    bonds: list[Mol2Bond]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _element_from_sybyl(sybyl: str) -> str:
    """Element symbol = SYBYL prefix before the first dot (or full string)."""
    return sybyl.split(".", 1)[0]


def _hyb_from_sybyl(sybyl: str) -> tuple[str, bool]:
    """SYBYL → (hybridization, is_aromatic). Metals/unknowns default to misc."""
    if "." not in sybyl:
        # No SYBYL hybridization given (e.g. metals: "Ni", "Fe"). misc bucket.
        return ("misc", False)
    suffix = sybyl.split(".", 1)[1]
    return _SYBYL_HYB.get(suffix, ("misc", False))


def parse_mol2(text: str) -> Mol2:
    """Parse a single-molecule MOL2 block string."""
    atoms: list[Mol2Atom] = []
    bonds: list[Mol2Bond] = []
    section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("@<TRIPOS>"):
            section = line[len("@<TRIPOS>") :].strip().upper()
            continue
        if section == "ATOM":
            atoms.append(_parse_atom_line(line, len(atoms)))
        elif section == "BOND":
            bonds.append(_parse_bond_line(line))
        # Other sections (MOLECULE header, SUBSTRUCTURE, etc.) are ignored.
    return Mol2(atoms=atoms, bonds=bonds)


def _parse_atom_line(line: str, idx0: int) -> Mol2Atom:
    """ATOM line: id name x y z sybyl_type [subst_id [subst_name [charge]]]."""
    parts = line.split()
    if len(parts) < 6:
        raise ValueError(f"MOL2 ATOM line too short: {line!r}")
    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
    sybyl = parts[5]
    element = _element_from_sybyl(sybyl)
    return Mol2Atom(
        idx=idx0,
        element=element,
        sybyl=sybyl,
        coords=np.array([x, y, z], dtype=np.float64),
    )


def _parse_bond_line(line: str) -> Mol2Bond:
    """BOND line: id atom_a atom_b type."""
    parts = line.split()
    if len(parts) < 4:
        raise ValueError(f"MOL2 BOND line too short: {line!r}")
    a = int(parts[1]) - 1  # MOL2 is 1-indexed
    b = int(parts[2]) - 1
    return Mol2Bond(a=a, b=b, mol2_type=parts[3])


# ---------------------------------------------------------------------------
# Graph-dict construction
# ---------------------------------------------------------------------------


def _safe_index(seq: tuple, value: Any, miss_idx: int) -> int:
    try:
        return seq.index(value)
    except ValueError:
        return miss_idx


def _tarjan_bridges(adj: list[list[tuple[int, int]]]) -> set[int]:
    """Return the set of bridge edge-ids in an undirected graph.

    A bridge is an edge whose removal disconnects the graph. The implementation
    is iterative Tarjan with explicit ``disc``/``low`` arrays.
    """
    n = len(adj)
    disc = [-1] * n
    low = [0] * n
    bridges: set[int] = set()
    timer = 0
    for start in range(n):
        if disc[start] != -1:
            continue
        # Iterative DFS rooted at ``start``. Stack entries are
        # (node, parent_edge_id, neighbor_iter_index).
        stack: list[list[int]] = [[start, -1, 0]]
        disc[start] = low[start] = timer
        timer += 1
        while stack:
            node, pedge, i = stack[-1]
            if i >= len(adj[node]):
                stack.pop()
                if not stack:
                    break
                parent = stack[-1][0]
                low[parent] = min(low[parent], low[node])
                if low[node] > disc[parent]:
                    bridges.add(pedge)
                continue
            stack[-1][2] = i + 1
            nbr, eid = adj[node][i]
            if eid == pedge:
                continue
            if disc[nbr] == -1:
                disc[nbr] = low[nbr] = timer
                timer += 1
                stack.append([nbr, eid, 0])
            else:
                low[node] = min(low[node], disc[nbr])
    return bridges


def _find_ring_atoms_and_bonds(
    n_atoms: int, bond_pairs: list[tuple[int, int]]
) -> tuple[set[int], set[frozenset[int]]]:
    """Return (atoms_in_any_ring, bonds_in_any_ring).

    An edge is a *bridge* iff its removal disconnects the graph, equivalently
    iff it is not part of any cycle. The complement of bridges is the set of
    ring bonds.
    """
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n_atoms)]
    for eid, (a, b) in enumerate(bond_pairs):
        adj[a].append((b, eid))
        adj[b].append((a, eid))
    bridges = _tarjan_bridges(adj)
    ring_edge_set: set[frozenset[int]] = set()
    ring_atom_set: set[int] = set()
    for eid, (a, b) in enumerate(bond_pairs):
        if eid not in bridges:
            ring_edge_set.add(frozenset((a, b)))
            ring_atom_set.add(a)
            ring_atom_set.add(b)
    return ring_atom_set, ring_edge_set


def mol2_to_graph_dict(mol2_text: str, coords: np.ndarray | None = None) -> dict[str, Any]:
    """Build a ReBind-format graph dict directly from a MOL2 block.

    If ``coords`` is supplied (e.g. parsed from a separate XYZ column),
    it overrides the MOL2 coordinates — useful when the canonical geometry
    lives in a different column. Otherwise we use the MOL2 atom positions.
    """
    parsed = parse_mol2(mol2_text)
    n = len(parsed.atoms)
    if n == 0:
        raise ValueError("MOL2 block has zero atoms")

    # ---- Coordinates ------------------------------------------------------
    if coords is not None:
        if coords.shape != (n, 3):
            raise ValueError(f"coords shape {coords.shape} does not match MOL2 atom count {n}")
        conformer = coords.tolist()
    else:
        conformer = np.stack([a.coords for a in parsed.atoms], axis=0).tolist()

    # ---- Bonds first (need degree, numH, in-ring) -------------------------
    bond_pairs: list[tuple[int, int]] = []
    bond_types: list[str] = []
    bond_conjugated: list[bool] = []
    for b in parsed.bonds:
        canonical_type, is_conj = _MOL2_BOND_TYPE.get(b.mol2_type, ("SINGLE", False))
        bond_pairs.append((b.a, b.b))
        bond_types.append(canonical_type)
        bond_conjugated.append(is_conj)

    ring_atoms, _ring_bonds = _find_ring_atoms_and_bonds(n, bond_pairs)
    degree = [0] * n
    num_h_neighbors = [0] * n
    for a, b in bond_pairs:
        degree[a] += 1
        degree[b] += 1
        if parsed.atoms[b].element == "H":
            num_h_neighbors[a] += 1
        if parsed.atoms[a].element == "H":
            num_h_neighbors[b] += 1

    # ---- Atom features ----------------------------------------------------
    node_attr: list[list[int]] = []
    node_type: list[int] = []
    for atom in parsed.atoms:
        z = _SYMBOL_TO_Z.get(atom.element)
        if z is None:
            raise ValueError(f"Unknown element symbol {atom.element!r} in MOL2")
        atom_num_idx = z - 1  # index into list(range(1, 119))
        hyb_name, is_aromatic = _hyb_from_sybyl(atom.sybyl)
        in_ring = atom.idx in ring_atoms
        feat = [
            atom_num_idx,
            _CHI_UNSPECIFIED,  # MOL2 has no chirality field
            _safe_index(_POSSIBLE_DEGREE, degree[atom.idx], len(_POSSIBLE_DEGREE)),
            _safe_index(_POSSIBLE_FORMAL_CHARGE, 0, len(_POSSIBLE_FORMAL_CHARGE)),
            _safe_index(_POSSIBLE_NUMH, num_h_neighbors[atom.idx], len(_POSSIBLE_NUMH)),
            0,  # num radical electrons (MOL2 has no radical info)
            _HYB.get(hyb_name, _HYB["misc"]),
            int(is_aromatic),
            int(in_ring),
        ]
        node_attr.append(feat)
        node_type.append(atom_num_idx)

    # ---- Bond features (bidirectional) ------------------------------------
    edges_a: list[int] = []
    edges_b: list[int] = []
    edge_attr: list[list[int]] = []
    for (a, b), btype, conj in zip(bond_pairs, bond_types, bond_conjugated, strict=True):
        feat = [
            _BOND_TYPE.get(btype, _BOND_TYPE["SINGLE"]),
            _BOND_DIR_NONE,
            _BOND_STEREO_NONE,
            int(conj),
        ]
        edges_a.append(a)
        edges_b.append(b)
        edge_attr.append(feat)
        edges_a.append(b)
        edges_b.append(a)
        edge_attr.append(feat)

    edge_index = [edges_a, edges_b]
    edge_type = [feat[0] for feat in edge_attr]
    edge_dir_type = [feat[1] for feat in edge_attr]

    return {
        "node_attr": node_attr,
        "edge_attr": edge_attr,
        "edge_index": edge_index,
        "node_type": node_type,
        "node_chiral_type": [feat[1] for feat in node_attr],
        "edge_type": edge_type,
        "edge_dire_type": edge_dir_type,
        "num_nodes": n,
        "num_edges": len(edge_attr),
        "conformer": conformer,
    }
