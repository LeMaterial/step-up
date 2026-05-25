"""Convert raw XYZ / MOL2 blocks into ReBind-format graph dicts.

Two paths:

- **QM9 (XYZ + SMILES column)**: uses RDKit's ``MolFromXYZBlock`` +
  ``rdDetermineBonds.DetermineBonds`` to recover the bond graph from
  coordinates. SMILES is intentionally ignored because the SMILES atom order
  doesn't match the XYZ order in QM9-full.csv.

- **Organometallic (MOL2 + XYZ)**: parses MOL2 directly via
  :mod:`step_up.data.mol2` — no RDKit perception. Connectivity and bond types
  come straight from the file, and SYBYL atom types (``C.3``, ``C.ar``,
  ``N.am``, ...) provide hybridization / aromaticity that RDKit with
  ``sanitize=False`` would discard. Rings are computed from the bond graph.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from .mol2 import mol2_to_graph_dict

# Load ReBind's `data/utils.py` directly so we share the canonical featurization
# without putting the entire ReBIND tree on sys.path. The submodule lives at
# step-up/external/ReBIND/data/utils.py.
_REBIND_ROOT = Path(__file__).resolve().parents[3] / "external" / "ReBIND"


def _load_rebind_data_utils():
    path = _REBIND_ROOT / "data" / "utils.py"
    if not path.exists():
        raise FileNotFoundError(
            f"ReBind submodule not found at {_REBIND_ROOT}. "
            "Run: git submodule update --init --recursive"
        )
    spec = importlib.util.spec_from_file_location("_step_up_rebind_data_utils", str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_step_up_rebind_data_utils"] = mod
    spec.loader.exec_module(mod)
    return mod


_rebind_utils = _load_rebind_data_utils()
mol_to_graph_dict = _rebind_utils.mol_to_graph_dict
ALLOWABLE_FEATURES = _rebind_utils.ALLOWABLE_FEATURES


def parse_xyz_block(block: str) -> tuple[list[str], np.ndarray]:
    """Parse a standard XYZ block string into (symbols, coords).

    Standard XYZ format:
        line 1: integer atom count
        line 2: comment (ignored)
        lines 3..: ``element x y z`` (whitespace-separated)
    """
    lines = block.strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ block too short ({len(lines)} lines)")
    n = int(lines[0].strip())
    atom_lines = lines[2 : 2 + n]
    if len(atom_lines) != n:
        raise ValueError(f"XYZ header says {n} atoms but body has {len(atom_lines)}")
    symbols: list[str] = []
    coords = np.empty((n, 3), dtype=np.float64)
    for i, line in enumerate(atom_lines):
        parts = line.split()
        symbols.append(parts[0])
        coords[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return symbols, coords


def mol_from_xyz_block(xyz_block: str, charge: int = 0) -> Chem.Mol:
    """Build an RDKit molecule from a raw XYZ block, inferring bonds from coords.

    Atom order matches the XYZ block (no remapping). Bond *orders* are inferred
    via ``rdDetermineBonds.DetermineBonds`` using the supplied formal charge.
    Raises ``ValueError`` if bond inference fails — callers (typically the CSV
    dataset's validation pass) are responsible for filtering bad rows out.

    Suitable for closed-shell organic systems (QM9). For organometallics we use
    the MOL2 path instead, which carries explicit bonds.

    NOTE: An earlier version of this function fell back to
    ``DetermineConnectivity`` (bond presence without bond orders). This was a
    silent data-corruption bug: ReBind's ``safe_index`` maps unspecified bond
    types to the AROMATIC bucket, so every bond on those rows was being
    featurized as aromatic. We now raise instead, so the bad rows get
    explicitly filtered upstream.
    """
    mol = Chem.MolFromXYZBlock(xyz_block)
    if mol is None:
        raise ValueError("RDKit failed to parse XYZ block")
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    return mol


def featurize_xyz(xyz_block: str, charge: int = 0) -> dict[str, Any]:
    """One-shot helper: XYZ-only featurization (QM9 path).

    Coordinates come from the XYZ block; bonds are inferred. The original
    SMILES is unused — RDKit's ``DetermineBonds`` recovers the same chemistry
    and the resulting atom order matches the XYZ.
    """
    mol = mol_from_xyz_block(xyz_block, charge=charge)
    return mol_to_graph_dict(mol)


def featurize_mol2_xyz(mol2_block: str, xyz_block: str) -> dict[str, Any]:
    """One-shot helper for MOL2-sourced rows (tmQMg, BOSTMC).

    No RDKit — the MOL2 file already contains all connectivity and bond types
    we need, and we keep the XYZ-block coordinates as the canonical geometry.
    """
    _, coords = parse_xyz_block(xyz_block)
    return mol2_to_graph_dict(mol2_block, coords=coords)
