"""Shared pytest fixtures.

Tests run against tiny CSV fixtures committed under ``tests/fixtures/``, so the
suite behaves the same on every machine, including CI runners that don't have
the full QM9-full.csv / tmQMg-full.csv / BOSTMC datasets. A missing fixture is
an error rather than a skip: a skipped test would let CI pass without
exercising the featurization, MOL2 parser, LJ patch, or model forward pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> Path:
    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.fail(f"committed test fixture missing: {path}", pytrace=False)
    return path


@pytest.fixture(scope="session")
def qm9_path() -> Path:
    return _fixture("qm9_mini.csv")


@pytest.fixture(scope="session")
def qm9_sdf_path() -> Path:
    """QM9 records in ReBind's published form: ``mol_id, split, sdf`` molblocks.

    Five molecules copied from gdb9.sdf (the HuggingFace ``RichXuOvO/HFQm9``
    copy that ReBind and GTMGC used), carrying their published split labels.
    """
    return _fixture("qm9_sdf_mini.csv")


@pytest.fixture(scope="session")
def qm9_sdf_tokens_path() -> Path:
    """Mole-BERT token ids for the molecules in ``qm9_sdf_mini.csv``.

    Committed so the GTMGC tests don't need the tokenizer checkpoint; regenerate
    with ``scripts/tokenize_molebert.py``.
    """
    return _fixture("qm9_sdf_mini_molebert_tokens.csv")


@pytest.fixture(scope="session")
def tmqmg_path() -> Path:
    return _fixture("tmqmg_mini.csv")


@pytest.fixture(scope="session")
def bostmc_path() -> Path:
    """BOSTMC-shaped CSV with columns ``refcode, xyz, charge, spinmult, mol2``.

    The structures are the public tmQMg complexes from ``tmqmg_mini.csv`` with
    made-up ``refcode`` / ``charge`` / ``spinmult`` values (singlets, then
    doublets), so no BOSTMC data is committed to the repo.
    """
    return _fixture("bostmc_mini.csv")
