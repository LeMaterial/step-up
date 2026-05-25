"""Shared pytest fixtures and skip-paths for tests that need the local CSVs."""

from __future__ import annotations

from pathlib import Path

import pytest

QM9_PATH = Path("/home/gridsan/jtoney/ElemNet/benchmarking/datasets/QM9-full.csv")
TMQMG_PATH = Path("/home/gridsan/jtoney/ElemNet/benchmarking/datasets/tmQMg-full.csv")
BOSTMC_PATH = Path("/home/gridsan/jtoney/BOSTMC/datasets/BOSTMC-low-spin.csv")


def _skip_if_missing(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"dataset not available at {p}")
    return p


@pytest.fixture(scope="session")
def qm9_path() -> Path:
    return _skip_if_missing(QM9_PATH)


@pytest.fixture(scope="session")
def tmqmg_path() -> Path:
    return _skip_if_missing(TMQMG_PATH)


@pytest.fixture(scope="session")
def bostmc_path() -> Path:
    return _skip_if_missing(BOSTMC_PATH)
