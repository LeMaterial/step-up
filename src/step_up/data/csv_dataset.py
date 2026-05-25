"""Unified CSV → graph-dict dataset.

Backs all three current targets (QM9-full, tmQMg-full, BOSTMC) with one class.
Each row is featurized lazily into a ReBind-compatible dict that the Collator
can pad and batch.

The dataset performs an upfront *validation pass* that attempts to featurize
every row and keeps only the indices that succeed. This is the explicit answer
to the silent-corruption failure mode where RDKit could not infer bond orders
on certain QM9 rows: rather than silently substituting a wrong feature value,
those rows are filtered out and reported.
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from .featurize import featurize_mol2_xyz, featurize_xyz

# Embedded XYZ/MOL2 fields blow past the default csv.field_size_limit. Raise it
# once at import time so pandas (which uses Python's csv internally for the C
# fallback path) doesn't truncate.
csv.field_size_limit(sys.maxsize)


@dataclass(frozen=True)
class DatasetSpec:
    """Where to find a CSV and how to read it."""

    path: Path
    source: Literal["smiles", "mol2"]


def _read_csv_subset(path: Path, columns: Sequence[str], nrows: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=list(columns), nrows=nrows, low_memory=False)
    df = df.reset_index(drop=True)
    return df


class CSVMoleculeDataset(Dataset):
    """Row-streaming dataset over one of QM9-full / tmQMg-full / BOSTMC.

    The CSV is loaded into a pandas DataFrame at construction time (the
    ``subset_size`` knob keeps memory bounded for smoke runs). When
    ``validate=True`` (the default), the dataset then walks every row, calls
    the featurizer, and keeps only indices whose featurization succeeds. The
    number of dropped rows is reported once.
    """

    def __init__(
        self,
        path: str | Path,
        source: Literal["smiles", "mol2"],
        subset_size: int | None = None,
        validate: bool = True,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.source = source

        if source == "smiles":
            # QM9 path: XYZ only — RDKit's DetermineBonds recovers chemistry.
            # We still load the smiles column (unused) so the column-set check
            # asserts the source is actually QM9-shaped.
            cols = ("smiles", "xyz")
        elif source == "mol2":
            cols = ("mol2", "xyz")
        else:
            raise ValueError(f"Unknown source: {source!r}")

        self._df = _read_csv_subset(self.path, cols, nrows=subset_size)

        if validate:
            self._valid_indices = self._validate_rows()
        else:
            self._valid_indices = list(range(len(self._df)))

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        real_idx = self._valid_indices[idx]
        return self._featurize(real_idx)

    def _featurize(self, real_idx: int) -> dict[str, Any]:
        row = self._df.iloc[real_idx]
        if self.source == "smiles":
            return featurize_xyz(str(row["xyz"]))
        return featurize_mol2_xyz(str(row["mol2"]), str(row["xyz"]))

    def _validate_rows(self) -> list[int]:
        n = len(self._df)
        kept: list[int] = []
        failures: dict[str, int] = {}
        pbar = tqdm(range(n), desc=f"validating {self.path.name}", unit="mol", dynamic_ncols=True)
        for i in pbar:
            try:
                self._featurize(i)
                kept.append(i)
            except Exception as e:
                # Bucket by error class so the summary line is interpretable.
                key = type(e).__name__
                failures[key] = failures.get(key, 0) + 1
        dropped = n - len(kept)
        if dropped:
            summary = ", ".join(f"{k}={v}" for k, v in sorted(failures.items()))
            print(
                f"[{self.path.name}] validation: kept {len(kept)}/{n} rows, "
                f"dropped {dropped} ({summary})",
                flush=True,
            )
        return kept
