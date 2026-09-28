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
from rdkit import Chem
from torch.utils.data import Dataset
from tqdm import tqdm

from .featurize import (
    featurize_mol2_xyz,
    featurize_molblock,
    featurize_xyz,
    mol_from_xyz_block,
    remove_hydrogens,
)

# Embedded XYZ/MOL2 fields blow past the default csv.field_size_limit. Raise it
# once at import time so pandas (which uses Python's csv internally for the C
# fallback path) doesn't truncate.
csv.field_size_limit(sys.maxsize)


@dataclass(frozen=True)
class DatasetSpec:
    """Where to find a CSV and how to read it."""

    path: Path
    source: Literal["smiles", "mol2"]


def _read_token_file(path: Path) -> dict[str, list[int]]:
    """Read ``id,input_ids`` rows into a mapping of id to per-atom token ids."""
    frame = pd.read_csv(path)
    missing = {"id", "input_ids"} - set(frame.columns)
    if missing:
        raise ValueError(f"token file {path} is missing columns: {sorted(missing)}")
    return {
        str(key): [int(v) for v in str(ids).split()]
        for key, ids in zip(frame["id"], frame["input_ids"], strict=True)
    }


def _read_csv_subset(path: Path, columns: Sequence[str], nrows: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=list(columns), nrows=nrows, low_memory=False)
    df = df.reset_index(drop=True)
    return df


class CSVMoleculeDataset(Dataset):
    """Dataset over one of QM9-full / tmQMg-full / BOSTMC.

    The needed CSV columns are loaded into memory with pandas; rows are
    featurized when accessed.

    Parameters
    ----------
    path
        Path to the CSV.
    source
        One of ``"smiles"`` (QM9-style: featurize from the ``xyz`` column via
        RDKit ``DetermineBonds``), ``"mol2"`` (organometallic: featurize from
        the ``mol2`` column directly via the in-house parser), or ``"sdf"``
        (featurize an SDF record held in the ``sdf`` column, keeping its own
        bonds and coordinates).
    subset_size
        Optional cap on the number of CSV rows to load (the first rows of the
        file). Useful for smoke runs.
    validate
        If ``True`` (default), every row is featurized once at construction
        time and rows whose featurization fails are dropped. Set to ``False``
        only for datasets known to be clean: a bad row then raises during
        training instead. The upfront pass takes about a minute on QM9-full.
    max_drop_fraction
        If validation drops more than this fraction of rows, raise. Defaults
        to ``0.5`` — a sane upper bound that flags catastrophic dataset issues
        (e.g. wrong column names, encoding problems) while tolerating the
        ~6% drop rate seen on QM9-full.csv.
    filter_column, filter_value
        If both set, restrict the dataset to rows where
        ``df[filter_column] == filter_value`` before validation. Used to e.g.
        pre-filter BOSTMC to ``spinmult == 1`` so the model trains on a single
        spin manifold.
    cache
        If ``True``, featurized graph dicts are kept in memory and reused by
        ``__getitem__`` instead of re-featurizing rows every epoch. The cache is
        filled during validation, or on first access when validation is off.
        Off by default: worth it for small / smoke datasets, but with
        ``num_workers > 0`` each DataLoader worker holds its own copy.
    id_column
        Optional column holding a stable per-molecule ID (e.g. ``refcode``),
        used as the split key by :meth:`split_keys`. Rows sharing an ID always
        land in the same split. Without it, a row's key is its row number in
        the CSV.
    token_file
        Optional CSV of ``id,input_ids`` (space-separated per-atom token ids,
        see ``scripts/tokenize_molebert.py``), joined on ``id_column``. GTMGC's
        published conformer models embed Mole-BERT tokenizer ids rather than
        atom types, so their batches need this.
    split_column
        Optional column holding a published split name per row (``train`` /
        ``val`` / ``test``), read by :meth:`split_labels`. Use it to reproduce
        someone else's split instead of hashing one.
    charge_column, spin_column
        Optional columns holding the molecule's formal charge (electrons) and
        spin multiplicity. When given, each graph dict carries ``charge`` and
        ``spin_multiplicity`` for the model's global conditioning. A missing
        column means neutral / closed-shell.
    remove_hs
        Drop hydrogens from every graph, so the model neither sees nor predicts
        them. This is a different model, not a different metric: it changes what
        the network is trained on. Each heavy atom keeps its hydrogen count in
        the ``numH`` feature either way.
    """

    def __init__(
        self,
        path: str | Path,
        source: Literal["smiles", "mol2", "sdf"],
        subset_size: int | None = None,
        validate: bool = True,
        max_drop_fraction: float = 0.5,
        filter_column: str | None = None,
        filter_value: Any = None,
        cache: bool = False,
        id_column: str | None = None,
        split_column: str | None = None,
        charge_column: str | None = None,
        spin_column: str | None = None,
        token_file: str | Path | None = None,
        remove_hs: bool = False,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.source = source
        self.remove_hs = remove_hs
        self.id_column = id_column
        self.split_column = split_column
        self.charge_column = charge_column
        self.spin_column = spin_column
        self.token_file = Path(token_file) if token_file is not None else None
        self._tokens: dict[str, list[int]] | None = None
        if self.token_file is not None:
            if id_column is None:
                raise ValueError("token_file needs id_column to join the tokens on")
            self._tokens = _read_token_file(self.token_file)
        self._cache_enabled = cache
        self._cache: dict[int, dict[str, Any]] = {}

        if source == "smiles":
            cols = ["smiles", "xyz"]
        elif source == "mol2":
            cols = ["mol2", "xyz"]
        elif source == "sdf":
            cols = ["sdf"]
        else:
            raise ValueError(f"Unknown source: {source!r}")
        for extra in (filter_column, id_column, split_column, charge_column, spin_column):
            if extra is not None and extra not in cols:
                cols.append(extra)

        # The DataFrame index is each row's position in the CSV. Filtering keeps it
        # (no reset_index), so split keys don't depend on subsetting or filtering.
        self._df = _read_csv_subset(self.path, cols, nrows=subset_size)

        if filter_column is not None:
            before = len(self._df)
            self._df = self._df[self._df[filter_column] == filter_value]
            print(
                f"[{self.path.name}] filter {filter_column}=={filter_value}: "
                f"kept {len(self._df)}/{before} rows",
                flush=True,
            )

        if len(self._df) == 0:
            raise ValueError(
                f"Dataset {self.path} is empty after loading "
                f"(subset_size={subset_size}, filter={filter_column}=={filter_value})"
            )

        if validate:
            self._valid_indices = self._validate_rows(max_drop_fraction)
        else:
            self._valid_indices = list(range(len(self._df)))

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        real_idx = self._valid_indices[idx]
        graph = self._cache.get(real_idx)
        if graph is None:
            graph = self._featurize(real_idx)
            if self._cache_enabled:
                self._cache[real_idx] = graph
        return graph

    def split_keys(self) -> list[str]:
        """Stable per-row keys for :func:`step_up.data.splits.stable_split`.

        Aligned with dataset indices. Each key is the row's ``id_column`` value
        if one was given, otherwise its row number in the CSV. Neither depends
        on ``subset_size``, filtering, or which rows fail featurization.
        """
        rows = self._df.iloc[self._valid_indices]
        if self.id_column is None:
            return [str(i) for i in rows.index]
        ids = rows[self.id_column]
        if ids.isna().any():
            raise ValueError(f"id_column {self.id_column!r} has missing values in {self.path}")
        return ids.astype(str).tolist()

    def rdkit_mol(self, idx: int):
        """RDKit molecule for item ``idx``, atoms in the same order as its graph dict.

        Available for the ``sdf`` and ``smiles`` sources. The MOL2 path never
        builds an RDKit molecule, so it raises instead.
        """
        row = self._df.iloc[self._valid_indices[idx]]
        if self.source == "sdf":
            mol = Chem.MolFromMolBlock(str(row["sdf"]), removeHs=False)
        elif self.source == "smiles":
            mol = mol_from_xyz_block(str(row["xyz"]))
        else:
            raise NotImplementedError(f"source {self.source!r} does not go through RDKit")
        # Must have the same atoms in the same order as the graph dict, or the
        # evaluator would write predicted coordinates onto the wrong atoms.
        return remove_hydrogens(mol, self.remove_hs) if mol is not None else None

    def split_labels(self) -> list[str]:
        """Published split name per row, aligned with dataset indices."""
        if self.split_column is None:
            raise ValueError("split_labels() needs split_column to be set")
        labels = self._df.iloc[self._valid_indices][self.split_column]
        if labels.isna().any():
            raise ValueError(
                f"split_column {self.split_column!r} has missing values in {self.path}"
            )
        return labels.astype(str).tolist()

    def _featurize(self, real_idx: int) -> dict[str, Any]:
        row = self._df.iloc[real_idx]
        if self.source == "smiles":
            graph = featurize_xyz(str(row["xyz"]), remove_hs=self.remove_hs)
        elif self.source == "sdf":
            graph = featurize_molblock(str(row["sdf"]), remove_hs=self.remove_hs)
        else:
            graph = featurize_mol2_xyz(str(row["mol2"]), str(row["xyz"]), remove_hs=self.remove_hs)
        # Molecule-level state for conditioning; the collator turns these into
        # model inputs. Absent columns mean a neutral closed-shell molecule.
        if self.charge_column is not None:
            graph["charge"] = float(row[self.charge_column])
        if self.spin_column is not None:
            graph["spin_multiplicity"] = float(row[self.spin_column])
        if self._tokens is not None:
            key = str(row[self.id_column])
            tokens = self._tokens.get(key)
            if tokens is None:
                raise KeyError(f"no Mole-BERT tokens for {key!r} in {self.token_file}")
            if len(tokens) != graph["num_nodes"]:
                raise ValueError(
                    f"{key!r} has {len(tokens)} tokens for {graph['num_nodes']} atoms; "
                    "the token file was built from a different featurization"
                )
            graph["input_ids"] = tokens
        return graph

    def _validate_rows(self, max_drop_fraction: float) -> list[int]:
        n = len(self._df)
        kept: list[int] = []
        failures: dict[str, int] = {}
        pbar = tqdm(range(n), desc=f"validating {self.path.name}", unit="mol", dynamic_ncols=True)
        for i in pbar:
            try:
                graph = self._featurize(i)
                kept.append(i)
                if self._cache_enabled:
                    self._cache[i] = graph
            except (FileNotFoundError, ImportError):
                # Environment problems (e.g. the ReBind submodule isn't checked out)
                # are not bad rows. Re-raise so the actionable message surfaces instead
                # of being counted as a per-row failure.
                raise
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
        # Fail fast on catastrophic drop rates — an empty dataset reaching the
        # training loop produces meaningless metrics and silently corrupt
        # checkpoints.
        if len(kept) == 0:
            raise RuntimeError(
                f"Validation dropped all {n} rows of {self.path}. "
                "Check the column names, encoding, and source format."
            )
        if dropped / n > max_drop_fraction:
            raise RuntimeError(
                f"Validation dropped {dropped}/{n} rows of {self.path} "
                f"({dropped / n:.1%}), exceeding max_drop_fraction="
                f"{max_drop_fraction:.1%}. Investigate before training."
            )
        return kept
