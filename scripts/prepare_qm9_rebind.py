"""Build a QM9 CSV carrying ReBind's published train/val/test split.

Source data is the HuggingFace dataset ``RichXuOvO/HFQm9``, the QM9 copy that
GTMGC and ReBind trained on:

    gdb9.sdf            133,885 molecules, one SDF record each
    train_indices.csv   110,000 0-based positions into gdb9.sdf
    valid_indices.csv    10,000
    test_indices.csv     10,831

The three index files are disjoint and together cover 130,831 molecules; the
3,054 QM9 entries that failed the original consistency check are left out.

Download them with::

    mkdir -p <raw-dir> && cd <raw-dir>
    for f in gdb9.sdf train_indices.csv valid_indices.csv test_indices.csv; do
      curl -sLO "https://huggingface.co/datasets/RichXuOvO/HFQm9/resolve/main/$f"
    done

Molblocks are copied out of the SDF verbatim (no RDKit round trip), so the
training pipeline featurizes exactly the published structures and bond orders.

Usage::

    uv run python scripts/prepare_qm9_rebind.py --raw-dir <raw-dir> --out <csv>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SPLIT_FILES = {"train": "train_indices.csv", "val": "valid_indices.csv", "test": "test_indices.csv"}


def read_molblocks(sdf_path: Path) -> list[str]:
    """Split an SDF into per-molecule blocks, keeping each record verbatim."""
    records = sdf_path.read_text().split("$$$$\n")
    if records and not records[-1].strip():
        records.pop()
    return records


def read_indices(raw_dir: Path) -> dict[str, list[int]]:
    indices: dict[str, list[int]] = {}
    for split, name in SPLIT_FILES.items():
        with open(raw_dir / name) as f:
            rows = list(csv.DictReader(f))
        indices[split] = [int(row["index"]) for row in rows]
    return indices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path, help="Directory holding gdb9.sdf")
    parser.add_argument("--out", required=True, type=Path, help="CSV to write")
    args = parser.parse_args(argv)

    csv.field_size_limit(sys.maxsize)
    blocks = read_molblocks(args.raw_dir / "gdb9.sdf")
    print(f"gdb9.sdf: {len(blocks)} records")
    indices = read_indices(args.raw_dir)

    assigned: dict[int, str] = {}
    for split, idx_list in indices.items():
        for i in idx_list:
            if i in assigned:
                raise ValueError(f"index {i} appears in both {assigned[i]} and {split}")
            if not 0 <= i < len(blocks):
                raise ValueError(f"index {i} is out of range for {len(blocks)} records")
            assigned[i] = split
    print("split sizes: " + ", ".join(f"{s}={len(v)}" for s, v in indices.items()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_id", "split", "sdf"])
        for i in sorted(assigned):
            block = blocks[i]
            # The first line of an SDF record is the molecule title, e.g. "gdb_73134".
            mol_id = block.lstrip("\n").split("\n", 1)[0].strip()
            if not mol_id:
                raise ValueError(f"record {i} has no title line to use as mol_id")
            writer.writerow([mol_id, assigned[i], block])
    print(f"wrote {len(assigned)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
