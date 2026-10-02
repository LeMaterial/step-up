"""Compare finished runs on one table, with D-MAE normalized by molecule size.

D-MAE is an absolute distance error, so it grows with the molecule: a 60-atom
complex and a 9-atom skeleton cannot be read off the same column. Every row here
therefore carries a *relative* D-MAE as well — the pooled error divided by that
test set's own mean pairwise distance, pooled the same way (over every ordered
pair including the zero diagonal, as ReBind's ``evaluate.py`` does). That scale
is computed from the reference structures in each run's ``structures_test.csv``,
so the normalizer comes from the same molecules that were scored.

The script also recomputes D-MAE from the dumped structures and checks it against
the metrics JSON. They are produced by separate passes over the same checkpoint,
so a disagreement means one of them is stale.

Usage::

    uv run python scripts/compare_runs.py
    uv run python scripts/compare_runs.py --runs outputs/bostmc_full outputs/bostmc_full_noh
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# (label, output_dir). The pairs sit next to each other so the hydrogen
# comparison reads down the table.
_DEFAULT_RUNS: tuple[tuple[str, str], ...] = (
    ("QM9, with H", "outputs/qm9_rebind"),
    ("QM9, heavy atom", "outputs/qm9_rebind_noh"),
    ("tmQMg, with H", "outputs/tmqmg_complete_full"),
    ("tmQMg, heavy atom", "outputs/tmqmg_complete_full_noh"),
    ("BOSTMC, with H", "outputs/bostmc_full"),
    ("BOSTMC, heavy atom", "outputs/bostmc_full_noh"),
)


def _coords(xyz_block: str) -> np.ndarray:
    lines = xyz_block.splitlines()
    count = int(lines[0])
    return np.array(
        [[float(v) for v in line.split()[1:4]] for line in lines[2 : 2 + count]],
        dtype=float,
    )


def _pooled_scale_and_dmae(frame: pd.DataFrame) -> tuple[float, float, float]:
    """Mean pairwise distance, pooled D-MAE, and mean atom count over a dump.

    Both pooled quantities use the same denominator as the published protocol:
    the total number of ordered atom pairs in the split, diagonal included.
    """
    total_distance, total_abs_error, total_pairs, total_atoms = 0.0, 0.0, 0, 0
    for true_block, pred_block in zip(frame["xyz_true"], frame["xyz_pred"], strict=True):
        target, pred = _coords(true_block), _coords(pred_block)
        d_true = np.linalg.norm(target[:, None, :] - target[None, :, :], axis=-1)
        d_pred = np.linalg.norm(pred[:, None, :] - pred[None, :, :], axis=-1)
        total_distance += float(d_true.sum())
        total_abs_error += float(np.abs(d_pred - d_true).sum())
        total_pairs += d_true.size
        total_atoms += len(target)
    return (
        total_distance / max(total_pairs, 1),
        total_abs_error / max(total_pairs, 1),
        total_atoms / max(len(frame), 1),
    )


def _load(label: str, run_dir: Path) -> dict | None:
    dump = run_dir / "structures_test.csv"
    metrics_path = run_dir / "eval_test.json"
    if not dump.exists() or not metrics_path.exists():
        missing = [str(p.name) for p in (metrics_path, dump) if not p.exists()]
        print(f"skipping {label}: missing {', '.join(missing)}")
        return None

    metrics = json.loads(metrics_path.read_text())
    frame = pd.read_csv(dump)
    scale, dmae_from_dump, mean_atoms = _pooled_scale_and_dmae(frame)

    keephs_path = run_dir / "eval_test_keephs.json"
    keephs = json.loads(keephs_path.read_text()) if keephs_path.exists() else {}

    return {
        "label": label,
        "n": len(frame),
        "mean_atoms": mean_atoms,
        "d_mae": metrics["d_mae"],
        "d_rmse": metrics["d_rmse"],
        "rmsd": metrics["c_rmsd"],
        "rmsd_method": metrics.get("c_rmsd_method", "?"),
        "rmsd_keephs": keephs.get("c_rmsd", math.nan),
        "scale": scale,
        "relative": metrics["d_mae"] / scale if scale else math.nan,
        "rmsd_over_scale": metrics["c_rmsd"] / scale if scale else math.nan,
        "dmae_from_dump": dmae_from_dump,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", help="Output directories; defaults to all six")
    parser.add_argument("--out", help="Also write the table as JSON here")
    args = parser.parse_args(argv)

    if args.runs:
        pairs = [(Path(r).name, r) for r in args.runs]
    else:
        pairs = list(_DEFAULT_RUNS)

    rows = [row for label, d in pairs if (row := _load(label, Path(d))) is not None]
    if not rows:
        print("no finished runs found")
        return 1

    header = (
        f"| {'Run':<20} | {'Test mols':>9} | {'Atoms':>6} | {'D-MAE':>6} | {'D-RMSE':>6} "
        f"| {'RMSD':>6} | {'Scale':>6} | {'Rel. D-MAE':>10} | {'RMSD/scale':>10} |"
    )
    print(header)
    print("|" + "|".join("-" * (len(part)) for part in header.split("|")[1:-1]) + "|")
    for r in rows:
        print(
            f"| {r['label']:<20} | {r['n']:>9,} | {r['mean_atoms']:>6.1f} | {r['d_mae']:>6.3f} "
            f"| {r['d_rmse']:>6.3f} | {r['rmsd']:>6.3f} | {r['scale']:>6.3f} "
            f"| {r['relative'] * 100:>9.1f}% | {r['rmsd_over_scale'] * 100:>9.1f}% |"
        )

    print()
    for r in rows:
        gap = abs(r["d_mae"] - r["dmae_from_dump"])
        flag = "OK" if gap < 5e-3 else "MISMATCH"
        print(
            f"{flag:8s} {r['label']:<20} json={r['d_mae']:.4f} "
            f"recomputed_from_dump={r['dmae_from_dump']:.4f}"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
