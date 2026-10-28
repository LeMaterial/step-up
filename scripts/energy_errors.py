"""Energy error of predicted structures: E(predicted) - E(ground truth), per molecule.

Reads a run's ``structures_test.csv`` and computes a single-point energy on both
geometries with the same method, charge and spin multiplicity, then reports the
MAE and RMSE of the difference in kcal/mol.

Methods: ``xtb`` (GFN2-xTB, needs the ``xtb`` binary on PATH) and ``uma`` (Meta's
UMA, OMol task, needs fairchem). Standard library only apart from those, so it
runs under either environment. Results are appended as they finish, and a rerun
skips molecules already done.

Usage::

    python scripts/energy_errors.py --run outputs/bostmc_full --method xtb --workers 32
    python scripts/energy_errors.py --run outputs/bostmc_full --method uma
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

csv.field_size_limit(sys.maxsize)

HARTREE_TO_KCAL = 627.509474
EV_TO_KCAL = 23.060548
_XTB_ENERGY = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
_FIELDS = ("id", "charge", "spin_multiplicity", "n_atoms", "e_true", "e_pred", "delta", "status")


def _xtb_energy(xyz: str, charge: int, multiplicity: int, timeout: int) -> float:
    """GFN2-xTB single-point energy in kcal/mol."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "mol.xyz").write_text(xyz + "\n")
        cmd = ["xtb", "mol.xyz", "--gfn", "2", "--sp", "--chrg", str(charge)]
        cmd += ["--uhf", str(multiplicity - 1)]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_STACKSIZE="2G")
        out = subprocess.run(
            cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout, check=False
        )
        match = _XTB_ENERGY.search(out.stdout)
        if out.returncode != 0 or match is None:
            raise RuntimeError("xtb did not converge")
        return float(match.group(1)) * HARTREE_TO_KCAL


def _xtb_row(task: tuple[dict, int]) -> dict:
    row, timeout = task
    charge, mult = round(float(row["charge"])), round(float(row["spin_multiplicity"]))
    result = {k: row[k] for k in ("id", "charge", "spin_multiplicity", "n_atoms")}
    try:
        e_true = _xtb_energy(row["xyz_true"], charge, mult, timeout)
    except Exception as exc:
        return {**result, "status": f"true_failed: {type(exc).__name__}"}
    try:
        e_pred = _xtb_energy(row["xyz_pred"], charge, mult, timeout)
    except Exception as exc:
        return {**result, "e_true": e_true, "status": f"pred_failed: {type(exc).__name__}"}
    return {**result, "e_true": e_true, "e_pred": e_pred, "delta": e_pred - e_true, "status": "ok"}


def _uma_predictor(model: str, device: str, snapshot: str | None):
    """UMA predict unit, by hub name or from a local snapshot directory."""
    from fairchem.core import pretrained_mlip

    if snapshot is None:
        return pretrained_mlip.get_predict_unit(model, device=device)
    from fairchem.core.units.mlip_unit import load_predict_unit
    from omegaconf import OmegaConf

    refs = Path(snapshot) / "references"
    # Formation-energy references are only used by the materials tasks, and any
    # per-element reference cancels in E(pred) - E(true) anyway.
    form_path = refs / "form_elem_refs.yaml"
    form_refs = OmegaConf.load(form_path)["refs"] if os.access(form_path, os.R_OK) else None
    return load_predict_unit(
        Path(snapshot) / "checkpoints" / f"{model}.pt",
        device=device,
        atom_refs=OmegaConf.load(refs / "iso_atom_elem_refs.yaml"),
        form_elem_refs=form_refs,
    )


def _uma_rows(rows: list[dict], model: str, device: str, snapshot: str | None):
    """Yield one result per row using UMA's OMol task. Energies in kcal/mol."""
    from io import StringIO

    from ase.io import read
    from fairchem.core import FAIRChemCalculator

    calc = FAIRChemCalculator(_uma_predictor(model, device, snapshot), task_name="omol")

    def energy(xyz: str, charge: int, mult: int) -> float:
        atoms = read(StringIO(xyz + "\n"), format="xyz")
        atoms.info.update({"charge": charge, "spin": mult})
        atoms.calc = calc
        value = float(atoms.get_potential_energy()) * EV_TO_KCAL
        if not math.isfinite(value):
            raise RuntimeError("non-finite energy")
        return value

    for row in rows:
        charge, mult = round(float(row["charge"])), round(float(row["spin_multiplicity"]))
        result = {k: row[k] for k in ("id", "charge", "spin_multiplicity", "n_atoms")}
        try:
            e_true = energy(row["xyz_true"], charge, mult)
            e_pred = energy(row["xyz_pred"], charge, mult)
            yield {
                **result,
                "e_true": e_true,
                "e_pred": e_pred,
                "delta": e_pred - e_true,
                "status": "ok",
            }
        except Exception as exc:
            yield {**result, "status": f"failed: {type(exc).__name__}"}


def summarize(run: Path, method: str) -> dict:
    rows: list[dict] = []
    for path in sorted(run.glob(f"energies_{method}*.csv")):
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))
    deltas = [float(r["delta"]) for r in rows if r["status"] == "ok"]
    summary = {
        "n_total": len(rows),
        "n_ok": len(deltas),
        "n_failed": len(rows) - len(deltas),
        "units": "kcal/mol",
    }
    if deltas:
        summary.update(
            energy_mae=sum(abs(d) for d in deltas) / len(deltas),
            energy_rmse=math.sqrt(sum(d * d for d in deltas) / len(deltas)),
            median_abs_error=statistics.median(abs(d) for d in deltas),
            mean_signed_error=sum(deltas) / len(deltas),
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run directory holding structures_test.csv")
    parser.add_argument("--method", required=True, choices=["xtb", "uma"])
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="xtb processes")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per xtb calculation")
    parser.add_argument("--uma-model", default="uma-s-1p1")
    parser.add_argument(
        "--uma-snapshot",
        help="Local UMA snapshot dir (with checkpoints/ and references/) instead of the hub",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, help="Only the first N molecules (for testing)")
    parser.add_argument("--shard", type=int, default=0, help="Which shard to compute (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Split the run across N jobs")
    parser.add_argument(
        "--summarize", action="store_true", help="Only merge existing shard files and report"
    )
    args = parser.parse_args(argv)

    run = Path(args.run)
    if args.summarize:
        return _write_summary(run, args.method)
    suffix = f".part{args.shard}of{args.num_shards}" if args.num_shards > 1 else ""
    out_path = run / f"energies_{args.method}{suffix}.csv"
    with open(run / "structures_test.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[args.shard :: args.num_shards]

    done: set[str] = set()
    if out_path.exists():
        with open(out_path, newline="") as handle:
            done = {r["id"] for r in csv.DictReader(handle)}
    todo = [r for r in rows if r["id"] not in done]
    print(
        f"{run.name} [{args.method}]: {len(todo)} to compute, {len(done)} already done", flush=True
    )

    with open(out_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS, restval="")
        if handle.tell() == 0:
            writer.writeheader()
        if args.method == "xtb":
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                results = pool.map(_xtb_row, [(r, args.timeout) for r in todo], chunksize=4)
                for i, result in enumerate(results, 1):
                    writer.writerow(result)
                    if i % 200 == 0:
                        handle.flush()
                        print(f"  {i}/{len(todo)}", flush=True)
        else:
            for i, result in enumerate(
                _uma_rows(todo, args.uma_model, args.device, args.uma_snapshot), 1
            ):
                writer.writerow(result)
                if i % 200 == 0:
                    handle.flush()
                    print(f"  {i}/{len(todo)}", flush=True)

    return _write_summary(run, args.method)


def _write_summary(run: Path, method: str) -> int:
    summary = {"run": run.name, "method": method, **summarize(run, method)}
    (run / f"energy_summary_{method}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
