"""Write the true and predicted geometry of every test-set molecule to a CSV.

One row per molecule, so the file can be read back with pandas and the two
structures overlaid directly:

    id, charge, spin_multiplicity, n_atoms, elements, xyz_true, xyz_pred,
    d_mae, d_rmse, rmsd

``xyz_true`` and ``xyz_pred`` are standard XYZ blocks (count line, comment line,
then ``element x y z``), the same format the source CSVs use, with atoms in the
same order in both.

Two things worth knowing about ``xyz_pred``:

- It is **Kabsch-aligned onto the true structure** per molecule. The models
  align inside their own prediction head over the *padded* batch tensor, so the
  raw output sits in a frame corrupted by padding zeros; aligning here is what
  makes the two blocks comparable. Only a rotation and translation are applied,
  so bond lengths and angles are the model's own.
- Hydrogens are absent for a model trained with ``remove_hs``, because that
  model never predicted any.

``charge`` and ``spin_multiplicity`` come from the source CSV. They are empty
when the dataset has no such column, which is not a claim of neutral or
closed-shell — it means the data does not say.

Usage::

    uv run python scripts/dump_structures.py -c configs/bostmc.yaml
    uv run python scripts/dump_structures.py -c configs/qm9_rebind.yaml \
        --checkpoint outputs/qm9_rebind/best.pt --split test
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import torch
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm

from step_up.eval.conformer_eval import kabsch_rmsd
from step_up.models import N_GLOBAL_FEATURES, build_model, build_model_collator
from step_up.train import TrainConfig, build_splits

_PERIODIC_TABLE = Chem.GetPeriodicTable()

_COLUMNS = (
    "id",
    "charge",
    "spin_multiplicity",
    "n_atoms",
    "elements",
    "xyz_true",
    "xyz_pred",
    "d_mae",
    "d_rmse",
    "rmsd",
)


def _xyz_block(symbols: list[str], coords: torch.Tensor, comment: str) -> str:
    lines = [str(len(symbols)), comment]
    for symbol, (x, y, z) in zip(symbols, coords.tolist(), strict=True):
        lines.append(f"{symbol} {x:.6f} {y:.6f} {z:.6f}")
    return "\n".join(lines)


def _aligned(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``pred`` rigidly moved onto ``target`` — rotation and translation only."""
    p = pred - pred.mean(dim=0, keepdim=True)
    q = target - target.mean(dim=0, keepdim=True)
    u, _, vt = torch.linalg.svd(p.T @ q)
    sign = torch.sign(torch.det(vt.T @ u.T))
    correction = torch.diag(torch.tensor([1.0, 1.0, float(sign)], dtype=p.dtype))
    rotation = vt.T @ correction @ u.T
    return (rotation @ p.T).T + target.mean(dim=0, keepdim=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", help="Defaults to <output_dir>/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", help="Defaults to <output_dir>/structures_<split>.csv")
    args = parser.parse_args(argv)

    cfg = TrainConfig.from_yaml(args.config)
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    cfg = replace(cfg, device=device)
    conditioned = cfg.charge_column is not None or cfg.spin_column is not None

    dataset, train_set, val_set, test_set = build_splits(cfg)
    subset = {"train": train_set, "val": val_set, "test": test_set}[args.split]

    model = build_model(
        cfg.model,
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        d_ffn=cfg.d_ffn,
        n_head=cfg.n_head,
        dropout=cfg.dropout,
        n_global_features=N_GLOBAL_FEATURES if conditioned else 0,
    )
    checkpoint = args.checkpoint or str(Path(cfg.output_dir) / "best.pt")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device).eval()

    collator = build_model_collator(cfg.model, conditioned=conditioned)
    loader = DataLoader(subset, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collator)

    out_path = Path(args.out or Path(cfg.output_dir) / f"structures_{args.split}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    position = 0
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_COLUMNS)
        for batch in tqdm(loader, desc=f"dumping {args.split}", dynamic_ncols=True):
            device_batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            with torch.no_grad():
                out = model(**device_batch)
            node_mask = device_batch["node_mask"].bool()
            for row in range(node_mask.shape[0]):
                keep = node_mask[row]
                pred = out.conformer_hat[row][keep].double().cpu()
                target = out.conformer[row][keep].double().cpu()
                # node_type is Z here: the collator shifts by 1 so 0 means padding.
                symbols = [
                    _PERIODIC_TABLE.GetElementSymbol(int(z))
                    for z in device_batch["node_type"][row][keep].cpu()
                ]
                delta = (torch.cdist(pred, pred) - torch.cdist(target, target)).abs()
                meta = dataset.metadata(subset.indices[position + row])
                writer.writerow(
                    [
                        meta["id"],
                        "" if meta["charge"] is None else meta["charge"],
                        "" if meta["spin_multiplicity"] is None else meta["spin_multiplicity"],
                        len(symbols),
                        " ".join(symbols),
                        _xyz_block(symbols, target, f"{meta['id']} reference"),
                        _xyz_block(symbols, _aligned(pred, target), f"{meta['id']} predicted"),
                        f"{float(delta.mean()):.6f}",
                        f"{float((delta**2).mean().sqrt()):.6f}",
                        f"{kabsch_rmsd(pred, target):.6f}",
                    ]
                )
            position += node_mask.shape[0]

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
