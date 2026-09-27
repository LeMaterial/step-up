"""Check whether a conditioned model actually uses its charge/spin input.

Scores a split three ways:

1. as trained,
2. with the molecule-level features zeroed at inference — if the metrics barely
   move, the model learned to ignore the conditioning, and
3. optionally broken down by a column of the source CSV (e.g. ``spinmult``), to
   see whether one subgroup is carrying the error.

Usage::

    uv run python scripts/analyze_conditioning.py -c configs/bostmc.yaml \
        --checkpoint outputs/bostmc_full/best.pt --breakdown-column spinmult
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import Subset

from step_up.eval.conformer_eval import evaluate_split
from step_up.models import N_GLOBAL_FEATURES, build_model, build_model_collator
from step_up.train import TrainConfig, build_splits


class ZeroConditioning:
    """Collator wrapper that blanks the molecule-level features."""

    def __init__(self, base):
        self.base = base

    def __call__(self, mols):
        batch = self.base(mols)
        batch["global_features"] = torch.zeros_like(batch["global_features"])
        return batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", help="Defaults to <output_dir>/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--breakdown-column", help="CSV column to group the split by")
    parser.add_argument("--out", help="Defaults to <output_dir>/conditioning_analysis.json")
    args = parser.parse_args(argv)

    cfg = TrainConfig.from_yaml(args.config)
    conditioned = cfg.charge_column is not None or cfg.spin_column is not None
    if not conditioned:
        raise ValueError(f"{args.config} trains without conditioning; nothing to ablate")
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    cfg = replace(cfg, device=device)

    dataset, train_set, val_set, test_set = build_splits(cfg)
    subset = {"train": train_set, "val": val_set, "test": test_set}[args.split]
    model = build_model(
        cfg.model,
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        d_ffn=cfg.d_ffn,
        n_head=cfg.n_head,
        dropout=cfg.dropout,
        n_global_features=N_GLOBAL_FEATURES,
    )
    checkpoint = args.checkpoint or str(Path(cfg.output_dir) / "best.pt")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device)
    collator = build_model_collator(cfg.model, conditioned=True)

    results: dict[str, dict] = {}

    def score(tag: str, rows: Subset, coll) -> None:
        metrics = evaluate_split(
            model, dataset, rows, coll, device=device, batch_size=cfg.eval_batch_size
        )
        results[tag] = metrics
        print(
            f"{tag:30s} n={int(metrics['n_molecules']):6d} D-MAE={metrics['d_mae']:.4f} "
            f"D-RMSE={metrics['d_rmse']:.4f} RMSD={metrics['c_rmsd']:.3f}",
            flush=True,
        )

    score(f"{args.split} (as trained)", subset, collator)
    score(f"{args.split} (conditioning zeroed)", subset, ZeroConditioning(collator))

    if args.breakdown_column:
        rows = dataset._df.iloc[[dataset._valid_indices[i] for i in subset.indices]]
        values = rows[args.breakdown_column].tolist()
        for value in sorted(set(values)):
            group = Subset(
                dataset, [i for i, v in zip(subset.indices, values, strict=True) if v == value]
            )
            # Both ways per group: conditioning can be useless on average yet matter
            # for the subgroups it describes, e.g. charged or open-shell species.
            score(f"{args.breakdown_column}={value}", group, collator)
            score(f"{args.breakdown_column}={value} (zeroed)", group, ZeroConditioning(collator))

    out_path = Path(args.out or Path(cfg.output_dir) / "conditioning_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
