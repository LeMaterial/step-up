"""Command-line entry point: ``uv run step-up <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .eval.conformer_eval import evaluate_split, write_metrics
from .models import N_GLOBAL_FEATURES, build_model, build_model_collator
from .train import TrainConfig, build_splits, train


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = TrainConfig.from_yaml(args.config)
    if args.dry_run:
        print(f"[dry-run] Loaded config from {args.config}")
        print(f"  dataset: {cfg.dataset_path} ({cfg.dataset_source})")
        if not Path(cfg.dataset_path).exists():
            print(f"  ERROR: dataset path does not exist: {cfg.dataset_path}")
            return 1
        print(f"  subset_size={cfg.subset_size} epochs={cfg.epochs} batch={cfg.batch_size}")
        print(f"  model: layers={cfg.n_layers} d_model={cfg.d_model} d_ffn={cfg.d_ffn}")
        print(f"  device={cfg.device} output_dir={cfg.output_dir}")
        return 0
    result = train(cfg)
    print(f"\nBest val D-MAE: {result['best_val_dmae']:.4f}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Score a checkpoint with ReBind's published metric definitions."""
    cfg = TrainConfig.from_yaml(args.config)
    checkpoint = str(args.checkpoint or Path(cfg.output_dir) / "best.pt")
    # A `.pt` path is one of our state dicts; anything else is a released
    # checkpoint (HuggingFace repo id or a directory), e.g. RichXuOvO/GTMGC-Qm9.
    is_state_dict = Path(checkpoint).suffix in {".pt", ".pth"}
    if is_state_dict and not Path(checkpoint).exists():
        print(f"ERROR: checkpoint not found: {checkpoint}")
        return 1

    dataset, train_set, val_set, test_set = build_splits(cfg)
    subset = {"train": train_set, "val": val_set, "test": test_set}[args.split]
    if len(subset) == 0:
        print(f"ERROR: the {args.split} split is empty")
        return 1

    conditioned = cfg.charge_column is not None or cfg.spin_column is not None
    if is_state_dict:
        model = build_model(
            cfg.model,
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            d_ffn=cfg.d_ffn,
            n_head=cfg.n_head,
            dropout=cfg.dropout,
            n_global_features=N_GLOBAL_FEATURES if conditioned else 0,
        ).to(cfg.device)
        model.load_state_dict(torch.load(checkpoint, map_location=cfg.device))
    elif cfg.model == "gtmgc":
        from .models.gtmgc import load_pretrained_gtmgc

        print(f"[evaluate] loading released checkpoint {checkpoint}", flush=True)
        model = load_pretrained_gtmgc(checkpoint).to(cfg.device)
    else:
        print(f"ERROR: {cfg.model} cannot load a released checkpoint: {checkpoint}")
        return 1
    metrics = evaluate_split(
        model,
        dataset,
        subset,
        build_model_collator(cfg.model, conditioned=conditioned),
        device=cfg.device,
        batch_size=cfg.eval_batch_size,
        remove_hs=not args.keep_hs,
    )
    out_path = Path(args.out) if args.out else Path(cfg.output_dir) / f"eval_{args.split}.json"
    write_metrics(metrics, out_path)
    hydrogens = "without hydrogens" if metrics["remove_hs"] else "with hydrogens"
    print(
        f"[{args.split}] D-MAE={metrics['d_mae']:.4f} D-RMSE={metrics['d_rmse']:.4f} "
        f"C-RMSD={metrics['c_rmsd']:.4f} ({hydrogens}, {metrics['c_rmsd_method']}) "
        f"over {int(metrics['n_molecules'])} molecules"
    )
    print(f"wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="step-up")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model from a YAML config")
    p_train.add_argument("-c", "--config", required=True, help="Path to YAML config")
    p_train.add_argument(
        "--dry-run", action="store_true", help="Validate config and dataset path without training"
    )
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("evaluate", help="Score a checkpoint (ReBind's metric protocol)")
    p_eval.add_argument("-c", "--config", required=True, help="Path to YAML config")
    p_eval.add_argument(
        "--checkpoint",
        help="Our .pt state dict, or a released checkpoint such as RichXuOvO/GTMGC-Qm9. "
        "Defaults to <output_dir>/best.pt",
    )
    p_eval.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_eval.add_argument("--out", help="Defaults to <output_dir>/eval_<split>.json")
    p_eval.add_argument(
        "--keep-hs", action="store_true", help="Keep hydrogens in C-RMSD (ReBind removes them)"
    )
    p_eval.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
