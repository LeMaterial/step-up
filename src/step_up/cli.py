"""Command-line entry point: ``uv run step-up <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .train import TrainConfig, train


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="step-up")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model from a YAML config")
    p_train.add_argument("-c", "--config", required=True, help="Path to YAML config")
    p_train.add_argument(
        "--dry-run", action="store_true", help="Validate config and dataset path without training"
    )
    p_train.set_defaults(func=_cmd_train)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
