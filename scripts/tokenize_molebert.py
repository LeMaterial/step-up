"""Precompute Mole-BERT tokenizer ids for a dataset.

GTMGC's published conformer models embed atoms as Mole-BERT tokenizer ids
(``embed_style="atom_tokenized_ids"``), so every molecule needs per-atom
``input_ids``. Upstream tokenizes the dataset once and saves it to disk
(``tokenize_mole.py``); we do the same, writing a small ``id,input_ids`` CSV that
``CSVMoleculeDataset`` joins on via ``token_file``.

The dataset is built from the same config the training run uses, so the rows and
the featurization match; rows that fail featurization are dropped here exactly as
they are in training.

Usage::

    uv run python scripts/tokenize_molebert.py -c configs/qm9_gtmgc.yaml \
        --out ~/step-up-data/qm9/qm9-molebert-tokens.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from step_up.data.csv_dataset import CSVMoleculeDataset
from step_up.models.gtmgc import load_molebert_tokenizer
from step_up.train import TrainConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="Training config to take data from")
    parser.add_argument("--out", required=True, type=Path, help="CSV to write")
    parser.add_argument("--checkpoint", default="RichXuOvO/MoleBERT-Tokenizer")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    cfg = TrainConfig.from_yaml(args.config)
    if cfg.id_column is None:
        raise ValueError("the config needs id_column so tokens can be keyed by molecule")
    dataset = CSVMoleculeDataset(
        path=cfg.dataset_path,
        source=cfg.dataset_source,  # type: ignore[arg-type]
        subset_size=cfg.subset_size,
        validate=cfg.validate_dataset,
        max_drop_fraction=cfg.max_drop_fraction,
        filter_column=cfg.filter_column,
        filter_value=cfg.filter_value,
        id_column=cfg.id_column,
        # token_file deliberately omitted: this script is what creates it.
    )
    keys = dataset.split_keys()
    tokenizer, collator = load_molebert_tokenizer(args.checkpoint)
    tokenizer = tokenizer.eval().to(args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "input_ids"])
        for start in tqdm(range(0, len(dataset), args.batch_size), desc="tokenizing", unit="batch"):
            chunk = list(range(start, min(start + args.batch_size, len(dataset))))
            graphs = [dataset[i] for i in chunk]
            batch = collator(graphs)
            with torch.no_grad():
                out = tokenizer(**{k: v.to(args.device) for k, v in batch.items()})
            token_ids = out["quantized_indices"].cpu().tolist()
            offset = 0
            for index, graph in zip(chunk, graphs, strict=True):
                n = graph["num_nodes"]
                writer.writerow(
                    [keys[index], " ".join(str(t) for t in token_ids[offset : offset + n])]
                )
                offset += n
                written += 1
            if offset != len(token_ids):
                raise RuntimeError(
                    f"tokenizer returned {len(token_ids)} ids for {offset} atoms in this batch"
                )
    print(f"wrote {written} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
