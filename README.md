# step-up

A benchmark for 2D to 3D conformer generation models.

## Getting Started

This project uses [`uv`](https://docs.astral.sh/uv/) for Python, dependency, and environment management.

The first model (ReBind) is vendored as a git submodule under `external/ReBIND`, so clone with `--recursive`:

```bash
git clone --recursive https://github.com/LeMaterial/step-up.git
cd step-up
uv sync --dev
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

Run the test suite (CPU-only, ~30s):

```bash
uv run pytest
```

Run formatting and lint checks:

```bash
uv run ruff format --check
uv run ruff check
```

## CPU Smoke Runs

Each config under `configs/*_smoke.yaml` trains a tiny model on a 100-molecule
subset for 3 epochs on CPU. The goal is to verify the pipeline end-to-end (data
loader --> model forward --> loss --> optimizer), not to produce meaningful
metrics. Loss should decrease monotonically across epochs.

```bash
uv run step-up train -c configs/qm9_smoke.yaml      # ~25s
uv run step-up train -c configs/tmqmg_smoke.yaml    # ~5 min (larger complexes)
uv run step-up train -c configs/bostmc_smoke.yaml   # ~6 min (larger complexes)
```

## Production Training

Three full-scale configs are staged. They target GPU (`device: cuda`) and follow
ReBind's published QM9 setup for the architecture and main optimizer settings
(8 encoder + 8 decoder layers, d_model=512, AdamW, lr=9e-5, 10% warmup, gradient
clipping at 1.0, batch 100, 20 epochs by default). The training loop differs from
ReBind's script in three ways: the learning rate decays on a cosine schedule after
warmup (ReBind: linear), AdamW keeps PyTorch's default beta2=0.999 (ReBind: 0.99),
and training runs in fp32 (ReBind: fp16 mixed precision).

| Config | Dataset | Rows | Notes |
|---|---|---|---|
| `configs/qm9.yaml` | QM9-full.csv | ~134K | Sanity baseline on organic systems |
| `configs/tmqmg.yaml` | tmQMg-full.csv | ~58K | Singlets, full d-block + La |
| `configs/bostmc.yaml` | BOSTMC-low-spin.csv | ~93K of ~121K | Singlets only (`spinmult == 1`), full d-block |

The BOSTMC config trains on singlets only: the model isn't conditioned on charge
or spin, so mixing spin states would make comparisons ambiguous. Doublets need
that conditioning first.

Train/val/test membership is a hash of each molecule's ID (`id_column`, e.g.
`refcode`, or the CSV row number if unset) and `split_seed`. A molecule
therefore stays in the same split regardless of `subset_size`, filtering, or
which rows fail featurization, so different runs and models are scored on the
same test molecules. Split sizes match `split_ratios` approximately.

To dry-run a config (validates the YAML and dataset path without training):

```bash
uv run step-up train -c configs/qm9.yaml --dry-run
```

```bash
sbatch scripts/train.sh configs/qm9.yaml
sbatch scripts/train.sh configs/tmqmg.yaml
sbatch scripts/train.sh configs/bostmc.yaml
```

The Slurm script lives at [scripts/train.sh](scripts/train.sh). It (1)
initializes the submodule, (2) runs `uv sync --dev`, and (3) launches
`uv run step-up train -c <config>`. Each run writes its config, TensorBoard
logs, per-epoch history, best checkpoint (by val D-MAE), and the test-set
metrics of that checkpoint (`test_metrics.json`) to the `output_dir` specified
in the config — default is `outputs/<dataset>_full/`.

### Adjusting training duration

To train longer, edit the config's `epochs` field. ReBind's paper used 20
epochs; more epochs only help if val D-MAE is still trending down at the end.
Inspect TensorBoard during the run:

```bash
uv run tensorboard --logdir outputs/qm9_full/tb
```

### Resuming after compute interruptions

The current loop saves `best.pt` (model weights only) on the best val D-MAE
epoch but doesn't snapshot the optimizer / scheduler state. Resumption is a
TODO for the next iteration. For long runs, prefer over-provisioning the time
budget in the Slurm header.

## Repository Layout

```
src/step_up/
|-- data/
|   |-- csv_dataset.py    # CSV --> graph-dict dataset
|   |-- featurize.py      # XYZ path (RDKit DetermineBonds for QM9)
|   |-- mol2.py           # direct MOL2 parser (no RDKit, used for organometallics)
|   |-- splits.py         # hash-based, stable train/val/test splits
|-- models/
|   |-- rebind.py         # thin wrapper over external/ReBIND + 3 runtime patches
|-- eval/
|   |-- metrics.py        # D-MAE, D-RMSE, coord-RMSD, per-element D-MAE
|-- train.py              # config-driven training loop
|-- cli.py                # `uv run step-up train -c <yaml>`

configs/           # per-dataset YAML configs (smoke + full)
external/ReBIND/   # git submodule, vendored upstream ReBind
scripts/train.sh   # Slurm job script (sbatch scripts/train.sh <config>)
tests/             # imports, dataset loading, MOL2 parsing, forward pass, metrics
```

## Data Path Notes

- **QM9 (smiles + xyz)**: built from the XYZ block via
  `Chem.MolFromXYZBlock` + `rdDetermineBonds.DetermineBonds`. The SMILES
  column is intentionally unused because its atom order doesn't match the
  XYZ block in QM9-full.csv.
- **Organometallics (mol2 + xyz)**: built directly from the MOL2 block via
  the in-house parser in `step_up/data/mol2.py`. No RDKit* is involved;
  connectivity and bond types come straight from the MOL2 file, SYBYL atom
  types provide hybridization and aromaticity (e.g. `C.ar`, `N.am`), and rings
  are computed from the bond graph via Tarjan bridge-finding.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, pre-commit hooks,
and contribution checks.
