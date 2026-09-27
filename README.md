# step-up

A benchmark for 2D to 3D conformer generation models.

## Getting Started

This project uses [`uv`](https://docs.astral.sh/uv/) for Python, dependency, and environment management.

The benchmark models are vendored as git submodules (`external/ReBIND`,
`external/GTMGC`), so clone with `--recursive`:

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

Each benchmark model is a thin wrapper over a vendored upstream implementation,
selected by `model:` in the config (`rebind` or `gtmgc`, see
`step_up.models.MODEL_NAMES`). Both descend from the same code base, share the
dataset and metric code, and need the same two fixes: the Laplacian positional
encoding is added out of place so fp32 training works, and charge/spin
conditioning is bolted on for the organometallic sets.

ReBind additionally hardcodes Lennard-Jones parameters only up to Kr, and
KeyErrors on anything heavier. [lj_params.py](src/step_up/models/lj_params.py)
carries the full UFF table through Lr, identical to theirs over Z=1..36, and the
wrapper patches it in. This is not cosmetic: 65% of tmQMg and 46% of BOSTMC
low-spin structures contain at least one element past Kr, usually the metal
centre itself.

Three full-scale configs are staged. They target GPU (`device: cuda`) and follow
ReBind's published QM9 setup from
`external/ReBIND/experiments/conformer_prediction/rebind.sh`: 8 encoder + 8
decoder layers, d_model=512, AdamW with betas (0.9, 0.99) and eps 1e-8, lr 9e-5,
linear warmup over 10% of steps followed by linear decay, gradient clipping at
1.0, batch 100 with the last partial batch dropped, and 20 epochs.

The one setting not mirrored is precision: ReBind trains with fp16 mixed
precision, this loop in fp32. fp32 is the more precise of the two, and upstream's
in-place Laplacian write (see [rebind.py](src/step_up/models/rebind.py)) only
survives autograd under autocast, so their fp16 looks like a constraint of the
code rather than a modelling choice.

| Config | Dataset | Rows | Split | Notes |
|---|---|---|---|---|
| `configs/qm9_rebind.yaml` | QM9 (gdb9.sdf) | 130,831 | published (ReBind/GTMGC) | Organic reproduction target |
| `configs/qm9_gtmgc.yaml` | QM9 (gdb9.sdf) | 130,831 | published (ReBind/GTMGC) | Same data, second model |
| `configs/tmqmg_complete.yaml` | tmQMg complete | 60,799 | published (TMCgen) | Matches the published baselines |
| `configs/tmqmg.yaml` | tmQMg, outliers removed | 58,409 | published (TMCgen) | Cleaner data, not comparable to them |
| `configs/bostmc.yaml` | BOSTMC low-spin | 121,496 | project random split | Singlets + doublets, full d-block |

Molecules carry a formal charge and spin multiplicity, which neither upstream
model takes. Both are fed in as scalars (charge in electrons, and unpaired
electrons = multiplicity - 1) through a small MLP added to every atom embedding.
Scalars rather than one-hot categories because BOSTMC's charges run from -8 to
+8 with single-structure tails, and because a scalar lets you ask a trained model
for a charge/spin state that never appeared in training. Set `charge_column` /
`spin_column` to switch it on; QM9 needs neither.

Splits come from one of three places, in order of precedence: `split_files`
(published ID lists, used for tmQMg and BOSTMC), `split_column` (a published
label per row, used for QM9), or a hash of each molecule's ID (`id_column`, or
the CSV row number) and `split_seed`. The hash keeps a molecule in the same
split regardless of `subset_size`, filtering, or which rows fail featurization,
so different runs and models are scored on the same test molecules.

To dry-run a config (validates the YAML and dataset path without training):

```bash
uv run step-up train -c configs/qm9.yaml --dry-run
```

```bash
sbatch scripts/train.sh configs/qm9_rebind.yaml
sbatch scripts/train.sh configs/tmqmg_complete.yaml
sbatch scripts/train.sh configs/bostmc.yaml
```

The Slurm script lives at [scripts/train.sh](scripts/train.sh). It (1)
initializes the submodule, (2) runs `uv sync --dev`, and (3) launches
`uv run step-up train -c <config>`. Each run writes its config, TensorBoard
logs, per-epoch history, best checkpoint (by val D-MAE), and the test-set
metrics of that checkpoint (`test_metrics.json`) to the `output_dir` specified
in the config — default is `outputs/<dataset>_full/`.

### Reproducing ReBind's published QM9 numbers

`configs/qm9_rebind.yaml` trains on the QM9 copy ReBind and GTMGC used
(HuggingFace `RichXuOvO/HFQm9`), with their published split. Fetch the raw data
and build the CSV once:

```bash
mkdir -p ~/step-up-data/qm9 && cd ~/step-up-data/qm9
for f in gdb9.sdf train_indices.csv valid_indices.csv test_indices.csv; do
  curl -sLO "https://huggingface.co/datasets/RichXuOvO/HFQm9/resolve/main/$f"
done
cd -
uv run python scripts/prepare_qm9_rebind.py \
  --raw-dir ~/step-up-data/qm9 --out ~/step-up-data/qm9/qm9-rebind.csv
```

Molblocks are copied out of `gdb9.sdf` verbatim, so bonds are the published ones
rather than re-perceived from coordinates (`dataset_source: sdf`), and each row
carries its published split label (110,000 / 10,000 / 10,831). About 1.4% of
records fail RDKit sanitization and are dropped at load time, leaving
108,478 / 9,873 / 10,661. ReBind's own evaluation skips such records too, though
their RDKit version dropped ~400 fewer.

```bash
sbatch scripts/train.sh configs/qm9_rebind.yaml
uv run step-up evaluate -c configs/qm9_rebind.yaml --split test
```

`step-up evaluate` uses ReBind's metric definitions from their `evaluate.py`:
D-MAE and D-RMSE pooled over every pairwise distance in the split (larger
molecules therefore count more), and C-RMSD from RDKit's `GetBestRMS` with
hydrogens removed. The D-MAE the training loop prints is a per-batch mean, so it
is not directly comparable.

All numbers below are the same protocol on the same 10,661-molecule test split.
Published values are as tabulated in the ReBind paper.

| Metric | ReBind paper | ReBind ours | GTMGC paper | GTMGC ours | GTMGC released weights |
|---|---|---|---|---|---|
| D-MAE | 0.254 | 0.233 | 0.281 | 0.264 | 0.265 |
| D-RMSE | 0.446 | 0.439 | 0.471 | 0.462 | 0.459 |
| C-RMSD | 0.321 | 0.265 | 0.414 | 0.372 | 0.344 |

Two things this table is good for, and one it isn't.

**Our training is sound.** Training GTMGC ourselves lands within 0.3% of their
released checkpoint on D-MAE (0.264 vs 0.265) and 0.6% on D-RMSE, having never
seen their weights.

**Model-to-model comparison holds up.** Under one protocol ReBind beats GTMGC by
12% on D-MAE, close to the 10% separating them in the paper, so the benchmark
reproduces the published ordering.

**Absolute comparisons to published tables do not.** Scoring GTMGC's *own*
released checkpoint with our code gives numbers 6% better than their published
row, and training cannot explain a gap on someone else's weights. Our whole
column therefore carries a systematic offset of roughly that size, which is most
of what makes our ReBind run look better than its paper. The likely causes are
the slightly different test set (we drop 10,661 of their ~10,697 molecules, a
newer RDKit rejecting a few more) and the published row possibly coming from a
different run than the released checkpoint. Quote our numbers against each
other, not against the papers.

### GTMGC on the same QM9 split

`configs/qm9_gtmgc.yaml` trains GTMGC (Xu et al., ICLR 2024) on exactly the data
and split above, so the two models are directly comparable. Their conformer
models embed Mole-BERT tokenizer ids rather than atom types, so precompute the
per-atom ids once:

```bash
uv run python scripts/tokenize_molebert.py -c configs/qm9_gtmgc.yaml \
  --out ~/step-up-data/qm9/qm9-molebert-tokens.csv
sbatch scripts/train.sh configs/qm9_gtmgc.yaml
```

`step-up evaluate` also scores released checkpoints, which is a useful check on
our pipeline: their weights under our protocol should reproduce their paper.

```bash
uv run step-up evaluate -c configs/qm9_gtmgc.yaml --checkpoint RichXuOvO/GTMGC-Qm9
```

Compute nodes on this cluster have no internet access, so anything pulled from
HuggingFace (the tokenizer, released checkpoints) has to be fetched from a login
node first. Either warm the cache by running the command there once, or download
the files and pass the directory:

```bash
mkdir -p ~/step-up-data/checkpoints/GTMGC-Qm9 && cd $_
for f in config.json pytorch_model.bin; do
  curl -sLO "https://huggingface.co/RichXuOvO/GTMGC-Qm9/resolve/main/$f"
done
```

### Organometallic results

All on each dataset's published/project split, scored with `step-up evaluate`.
RMSD here is Kabsch alignment on each molecule's real atoms, without symmetry
matching, so it is comparable across rows but *not* to a GetBestRMS C-RMSD.
"Relative" is D-MAE over the mean pairwise distance of that test set, which is
what makes organic and organometallic errors comparable at all: D-MAE is an
absolute distance error and grows with molecular size.

| Run | Test molecules | D-MAE | D-RMSE | RMSD | Relative D-MAE |
|---|---|---|---|---|---|
| QM9, ReBind | 10,661 | 0.233 | 0.439 | 0.858 | 7.2% |
| QM9, GTMGC | 10,661 | 0.264 | 0.462 | 0.931 | 8.2% |
| tmQMg, outliers removed | 1,322 | 0.890 | 1.351 | 2.316 | 15.7% |
| tmQMg, complete | 1,360 | 0.892 | 1.353 | 2.325 | 15.7% |
| BOSTMC low-spin | 12,150 | 0.964 | 1.496 | 2.532 | 16.1% |

Hydrogens are predicted throughout: every dataset carries explicit hydrogen and
the loss covers every atom. D-MAE and D-RMSE include them, here and in ReBind's
own `evaluate.py`. The RMSD column above, however, does *not* exclude them the
way the QM9 C-RMSD does — the MOL2 rows build no RDKit molecule, so
`Chem.RemoveHs` never ran. The evaluator now drops hydrogens from that path too,
by atom type, so reruns will report lower RMSDs than this table.

The three organometallic rows predate the full UFF Lennard-Jones table and were
trained with a flat sigma/epsilon for every element past Kr, so they are a floor
rather than a result. They want rerunning before the numbers travel anywhere.
The QM9 rows are unaffected: that table only covers Z=1..36 either way.

Organometallic error is about twice QM9's in relative terms, not the four times
raw D-MAE suggests. Global structure degrades further than pairwise distances do:
RMSD is 27% of the mean pairwise distance on QM9 against 41-42% on the
organometallic sets.

Training on tmQMg's 2,379 flagged-unphysical structures costs 0.7% D-MAE
(0.8902 vs 0.8962 on the identical outlier-free test set) and nothing on RMSD, so
the run that matches the published baselines is not meaningfully handicapped.

### Charge and spin conditioning is used but barely pays

`scripts/analyze_conditioning.py` re-scores a split with the molecule-level
features zeroed. On BOSTMC the conditioning is clearly active — perturbing charge
and spin moves predicted atoms by 1.3 A against a 3.3 A coordinate spread — but
zeroing it costs only 0.2% D-MAE overall, and at most ~2.5% on any well-populated
charge state. Open-shell doublets are not the hard case: they score 14.4%
relative against singlets' 16.5%.

So scalars added to the atom embeddings are the wrong lever, or at least a weak
one. Before spending another multi-day run on it, try conditioning that a
LayerNorm cannot wash out: FiLM-style scale and shift per block, or a global
token the attention has to read.

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
|   |-- common.py         # shared conditioning + out-of-place Laplacian helpers
|   |-- lj_params.py      # full UFF Lennard-Jones table (ReBind's stops at Kr)
|   |-- rebind.py         # thin wrapper over external/ReBIND + 3 runtime patches
|   |-- gtmgc.py          # thin wrapper over external/GTMGC + Mole-BERT tokenizer
|-- eval/
|   |-- metrics.py        # D-MAE, D-RMSE, coord-RMSD, per-element D-MAE
|   |-- conformer_eval.py # ReBind's published metric protocol
|-- train.py              # config-driven training loop
|-- cli.py                # `uv run step-up train|evaluate -c <yaml>`

configs/           # per-dataset YAML configs (smoke + full)
external/ReBIND/   # git submodule, vendored upstream ReBind
external/GTMGC/    # git submodule, vendored upstream GTMGC
scripts/train.sh   # Slurm job script (sbatch scripts/train.sh <config>)
scripts/prepare_qm9_rebind.py  # QM9 + ReBind's published split --> CSV
scripts/tokenize_molebert.py   # per-atom Mole-BERT ids, needed by GTMGC
tests/             # imports, dataset loading, MOL2 parsing, forward pass, metrics
```

## Data Path Notes

- **QM9 (sdf)**: molblocks copied out of `gdb9.sdf`, featurized with ReBind's
  own `mol_to_graph_dict`. Bonds and coordinates come from the record, so
  nothing is inferred. This is the path used to reproduce their numbers.
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
