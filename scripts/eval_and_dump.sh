#!/bin/bash
#SBATCH --job-name=step-up-eval
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=eval-%j.out

# Usage: sbatch scripts/eval_and_dump.sh configs/bostmc.yaml
#
# Scores the test split with ReBind's metric definitions and writes the
# per-molecule true/predicted structures beside it. Both read the same
# checkpoint and the same split, so the pooled metrics and the CSV always
# describe one state of one model.
#
# RMSD is reported twice on purpose: once hydrogen-free (comparable with the
# published C-RMSD column) and once keeping hydrogen, so a model trained with
# explicit hydrogen can be compared with its heavy-atom twin on equal terms.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml>"
  exit 2
fi
CONFIG="$(realpath "$1")"

REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

SETUP_LOCK="$REPO_ROOT/.git/step-up-setup.lock"
if [[ ! -f external/ReBIND/models/rebind/modeling_rebind.py ]]; then
  flock "$SETUP_LOCK" git submodule update --init --recursive
fi
flock "$SETUP_LOCK" uv sync --dev

OUT_DIR="$(uv run python -c "
from step_up.train import TrainConfig
print(TrainConfig.from_yaml('$CONFIG').output_dir)
")"

uv run step-up evaluate -c "$CONFIG" --split test --out "$OUT_DIR/eval_test.json"
uv run step-up evaluate -c "$CONFIG" --split test --keep-hs \
  --out "$OUT_DIR/eval_test_keephs.json"
uv run python scripts/dump_structures.py -c "$CONFIG" --split test
