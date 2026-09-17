#!/bin/bash
#SBATCH --job-name=step-up
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=train-%j.out

# Usage: sbatch scripts/train.sh configs/qm9.yaml
#   - Edit partition / gres / time to match your cluster.
#   - The job runs from the repo root and uses uv for everything.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml>"
  exit 2
fi
# Resolve the config before changing directory so relative paths keep working.
CONFIG="$(realpath "$1")"

# Under sbatch, $0 is Slurm's spooled copy of this script (in the slurmd spool
# directory), not the file in the repo, so locate the repo from the submission
# directory instead. Outside Slurm, fall back to the script's own location.
REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Make sure the submodule is initialized in case the job runs on a fresh checkout.
git submodule update --init --recursive

uv sync --dev

uv run step-up train -c "$CONFIG"
