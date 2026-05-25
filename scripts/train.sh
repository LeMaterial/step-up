#!/bin/bash
#SBATCH --job-name=step-up
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=train.out

# Usage: sbatch scripts/train.slurm configs/qm9.yaml
#   - Edit partition / gres / time to match your cluster.
#   - The job runs from the repo root and uses uv for everything.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml>"
  exit 2
fi
CONFIG="$1"

cd "$(dirname "$0")/.."

# Make sure the submodule is initialized in case the job runs on a fresh checkout.
git submodule update --init --recursive

uv sync --dev

uv run step-up train -c "$CONFIG"
