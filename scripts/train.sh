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
# Jobs launched together share this clone, and git's config lock is not safe against
# concurrent writers (a second job dies with "could not lock config file"), so skip
# the update once the submodule is populated and serialize the setup steps.
SETUP_LOCK="$REPO_ROOT/.git/step-up-setup.lock"
if [[ ! -f external/ReBIND/models/rebind/modeling_rebind.py ]]; then
  flock "$SETUP_LOCK" git submodule update --init --recursive
fi

flock "$SETUP_LOCK" uv sync --dev

uv run step-up train -c "$CONFIG"
