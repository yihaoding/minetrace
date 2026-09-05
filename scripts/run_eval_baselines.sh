#!/bin/bash
#SBATCH --job-name=gad_baselines
#SBATCH --partition=ondemand,work
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#
# Expert tree vs decision tree / random forest / XGBoost / L1-logistic,
# all 9 metals x 4 negative-sampling strategies, identical held-out splits.
#
#   sbatch scripts/run_eval_baselines.sh

# NOTE: #SBATCH directives are not shell-expanded; log paths are
# relative to the directory you submit from.
set -euo pipefail
PROJ=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}
cd "$PROJ"

export HF_HOME="$PROJ/.cache/hf"
export XDG_CACHE_HOME="$PROJ/.cache"
export TMPDIR="$PROJ/.cache/tmp"
mkdir -p "$TMPDIR" logs

PY=${PYTHON_BIN:-python}
"$PY" -u scripts/eval_baselines.py --out "$PROJ/reports/eval_baselines_summary.json"
