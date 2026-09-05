#!/bin/bash
#SBATCH --job-name=ipfix_eval
#SBATCH --partition=work
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}/logs/ipfix_eval_%j.out
#SBATCH --error=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}/logs/ipfix_eval_%j.err
#
# Re-run both evals after the IP-01/02/03 inversion + attribution fix.
#   - eval_comprehensive     : did fixing the double inversion move AUC?
#                              (only metals with an inverted expert can move:
#                               Mn / REE per reports/eval_interpretability_summary.json)
#   - eval_interpretability  : narrative_weight_miss_frac must now be 0.0,
#                              narrative_view_gap must now be ~0.
set -euo pipefail

PROJ=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}
PY=/group/pmc050/yding/miniconda3/envs/geochem/bin/python

# keep every cache inside the project — never $HOME
export HF_HOME="$PROJ/.cache/hf"
export XDG_CACHE_HOME="$PROJ/.cache"
export TMPDIR="${TMPDIR:-$PROJ/tmp}"
mkdir -p "$PROJ/.cache" "$PROJ/tmp" "$PROJ/logs"

cd "$PROJ"

echo "===== eval_comprehensive (post-fix) ====="
"$PY" scripts/eval_comprehensive.py 2>&1 | tee "reports/eval_comprehensive_postipfix.log"

echo
echo "===== eval_interpretability (post-fix) ====="
"$PY" scripts/eval_interpretability.py 2>&1 | tee "reports/eval_interpretability_postipfix.log"

echo "done"
