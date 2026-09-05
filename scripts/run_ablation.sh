#!/bin/bash
# Run expert-layer ablation: 5 configs that skip various layers.
# The baseline (all 3 layers) is assumed to already exist as
# reports/eval_comprehensive_summary.json + reports/eval_scores.json.
#
# Each run takes ~5-8 minutes; sequential total ~25-40 minutes.

set -e
cd ${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}
mkdir -p reports/ablation_logs

declare -a CFGS=(
    "geochem"
    "geophys"
    "structure"
    "geochem+geophys"
    "geochem+structure"
)

for cfg in "${CFGS[@]}"; do
    tag="${cfg//+/_}"
    log="reports/ablation_logs/eval_${tag}.log"
    echo "─────────────────────────────────────────────────"
    echo "  ABLATE_LAYERS=$cfg → $log"
    echo "─────────────────────────────────────────────────"
    ABLATE_LAYERS="$cfg" conda run -n geochem python scripts/eval_comprehensive.py > "$log" 2>&1
    tail -16 "$log"
done

echo
echo "All ablation runs complete. JSONs:"
ls -la reports/eval_comprehensive_summary_*.json
