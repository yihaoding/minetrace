#!/bin/bash
#SBATCH --job-name=gad_all_evals
#SBATCH --partition=ondemand,work
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/all_evals_%j.out
#SBATCH --error=logs/all_evals_%j.err
#
# 一次跑完论文用到的全部评测。约 3.5 小时。
#   sbatch scripts/run_all_evals.sh
#
# 每一步都会把结果写到 reports/ 下的 JSON；中途失败可注释掉已完成的行重跑。

# NOTE: #SBATCH directives are not shell-expanded; log paths are
# relative to the directory you submit from.
set -euo pipefail
PROJ=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}
cd "$PROJ"

export HF_HOME="$PROJ/.cache/hf"
export XDG_CACHE_HOME="$PROJ/.cache"
export TMPDIR="$PROJ/.cache/tmp"
export PYTHONHASHSEED=0
mkdir -p "$TMPDIR" logs reports

PY=${PYTHON_BIN:-python}

echo "=== [1/5] 专家树主评测（9 金属 × 4 负采样） ==="
"$PY" -u scripts/eval_comprehensive.py

echo "=== [2/5] 五模型性能对比 ==="
"$PY" -u scripts/eval_baselines.py --out "$PROJ/reports/eval_baselines_summary.json"

echo "=== [3/5] 专家树自测可解释性 ==="
"$PY" -u scripts/eval_interpretability.py

echo "=== [4/5] 五模型同口径解释质量 ==="
"$PY" -u scripts/eval_explanation_quality.py --out "$PROJ/reports/eval_explanation_quality.json"

echo "=== [5/5] 缺失模态衰减 + 选择性预测 ==="
"$PY" -u scripts/eval_degradation.py --out "$PROJ/reports/eval_degradation.json"
"$PY" -u scripts/eval_confidence.py  --out "$PROJ/reports/eval_confidence.json"

echo "=== 生成报告 ==="
"$PY" -u scripts/report_baselines.py

echo "全部完成。产出："
ls -la reports/eval_*.json reports/*.md | tail -20
