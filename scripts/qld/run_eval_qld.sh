#!/bin/bash
#SBATCH --job-name=eval_qld
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}/logs/eval_qld_%j.out
#SBATCH --error=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}/logs/eval_qld_%j.err
cd ${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}
export TMPDIR=$PWD/.cache/tmp ABLATE_LAYERS=geochem OMP_NUM_THREADS=8
${PYTHON_BIN:-python} -u scripts/qld/eval_comprehensive_qld.py
