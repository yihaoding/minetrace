#!/bin/bash
#SBATCH --job-name=qld_convert
#SBATCH --partition=work
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}/logs/qld_convert_%j.out
#SBATCH --error=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}/logs/qld_convert_%j.err
cd ${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}
export TMPDIR=$PWD/.cache/tmp
${PYTHON_BIN:-python} scripts/qld/convert_qld_assays.py
