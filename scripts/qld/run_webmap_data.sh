#!/bin/bash
#SBATCH --job-name=qld_webmap
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}/logs/qld_webmap_%j.out
#SBATCH --error=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}/logs/qld_webmap_%j.err
cd ${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}
export TMPDIR=$PWD/.cache/tmp ABLATE_LAYERS=geochem NPROC=8 OMP_NUM_THREADS=1
${PYTHON_BIN:-python} -u scripts/qld/build_webmap_data.py Cu Au
