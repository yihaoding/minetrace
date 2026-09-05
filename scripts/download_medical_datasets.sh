#!/bin/bash
#SBATCH --job-name=dl_medical
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}/logs/dl_medical_%j.out
#SBATCH --error=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}/logs/dl_medical_%j.err

# Medical AD benchmark data (decision 2026-08-14: medical elevated to main domain).
# 1) MedIAnomaly pre-processed sets (Zenodo 12677223, ~2.4 GB)
# 2) ISIC2018 Task3 (required separately by MedIAnomaly, ~3.2 GB, public S3)
# 3) BMAD via Google Drive folder (gdown; best-effort — falls back to manual if Drive blocks)

set -uo pipefail

ROOT=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}/datasets/medical
ARC=$ROOT/archives
PROJ=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}
FAIL=0
log() { echo "[$(date '+%F %T')] $*"; }

# ---------- MedIAnomaly (Zenodo) ----------
ZEN=https://zenodo.org/api/records/12677223/files
for f in BrainTumor BraTS2021 LAG RSNA VinCXR Camelyon16; do
    log "=== MedIAnomaly: $f ==="
    if wget -c -q "$ZEN/$f.tar.gz/content" -O "$ARC/$f.tar.gz"; then
        tar -xzf "$ARC/$f.tar.gz" -C "$ROOT/medianomaly/" && log "$f OK" || { log "$f EXTRACT FAILED"; FAIL=1; }
    else
        log "$f DOWNLOAD FAILED"; FAIL=1
    fi
done

# ---------- ISIC2018 Task3 (public S3) ----------
mkdir -p "$ROOT/medianomaly/ISIC2018_Task3"
for f in Training_Input Training_GroundTruth Test_Input Test_GroundTruth; do
    log "=== ISIC2018_Task3_$f ==="
    if wget -c -q "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_$f.zip" \
            -O "$ARC/ISIC2018_Task3_$f.zip"; then
        unzip -oq "$ARC/ISIC2018_Task3_$f.zip" -d "$ROOT/medianomaly/ISIC2018_Task3/" \
            && log "ISIC $f OK" || { log "ISIC $f EXTRACT FAILED"; FAIL=1; }
    else
        log "ISIC $f DOWNLOAD FAILED"; FAIL=1
    fi
done

# ---------- BMAD (Google Drive, best-effort) ----------
log "=== BMAD: installing gdown to project dir ==="
GD=$PROJ/.cache/pip-tools
mkdir -p "$GD"
export PIP_CACHE_DIR=$PROJ/.cache/pip
/group/pmc050/yding/miniconda3/envs/geochem/bin/python -m pip install -q --target "$GD" gdown \
    && export PYTHONPATH="$GD" \
    || { log "gdown INSTALL FAILED"; FAIL=1; }
if [ -d "$GD/gdown" ]; then
    log "=== BMAD: downloading Drive folder ==="
    /group/pmc050/yding/miniconda3/envs/geochem/bin/python -m gdown --folder \
        'https://drive.google.com/drive/folders/1AC-wWZl_K18CWL2eIxUScoSOoxT4IBuw' \
        -O "$ROOT/bmad" --remaining-ok \
        && log "BMAD Drive OK" || { log "BMAD DRIVE FAILED (quota/permission) — needs manual download"; FAIL=1; }
fi

log "=== Summary ==="
du -sh "$ROOT"/medianomaly/* "$ROOT"/bmad/* 2>/dev/null
log "Done. FAIL=$FAIL"
exit $FAIL
