#!/bin/bash
#SBATCH --job-name=dl_datasets
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}/logs/dl_datasets_%j.out
#SBATCH --error=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}/logs/dl_datasets_%j.err

# Download AD benchmark datasets for the falsification-gated agentic AD project.
# MVTec AD 2 is NOT here: its download link requires registration on
# https://www.mvtec.com/company/research/datasets/mvtec-ad-2 — add it below once obtained.

set -uo pipefail

ROOT=${PROJECT_ROOT:-/mmfs1/data/group/pmc050/yding/gad_reasoning}/datasets
ARC=$ROOT/industrial/archives
FAIL=0

LOCO_URL='https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz'
VISA_URL='https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar'
EXATHLON_REPO='https://github.com/exathlonbenchmark/exathlon.git'

log() { echo "[$(date '+%F %T')] $*"; }

# ---------- MVTec LOCO AD (~6.1 GB tar.xz) ----------
log "=== MVTec LOCO AD: download ==="
if wget -c -q --show-progress --progress=dot:giga "$LOCO_URL" \
        -O "$ARC/mvtec_loco_anomaly_detection.tar.xz"; then
    log "=== MVTec LOCO AD: extract ==="
    if tar -xf "$ARC/mvtec_loco_anomaly_detection.tar.xz" -C "$ROOT/industrial/mvtec_loco_ad/"; then
        log "LOCO OK"
    else
        log "LOCO EXTRACT FAILED"; FAIL=1
    fi
else
    log "LOCO DOWNLOAD FAILED"; FAIL=1
fi

# ---------- VisA (~1.9 GB tar) ----------
log "=== VisA: download ==="
if wget -c -q --show-progress --progress=dot:giga "$VISA_URL" \
        -O "$ARC/VisA_20220922.tar"; then
    log "=== VisA: extract ==="
    if tar -xf "$ARC/VisA_20220922.tar" -C "$ROOT/industrial/visa/"; then
        log "VisA OK"
    else
        log "VisA EXTRACT FAILED"; FAIL=1
    fi
else
    log "VisA DOWNLOAD FAILED"; FAIL=1
fi

# ---------- Exathlon (git repo, data zips in data/raw) ----------
log "=== Exathlon: clone ==="
if [ -d "$ROOT/time_sequence/exathlon/.git" ]; then
    log "Exathlon repo already present, pulling"
    git -C "$ROOT/time_sequence/exathlon" pull --ff-only || true
else
    git clone --depth 1 "$EXATHLON_REPO" "$ROOT/time_sequence/exathlon" || { log "EXATHLON CLONE FAILED"; FAIL=1; }
fi
if [ -d "$ROOT/time_sequence/exathlon/data" ]; then
    log "=== Exathlon: extract data zips ==="
    find "$ROOT/time_sequence/exathlon/data" -name '*.zip' -print -execdir unzip -oq {} \; || FAIL=1
fi

# ---------- Summary ----------
log "=== Summary ==="
du -sh "$ARC"/* "$ROOT/industrial/mvtec_loco_ad"/* "$ROOT/industrial/visa"/* \
      "$ROOT/time_sequence/exathlon" 2>/dev/null
log "Done. FAIL=$FAIL"
exit $FAIL
