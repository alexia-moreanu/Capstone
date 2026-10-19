#!/usr/bin/env bash
# Run the real TCGA pipeline in stages. Each stage can be re-run; finished work is skipped.
#
#   ./scripts/run_tcga.sh setup                 # virtual env + packages
#   ./scripts/run_tcga.sh omics                 # clinical + 5 omics -> TabAE tokens       (a few GB)
#   ./scripts/run_tcga.sh slides-pilot 20       # 20 slides end to end; prints time per slide
#   ./scripts/run_tcga.sh slides                # all diagnostic slides, 10 at a time, images deleted
#   ./scripts/run_tcga.sh cohort                # join PFI labels + features
#   ./scripts/run_tcga.sh train [suite]         # replication | fusion | protocol | extension | all
#   ./scripts/run_tcga.sh report
#
# Settings (override like PROJECT=TCGA-LUAD ./scripts/run_tcga.sh omics):
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT=${PROJECT:-TCGA-BRCA}
OMICS=${OMICS:-"rna_seq mirna_seq copynumbervariation dnamethylationvariation snp"}
WEIGHTS=${WEIGHTS:-checkpoints/ctranspath.pth}   # from github.com/Xiyue-Wang/TransPath
DEVICE=${DEVICE:-}                               # mps on Apple silicon, cuda on Colab; auto if empty
DOWNLOAD_DIR=${DOWNLOAD_DIR:-data/tmp_slides}    # where each slide waits until it is processed
EPOCHS=${EPOCHS:-100}
REPEATS=${REPEATS:-5}
RAW=data/raw/$PROJECT
PROC=data/processed/$PROJECT
COHORT=data/cohort/$PROJECT
RESULTS=results/${PROJECT#TCGA-}
RESULTS=$(echo "$RESULTS" | tr '[:upper:]' '[:lower:]')

[ -d .venv ] && source .venv/bin/activate || true
device_flag() { [ -n "$DEVICE" ] && echo "--device $DEVICE" || true; }

need_weights() {
  if [ ! -f "$WEIGHTS" ]; then
    echo "CTransPath weights not found at $WEIGHTS."
    echo "Download ctranspath.pth (link in the README of github.com/Xiyue-Wang/TransPath) and put it there,"
    echo "or set WEIGHTS=/path/to/ctranspath.pth."
    exit 1
  fi
}

case "${1:-help}" in
  setup)
    python3 -m venv .venv && source .venv/bin/activate
    pip install -U pip
    pip install -r requirements.txt -r requirements-slides.txt
    python -m pytest -q tests
    ;;
  omics)
    python scripts/00_download_gdc.py --project "$PROJECT" --out "$RAW" --kinds clinical $OMICS
    python scripts/00_download_gdc.py --project "$PROJECT" --out "$RAW" --kinds slides --manifest-only
    python scripts/01_preprocess_omics.py --raw "$RAW" --out "$PROC" --omics $OMICS
    python scripts/02_train_tabae.py --windows "$PROC/windows" --cohort "$COHORT" --omics $OMICS --preset paper
    ;;
  slides-pilot)
    need_weights
    N=${2:-20}
    start=$(date +%s)
    python scripts/03_extract_wsi_features.py --manifest "$RAW/manifest_slides.tsv" --out "$COHORT/slides" \
        --weights "$WEIGHTS" $(device_flag) --download-dir "$DOWNLOAD_DIR" --delete --limit "$N"
    end=$(date +%s)
    total=$(python -c "import pandas as p; m=p.read_csv('$RAW/manifest_slides.tsv', sep='\t'); print(m.file_name.str.contains('DX').sum())")
    echo "Pilot: $N slides in $((end - start)) s -> about $(( (end - start) * total / N / 3600 )) h for all $total DX slides"
    ;;
  slides)
    need_weights
    python scripts/03_extract_wsi_features.py --manifest "$RAW/manifest_slides.tsv" --out "$COHORT/slides" \
        --weights "$WEIGHTS" $(device_flag) --download-dir "$DOWNLOAD_DIR" --delete --batch 10
    ;;
  cohort)
    python scripts/04_build_cohort.py --cdr "$RAW/TCGA-CDR-SupplementalTableS1.xlsx" --cohort "$COHORT" \
        --projects "$PROJECT" --endpoint PFI
    ;;
  train)
    python scripts/05_run_experiments.py --cohort "$COHORT" --out "$RESULTS" --suite "${2:-replication}" \
        --preset paper --omics $OMICS --epochs "$EPOCHS" --repeats "$REPEATS"
    ;;
  baselines)
    python scripts/06_classical_baselines.py --cohort "$COHORT" --out "$RESULTS" --omics $OMICS --repeats "$REPEATS"
    ;;
  report)
    python scripts/07_evaluate.py --results "$RESULTS" --project "$PROJECT"
    ;;
  *)
    sed -n '2,12p' "$0"
    ;;
esac
