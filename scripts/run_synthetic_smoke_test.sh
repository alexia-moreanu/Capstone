#!/usr/bin/env bash
# End-to-end check of the whole pipeline on synthetic data (CPU, small model).
# Results go to results/synthetic_smoke_test/. They are NOT TCGA results.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/synthetic
OUT=results/synthetic_smoke_test
OMICS="rna_seq mirna_seq copynumbervariation dnamethylationvariation snp"
EPOCHS=${EPOCHS:-25}
REPEATS=${REPEATS:-1}

python scripts/make_synthetic.py --out $D --patients 500
python scripts/01_preprocess_omics.py --out $D/processed --omics $OMICS
python scripts/02_train_tabae.py --windows $D/processed/windows --cohort $D/cohort --omics $OMICS --preset tiny --epochs 8
python scripts/04_build_cohort.py --cdr $D/raw/synthetic_cdr.csv --cohort $D/cohort --projects TCGA-SYN
python scripts/05_run_experiments.py --cohort $D/cohort --out $OUT --suite all --preset tiny \
    --omics $OMICS --epochs $EPOCHS --repeats $REPEATS --batch-size 64 --lr 3e-4 --weight-decay 1e-4
python scripts/06_classical_baselines.py --cohort $D/cohort --out $OUT --omics $OMICS --repeats $REPEATS
python scripts/07_evaluate.py --results $OUT --project TCGA-SYN --n-boot 500 \
    --note "**Synthetic data.** These numbers test the code path only; they are not TCGA results and are not comparable to the paper."
