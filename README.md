# Capstone: slide-only cancer survival prediction with a histology–genomic cross-attention model

Alexia Moreanu · Minerva University NS Capstone · Advisor: Prof. Jon Wilkins

This repository holds a JAX reimplementation of **CATfusion** (Hu et al., *Briefings in
Bioinformatics* 2025, [doi:10.1093/bib/bbaf121](https://doi.org/10.1093/bib/bbaf121)) and the
capstone extension: keeping the benefit of genomic training data for patients who only
have a tissue slide.

| Sub-question | What answers it here |
|---|---|
| **SQ1** Can a JAX implementation reproduce CATfusion's TCGA-BRCA C-index (0.724 ± 0.027)? | `05_run_experiments.py --suite replication`, `07_evaluate.py` |
| **SQ2** Does fusion beat slide-only and genomics-only on the same folds? | paired bootstrap in `07_evaluate.py` |
| **SQ3** With genomics missing at test time, how much of the fusion advantage is kept? | `--suite extension` (modality dropout, distillation), gap recovered |
| **SQ4** Do the results hold across parameters and in TCGA-LUAD? | sweep flags, `--projects TCGA-LUAD` |

> **Status (Oct 2026).** The full pipeline is implemented and tested, and it runs end to
> end on a synthetic TCGA-shaped cohort ([results](results/synthetic_smoke_test/report.md)).
> No TCGA results are in this repo yet. The real data must be downloaded on a machine
> with GDC access (see *Running on TCGA-BRCA*). **[REPLICATION.md](REPLICATION.md)**
> lists how the code maps to the paper, and ten findings from the authors' released code
> that affect the replication. Read that before interpreting any numbers.

## Repository layout

```
catfusion/                 the library (JAX/Flax)
  config.py                paper hyper-parameters (copied from the released code) + tiny CPU preset
  models/catfusion.py      CATfusion: bidirectional cross-attention + self-attention, Cox head,
                           optional learned "missing genomics" token (extension A)
  models/tabae.py          TabAE masked autoencoder for omics + official sliding-window preprocessing
  models/baselines.py      AttMIL-WSI, AttMIL-Genomic, concat / Kronecker / low-rank fusion
  losses.py                Cox partial likelihood (exact and the released variant), distillation loss
  metrics.py               C-index, survival AUC, Kaplan-Meier, log-rank, paired bootstrap, gap recovered
  data/omics.py            official omics filters (CV ranking, expression filters, mutation counts)
  data/cohort.py           cohort loading, 100-mosaic sampling, patient-level folds
  train.py                 training loop and repeated K-fold cross-validation (incl. extensions)
  tabae_train.py           self-supervised TabAE training and feature export
scripts/
  00_download_gdc.py       TCGA clinical (TCGA-CDR), omics and slide manifests from the GDC
  01_preprocess_omics.py   raw files -> filtered, z-scored, windowed omics
  02_train_tabae.py        TabAE per omics type -> genomic tokens
  03_extract_wsi_features.py  slides -> SISH mosaics -> CTransPath features (PyTorch; batch download + delete)
  04_build_cohort.py       join PFI labels, slide features, genomic features
  05_run_experiments.py    replication / fusion / protocol / extension suites
  06_classical_baselines.py  Cox (clinical), random survival forest, Lasso-Cox
  07_evaluate.py           tables, bootstrap comparisons, gap recovered, KM and missingness figures
  make_synthetic.py        synthetic cohort with known ground truth
  run_synthetic_smoke_test.sh  everything above, end to end, on synthetic data
tests/                     unit tests (pytest)
results/                   run outputs (predictions, histories, reports)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # JAX side (training, evaluation)
pip install -r requirements-slides.txt   # only for step 3 (slide features)
pytest -q                                # 16 tests, ~1 min on CPU
```

## Quick check on synthetic data

```bash
./scripts/run_synthetic_smoke_test.sh     # CPU; writes results/synthetic_smoke_test/report.md
```

It uses a small model (hidden 64), 25 epochs, 1 repeat, and lr 3e-4 / weight decay 1e-4. At the paper's
weight decay of 0.7 the small model cannot move away from zero in that few epochs (REPLICATION.md, F7).
It took about 2.5 hours on a 2-core CPU. On synthetic data the expected ordering comes out
(fusion 0.698 > slide only 0.671 > genomics only 0.610; KM split p < 1e-16). See the
[report](results/synthetic_smoke_test/report.md).

## Running on TCGA-BRCA

The GDC has to be reachable, so run this from your own Terminal or from Colab.

**Short version:** `scripts/run_tcga.sh` runs the steps below in stages:

```bash
./scripts/run_tcga.sh setup            # venv + packages + tests
./scripts/run_tcga.sh omics            # clinical + omics -> TabAE tokens (no slides)
./scripts/run_tcga.sh slides-pilot 20  # time 20 slides, prints an estimate for all of them
./scripts/run_tcga.sh slides           # all slides, downloaded 10 at a time and deleted after extraction
./scripts/run_tcga.sh cohort && ./scripts/run_tcga.sh train replication && ./scripts/run_tcga.sh report
```

On Colab, open [`notebooks/run_tcga_colab.ipynb`](notebooks/run_tcga_colab.ipynb). It keeps data and results
in Google Drive, so a dropped session resumes where it stopped.

**Long version** (paths are examples):

```bash
# 0. clinical + omics (a few GB) and the slide manifest
python scripts/00_download_gdc.py --project TCGA-BRCA --out data/raw/TCGA-BRCA \
    --kinds clinical rna_seq mirna_seq dnamethylationvariation copynumbervariation snp
python scripts/00_download_gdc.py --project TCGA-BRCA --out data/raw/TCGA-BRCA --kinds slides --manifest-only

# 1-2. omics -> TabAE tokens
python scripts/01_preprocess_omics.py --raw data/raw/TCGA-BRCA --out data/processed/TCGA-BRCA \
    --omics rna_seq mirna_seq copynumbervariation dnamethylationvariation snp
python scripts/02_train_tabae.py --windows data/processed/TCGA-BRCA/windows --cohort data/cohort/TCGA-BRCA \
    --omics rna_seq mirna_seq copynumbervariation dnamethylationvariation snp --preset paper

# 3. slides: download 10 at a time, extract, delete (pilot first with --limit 20 and time it)
#    CTransPath weights: ctranspath.pth from github.com/Xiyue-Wang/TransPath
python scripts/03_extract_wsi_features.py --manifest data/raw/TCGA-BRCA/manifest_slides.tsv \
    --out data/cohort/TCGA-BRCA/slides --weights checkpoints/ctranspath.pth --device mps --delete --limit 20

# 4. labels (PFI, as in the paper)
python scripts/04_build_cohort.py --cdr data/raw/TCGA-BRCA/TCGA-CDR-SupplementalTableS1.xlsx \
    --cohort data/cohort/TCGA-BRCA --projects TCGA-BRCA

# 5-7. experiments and report
python scripts/05_run_experiments.py --cohort data/cohort/TCGA-BRCA --out results/brca --suite replication \
    --preset paper --epochs 100 --repeats 5          # paper recipe: lr 1e-5, wd 0.7, official selection + Cox loss
python scripts/05_run_experiments.py --cohort data/cohort/TCGA-BRCA --out results/brca --suite fusion --preset paper
python scripts/05_run_experiments.py --cohort data/cohort/TCGA-BRCA --out results/brca --suite protocol --preset paper
python scripts/05_run_experiments.py --cohort data/cohort/TCGA-BRCA --out results/brca --suite extension --preset paper
python scripts/06_classical_baselines.py --cohort data/cohort/TCGA-BRCA --out results/brca
python scripts/07_evaluate.py --results results/brca --project TCGA-BRCA
```

**Compute.** The paper-size model has about 100M parameters, and each patient is about
140 tokens. Steps 0–2 and 4–7 need no slides. Step 3 is the expensive one; the PyTorch
extractor uses the Mac GPU with `--device mps`. For the repeated 5-fold runs at paper
scale, a GPU/TPU (Colab, TPU Research Cloud) is much faster than a laptop CPU. JAX runs
the same code there unchanged (`pip install "jax[cuda12]"` or `"jax[tpu]"`).

## Reference

Hu, Y., Li, X., Yi, Y., Huang, Y., Wang, G., & Wang, D. (2025). Deep learning-driven survival prediction in
pan-cancer studies by integrating multimodal histology-genomic data. *Briefings in Bioinformatics, 26*(2), bbaf121.
Official PyTorch code: https://github.com/Wanglabsmu/CATfusion
