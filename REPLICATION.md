# Replication log: CATfusion (Hu et al., 2025) in JAX

This file records how the JAX code maps onto the paper and the authors' released code
(github.com/Wanglabsmu/CATfusion, read October 2026). It also lists every place where the
paper, the code and this repository differ. It is meant to feed the Methods and Limitations
sections of the capstone paper.

## 1. Mapping: paper → released code → this repo

| Component | Paper | Released PyTorch code | This repo |
|---|---|---|---|
| Genomic encoder | TabAE: 1024-long windows, 40% masked, 4-block encoder, 2-block decoder, MSE on masked tokens (Eq. 1) | `TabAE_model/models/mae_model.py`: embed 1024, 16 heads, decoder 512, sinusoidal positions | `catfusion/models/tabae.py` (same sizes) |
| Omics preprocessing | "Supplementary File 1" | `processing_omic_data.py`, `getData()` | `catfusion/data/omics.py`, `sliding_windows()` |
| TabAE training | 4:1 split, TGCT held out | SGD momentum 0.9, wd 5e-5, lr 1e-3, cosine to 1e-5 | `catfusion/tabae_train.py`, `scripts/02_train_tabae.py` |
| Slide features | SISH mosaics + CTransPath, 768-d | not released ("use AttMIL to extract features") | `scripts/03_extract_wsi_features.py` |
| Slide tokens | — | 100 mosaics per slide, drawn at random each time, sorted by `2x + y` | `Cohort.sample_patches()` |
| Fusion model | 2 bidirectional cross-attention blocks + 10 self-attention blocks, 768-d | `models/*.py`: 12 heads, MLP 3072, dropout 0.3, attention dropout 0.2 | `catfusion/models/catfusion.py` (parameter-for-parameter port) |
| Output | single risk node, Cox loss (Eq. 4–5) | `[cls ; mean(tokens)]` → 64 → 16 → sigmoid × 8 − 4 | `ScaledSigmoidHead` |
| Optimiser | — | Adam lr 1e-5, weight decay 0.7, StepLR(50, γ = 1.2), batch 64 | `make_optimizer()`; `TrainConfig.paper()` |
| Cross-validation | 5-fold, "five replicates" | `StratifiedKFold(5, shuffle, random_state=24)` | `make_folds()` |
| Metrics | C-index, survival AUC, KM + log-rank at median | lifelines `concordance_index`, log-rank at median | `catfusion/metrics.py` (C-index matches scikit-survival; log-rank matches `compare_survival`) |
| Baselines | AttMIL-WSI, AttMIL-Genomic; concat / Kronecker / low-rank; RSF, Lasso-Cox, Cox on clinical covariates | not released | `catfusion/models/baselines.py`, `scripts/06_classical_baselines.py` |
| Interpretability | attention heatmaps, HoVer-Net cell types | not released | not implemented (out of scope for the proposal) |

`tests/test_core.py` checks the pieces the replication depends on: the Cox losses against
their definitions, the C-index and log-rank test against scikit-survival, the size of the
paper-scale model (about 100M parameters), the TabAE masking, and the windowing.

## 2. Findings from reading the released code

These matter for SQ1 (whether the published C-index can be reproduced). They also add to
the critical evaluation in proposal section B3.4.

**F1. The endpoint is progression-free interval, not overall survival.**
The training script reads `PFI` and `PFI.time` from the TCGA Clinical Data Resource
(Liu et al., 2018), and the Kaplan–Meier axes in Fig. 3a say "Progression Free Interval".
The text only says "survival". `scripts/04_build_cohort.py` defaults to PFI. `--endpoint OS`
exists for comparison.

**F2. Model selection uses the test fold.**
In `train_five_fold_cross_validation.py` the "validation" loader is the held-out test fold.
The checkpoint with the lowest loss on that fold is saved and later evaluated on the same
fold, which makes the reported C-index optimistic.
`--selection official` reproduces this. `--selection inner_val` holds out 15% of the
training fold instead. The `protocol` suite measures the difference.

**F3. The Cox loss builds risk sets from batch order, not survival time.**
`CoxLoss2` uses `torch.tril(ones)` as the risk-set indicator without sorting the batch by
time. Each patient's "risk set" is therefore whoever happens to come earlier in the shuffled
batch. It also divides by (events + 1). `cox_loss="official"` reproduces it exactly
(unit-tested against the torch formula), and `cox_loss="exact"` is the textbook partial
likelihood.

**F4. The per-cancer results come from one pan-cancer model.**
The released script loads all cancer types into one table, splits it stratified by cancer
type, and trains one model. The per-type C-indices (e.g. BRCA 0.724) are computed on
that model's predictions for each type.
Training on TCGA-BRCA alone (proposal B5.1) is therefore a deviation. It may land lower,
because the model sees about 1/10 of the training data, or higher, because it is
specialised. Every script takes `--projects`, so the same code runs on a pan-cancer
cohort and can report BRCA results from it. Running both answers SQ1 more cleanly.

**F5. Exported TabAE features come out in a random token order.**
The feature export calls `forward_encoder(mask_ratio=0.0)`. Its `random_masking` still
shuffles tokens with fresh noise, so each patient's 10 RNA tokens are permuted differently
before CATfusion adds its per-position embeddings.
We export tokens in their original order (`TabAE.features`).

**F6. The preprocessing and the model config disagree on token counts.**
The preprocessing script keeps 20,480 CpGs (20 tokens), but the CATfusion config expects
8 methylation tokens. For copy number, the preprocessing keeps 10,240 genes, while the
config expects 27 tokens (about 27k genes, i.e. all of them).
We follow the model config: 8,192 CpGs, all CNV genes.

**F7. Training length is not given, and the learning rate grows.**
The CLI default is `--epochs 1`, and the paper does not state the number of epochs.
`StepLR(step_size=50, gamma=1.2)` multiplies the learning rate by 1.2 every 50 epochs.
Weight decay 0.7 is added to the gradient inside Adam.
On the synthetic cohort, this weight decay pins the small model near zero within the
epochs we can afford on CPU. The smoke test therefore uses lr 3e-4 and weight decay 1e-4
(see section 4). For the real replication, start with the paper values and report any
change.

**F8. The equations sum the two attention paths; the code averages them.**
Paper Eq. 2–3 add the self- and cross-modal attention outputs. The code averages them,
`(ctx_self + ctx_cross) / 2`, and passes both modalities through one shared output
projection. We follow the code.

**F9. Possible patient leakage across folds.**
The released table has one row per slide file. TCGA patients can have several diagnostic
slides, so the same patient may appear in both training and test folds.
We split by patient and pool all of a patient's DX slides into one bag of patches.

**F10. Evaluation is not deterministic.**
Each evaluation draws a fresh random set of 100 mosaics. We fix the draw with a seed, so
predictions are reproducible.

**F11. Pooling predictions across folds changes the C-index.**
The paper computes one C-index over the pooled out-of-fold predictions ("five replicates on
the aggregated risk predictions"). Each fold's model puts risks on its own scale, so pooling
also scores how well the five models agree on scale, not only how well each ranks its own
test patients.
In the synthetic run the gap was large: genomics-only scored 0.610 pooled but 0.679 ± 0.028
per fold. `07_evaluate.py` reports both. Compare against the paper with the pooled number,
and use the per-fold number for model comparisons.

## 3. Data-source deviations

| Data | Paper / official code | This repo |
|---|---|---|
| Survival labels | TCGA-CDR (Liu et al., 2018), PFI | same (`00_download_gdc.py --kinds clinical`) |
| mRNA | GDC FPKM matrices (the code reads GDC sample sheets, even though the paper says Firehose) | GDC STAR-Counts `fpkm_unstranded` (current GDC release, hg38) |
| miRNA | GDC reads-per-million | same |
| Methylation | GDC 450K | same, top 8,192 CpGs by CV (F6) |
| Copy number | Firehose GISTIC2 `all_data_by_genes` | same; GDC gene-level copy number as a fallback |
| Mutations | MAF → gene × variant-type counts | GDC masked somatic MAF, same counting |
| Slides | SISH mosaics, CTransPath | SISH two-stage k-means mosaics re-implemented (9 colour clusters, 5% per cluster), CTransPath |

The paper used data downloaded in January 2023. The GDC has re-processed some data
types since then, so expect small differences in sample counts. Report the counts that
`04_build_cohort.py` prints and compare them with Supplementary Table 1.

## 4. Status

- [x] JAX port of TabAE, CATfusion, AttMIL baselines, three fusion operators
- [x] Official and exact Cox losses; official and clean model selection
- [x] Extensions: genomic modality dropout with a learned missing token; slide-only distillation
- [x] Evaluation: C-index, survival AUC, KM/log-rank, paired bootstrap, gap recovered
- [x] Data scripts: GDC download, omics preprocessing, SISH + CTransPath slide features
- [x] End-to-end run on a synthetic cohort (`results/synthetic_smoke_test/`)
- [ ] Download TCGA-BRCA and extract slide features (needs a machine with GDC access; slides take hundreds of GB)
- [ ] SQ1/SQ2 replication on real TCGA-BRCA data
- [ ] SQ3/SQ4 on TCGA-BRCA and TCGA-LUAD

The synthetic results show that the code runs end to end and recovers the planted signal.
They are **not** a replication result.
