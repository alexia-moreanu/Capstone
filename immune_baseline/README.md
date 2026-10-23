# Immune signatures as an interpretable survival baseline (TCGA-LUAD)

Alexia Moreanu · IL181 Molecular Immunology, Assignment 1 · capstone extension

This folder asks how much overall survival in lung adenocarcinoma can be predicted from five
immune gene signatures, after adjusting for age and stage. I use it in the capstone as
(1) an interpretable baseline that CATfusion has to beat on TCGA-LUAD and (2) a probe for what
the slide and genomic branches learn.

| Signature | Genes | Immune process |
|---|---|---|
| cDC1 | CLEC9A, XCR1, BATF3 | cross-presentation and CD8⁺ T cell priming |
| Cytolytic | GZMA, PRF1 | cytotoxic killing |
| TLS | CXCL13, CCL19, CCL21, LAMP3, MS4A1 | tertiary lymphoid structures |
| MHC-I | HLA-A/B/C, B2M, TAP1 | antigen presentation |
| Treg | FOXP3, IL2RA, CTLA4 | suppression |

## Main results (485 patients, 177 deaths)

- Only cDC1 is independently associated with survival: HR 0.83 per SD (95% CI 0.72–0.97,
  p = 0.018; FDR-adjusted p = 0.09). Kaplan–Meier log-rank p = 0.009.
- Cross-validated C-index: age + stage 0.672, age + stage + cDC1 **0.677**,
  age + stage + all five signatures 0.665 (overfitting).

## Files

```
immune_signatures.py         loading, signature scoring, Cox models, C-index (numpy)
luad_immune_baseline.ipynb   the full analysis, with outputs
figures/                     the four figures in the write-up
results/                     per-patient signature scores, adjusted Cox table, C-index table
```

## Running it

Download from the [GDC TCGA-LUAD cohort on UCSC Xena](https://xenabrowser.net) into
`immune_baseline/data/` (ignored by git):

- `TCGA-LUAD.star_tpm.tsv.gz` (STAR - TPM)
- `gencode.v36.annotation.gtf.gene.probemap` (its ID/gene mapping)
- `TCGA-LUAD.clinical.tsv.gz` (phenotype)

Then run the notebook from this folder. Requirements: pandas, numpy, statsmodels,
scikit-learn, scipy, matplotlib. The first run reads only the 18 signature genes from the
60k-gene matrix and caches them as `data/luad_signature_genes.tsv`.

`results/luad_immune_scores.tsv` is keyed by TCGA patient barcode, so it joins directly with
the capstone's model predictions.
