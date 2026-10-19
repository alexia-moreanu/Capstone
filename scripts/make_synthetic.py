"""Generate a synthetic TCGA-like cohort to test the whole pipeline end to end.

    python scripts/make_synthetic.py --out data/synthetic --patients 500

This is NOT real data and its results say nothing about cancer. It exists so every
step (omics filtering -> TabAE -> CATfusion -> evaluation) can be run and checked
before the real TCGA-BRCA download finishes.

Ground truth: each patient has three hidden factors.
    shared  - visible in both the slide and the genomics
    slide   - visible only in the slide
    genomic - visible only in the genomics
    log-hazard = 0.7*shared + 0.5*slide + 0.6*genomic
so a correct pipeline should rank: fusion > either single modality.
Files are written in the same formats the real pipeline produces after step 0/3.
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import pandas as pd


def barcode(i):
    return f"TCGA-S{i // 10000}-{i % 10000:04d}"


def expression(rng, n_feat, n, loadings, scale, positive=True):
    """features x samples matrix with a few blocks of features driven by the hidden factors."""
    base = rng.normal(2.0, 1.0, (n_feat, 1))
    x = base + rng.normal(0, 0.6, (n_feat, n))
    for start, factor in loadings:
        x[start:start + 150] += scale * factor[None, :] * rng.choice([-1, 1], (150, 1))
    return np.exp(x) if positive else x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patients", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    n = args.patients
    ids = [barcode(i) for i in range(n)]
    samples = [f"{p}-01A" for p in ids]
    shared, slide_only, genomic = rng.normal(size=(3, n))
    log_hazard = 0.7 * shared + 0.5 * slide_only + 0.6 * genomic

    # ---- survival (Weibull) with uniform censoring
    t_event = (-np.log(rng.uniform(size=n)) / (3e-5 * np.exp(log_hazard))) ** (1 / 1.2)
    t_censor = rng.uniform(300, 4000, n)
    time = np.minimum(t_event, t_censor).round() + 1
    event = (t_event <= t_censor).astype(int)
    stage = np.digitize(log_hazard + rng.normal(0, 1, n), [-1, 0, 1])
    cdr = pd.DataFrame({"bcr_patient_barcode": ids, "type": "SYN", "PFI": event, "PFI.time": time,
                        "OS": event, "OS.time": time, "age_at_initial_pathologic_diagnosis": rng.normal(58, 13, n).round(),
                        "gender": np.where(rng.uniform(size=n) < 0.98, "FEMALE", "MALE"),
                        "ajcc_pathologic_tumor_stage": [f"Stage {'I' * (s + 1) if s < 3 else 'IV'}" for s in stage]})
    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    cdr.to_csv(os.path.join(args.out, "raw", "synthetic_cdr.csv"), index=False)

    # ---- omics matrices (features x samples), as stage A of 01_preprocess_omics.py would write them
    mats = os.path.join(args.out, "processed", "matrices")
    os.makedirs(mats, exist_ok=True)
    gene = lambda k, prefix="G": [f"{prefix}{i}" for i in range(k)]  # noqa: E731
    pd.DataFrame(expression(rng, 12000, n, [(0, shared), (300, genomic)], 0.8), index=gene(12000), columns=samples
                 ).to_pickle(os.path.join(mats, "rna_seq.pkl"))
    pd.DataFrame(expression(rng, 1900, n, [(0, genomic)], 0.8), index=gene(1900, "hsa-mir-"), columns=samples
                 ).to_pickle(os.path.join(mats, "mirna_seq.pkl"))
    beta = 1 / (1 + np.exp(-expression(rng, 9000, n, [(0, genomic), (300, shared)], 0.8, positive=False) + 2))
    pd.DataFrame(beta, index=gene(9000, "cg"), columns=samples).to_pickle(os.path.join(mats, "dnamethylationvariation.pkl"))
    cnv = expression(rng, 4000, n, [(0, genomic)], 0.4, positive=False) - 2.0
    pd.DataFrame(cnv * 0.3, index=gene(4000), columns=samples).to_pickle(os.path.join(mats, "copynumbervariation.pkl"))
    rows = []
    for j, s in enumerate(samples):
        k = rng.poisson(40 * np.exp(0.5 * genomic[j]))
        rows += [(f"G{g}", rng.choice(["SNP", "DEL", "INS"], p=[0.8, 0.12, 0.08]), s)
                 for g in rng.integers(0, 3000, k)]
    pd.DataFrame(rows, columns=["Hugo_Symbol", "Variant_Type", "Tumor_Sample_Barcode"]).to_pickle(
        os.path.join(mats, "snp.pkl"))

    # ---- slide features (as 03_extract_wsi_features.py writes them)
    slide_dir = os.path.join(args.out, "cohort", "slides")
    os.makedirs(slide_dir, exist_ok=True)
    directions = rng.normal(size=(2, 768)) / np.sqrt(768) * 6
    for j, p in enumerate(ids):
        k = int(rng.integers(60, 300))
        feats = rng.normal(0, 1, (k, 768)).astype(np.float32)
        tumour = rng.uniform(size=k) < 0.3  # only some mosaics carry prognostic signal
        feats[tumour] += shared[j] * directions[0] + slide_only[j] * directions[1]
        coords = np.stack([rng.integers(0, 80, k), rng.integers(0, 60, k)], 1) * 1024
        with h5py.File(os.path.join(slide_dir, f"{p}-01Z-00-DX1.SYNTHETIC.h5"), "w") as f:
            f.create_dataset("features", data=feats)
            f.create_dataset("coords", data=coords)
    print(f"synthetic cohort: {n} patients, event rate {event.mean():.2f}, written to {args.out}")


if __name__ == "__main__":
    main()
