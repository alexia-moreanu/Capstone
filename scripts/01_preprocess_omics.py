"""Step 1 - turn downloaded omics files into TabAE input windows.

    python scripts/01_preprocess_omics.py --raw data/raw/TCGA-BRCA --out data/processed \
        --omics rna_seq mirna_seq copynumbervariation dnamethylationvariation snp

Stage A builds one features x samples matrix per omics type (data/processed/matrices/<omic>.pkl).
Stage B applies the official filters, z-scores each feature and cuts every patient's
profile into 1024-long windows (data/processed/windows/<omic>.npz).
Pass several --raw folders to build a pan-cancer matrix (the paper's setting).
If data/processed/matrices/<omic>.pkl already exists (e.g. synthetic data), stage A is skipped.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import os
import sys
import tarfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catfusion.data.omics import PREPROCESSORS, primary_tumor_columns, zscore_samples  # noqa: E402
from catfusion.models.tabae import sliding_windows  # noqa: E402


def _files(raw, kind):
    manifest = pd.read_csv(os.path.join(raw, f"manifest_{kind}.tsv"), sep="\t")
    for row in manifest.itertuples():
        path = os.path.join(raw, kind, f"{row.file_id}__{row.file_name}")
        if os.path.exists(path):
            yield row.sample_id, path


def read_rna(raw):
    cols = {}
    for sample, path in _files(raw, "rna_seq"):
        df = pd.read_csv(path, sep="\t", comment="#")
        df = df[~df.gene_id.str.startswith("N_")]
        cols[sample] = df.set_index("gene_name")["fpkm_unstranded"]
    return pd.DataFrame(cols)


def read_mirna(raw):
    return pd.DataFrame({s: pd.read_csv(p, sep="\t").set_index("miRNA_ID")["reads_per_million_miRNA_mapped"]
                         for s, p in _files(raw, "mirna_seq")})


def read_methylation(raw):
    return pd.DataFrame({s: pd.read_csv(p, sep="\t", header=None, index_col=0)[1]
                         for s, p in _files(raw, "dnamethylationvariation")})


def read_cnv(raw):
    tars = glob.glob(os.path.join(raw, "*_gistic2.tar.gz"))
    if tars:  # Firehose GISTIC2, as in the official code
        with tarfile.open(tars[0]) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("all_data_by_genes.txt"))
            df = pd.read_csv(tar.extractfile(member), sep="\t")
        return df.drop(columns=["Locus ID", "Cytoband"]).set_index("Gene Symbol")
    # GDC fallback: absolute copy number -> log2(CN/2), roughly GISTIC's scale
    return pd.DataFrame({s: np.log2((pd.read_csv(p, sep="\t").set_index("gene_name")["copy_number"] + 0.1) / 2)
                         for s, p in _files(raw, "copynumbervariation_gdc")})


def read_maf(raw):
    frames = []
    for _, path in _files(raw, "snp"):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            frames.append(pd.read_csv(f, sep="\t", comment="#", low_memory=False,
                                      usecols=["Hugo_Symbol", "Variant_Type", "Tumor_Sample_Barcode"]))
    return pd.concat(frames, ignore_index=True)


READERS = {"rna_seq": read_rna, "mirna_seq": read_mirna, "dnamethylationvariation": read_methylation,
           "copynumbervariation": read_cnv, "snp": read_maf}


def build_matrix(raws, omic):
    parts = [READERS[omic](raw) for raw in raws]
    if omic == "snp":
        return pd.concat(parts, ignore_index=True)  # long MAF table; pivoted in stage B
    return pd.concat(parts, axis=1, join="inner")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", nargs="*", default=[], help="folders written by 00_download_gdc.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--omics", nargs="+", default=["rna_seq"])
    ap.add_argument("--window", type=int, default=1024)
    args = ap.parse_args()
    for sub in ("matrices", "windows"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    for omic in args.omics:
        matrix_path = os.path.join(args.out, "matrices", f"{omic}.pkl")
        if os.path.exists(matrix_path):
            raw = pd.read_pickle(matrix_path)
        else:
            print(f"[{omic}] reading raw files")
            raw = build_matrix(args.raw, omic)
            raw.to_pickle(matrix_path)

        filtered = PREPROCESSORS[omic](raw)  # features x sample barcodes
        filtered = primary_tumor_columns(filtered)  # one primary-tumour sample per patient
        x = zscore_samples(filtered)  # samples x features
        windows = sliding_windows(x, args.window)
        np.savez_compressed(os.path.join(args.out, "windows", f"{omic}.npz"), windows=windows,
                            patient_ids=np.array(filtered.columns), features=np.array(filtered.index.astype(str)))
        print(f"[{omic}] {filtered.shape[1]} patients x {filtered.shape[0]} features -> windows {windows.shape}")


if __name__ == "__main__":
    main()
