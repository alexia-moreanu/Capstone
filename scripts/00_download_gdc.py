"""Step 0 - download TCGA data from the NIH Genomic Data Commons (run on your own machine).

    python scripts/00_download_gdc.py --project TCGA-BRCA --out data/raw/TCGA-BRCA \
        --kinds clinical rna_seq mirna_seq dnamethylationvariation copynumbervariation snp
    python scripts/00_download_gdc.py --project TCGA-BRCA --out data/raw/TCGA-BRCA --kinds slides --manifest-only

Slides are hundreds of GB, so for slides this script only writes a manifest;
scripts/03_extract_wsi_features.py downloads them one batch at a time, extracts
features and deletes the images (proposal A3/A4).

Sources, matching the paper as closely as the current GDC allows:
  survival endpoint   TCGA Pan-Cancer Clinical Data Resource (Liu et al., Cell 2018), PFI / PFI.time
  rna_seq             GDC "Gene Expression Quantification" (STAR - Counts), fpkm_unstranded
  mirna_seq           GDC "miRNA Expression Quantification", reads_per_million_miRNA_mapped
  methylation         GDC "Methylation Beta Value", Illumina 450K
  copy number         Broad Firehose GISTIC2 all_data_by_genes (as in the official code)
  mutations           GDC "Masked Somatic Mutation" MAF
  slides              GDC "Slide Image", "Diagnostic Slide" (FFPE, DX)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys

import pandas as pd
import requests

GDC = "https://api.gdc.cancer.gov"
# TCGA-CDR-SupplementalTableS1.xlsx from https://gdc.cancer.gov/about-data/publications/pancanatlas
CDR_URL = f"{GDC}/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81"
FIREHOSE_GISTIC = ("https://gdac.broadinstitute.org/runs/analyses__2016_01_28/data/{code}-TP/20160128/"
                   "gdac.broadinstitute.org_{code}-TP.CopyNumber_Gistic2.Level_4.2016012800.0.0.tar.gz")

QUERIES = {
    "rna_seq": {"data_type": "Gene Expression Quantification", "analysis.workflow_type": "STAR - Counts"},
    "mirna_seq": {"data_type": "miRNA Expression Quantification"},
    "dnamethylationvariation": {"data_type": "Methylation Beta Value", "platform": "Illumina Human Methylation 450"},
    "copynumbervariation_gdc": {"data_type": "Gene Level Copy Number"},  # fallback if Firehose is unreachable
    "snp": {"data_type": "Masked Somatic Mutation"},
    "slides": {"data_type": "Slide Image", "experimental_strategy": "Diagnostic Slide"},
}
FIELDS = ["file_id", "file_name", "file_size", "md5sum", "data_type",
          "cases.submitter_id", "cases.samples.submitter_id", "cases.samples.sample_type"]


def query_files(project: str, criteria: dict) -> pd.DataFrame:
    content = [{"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
               {"op": "in", "content": {"field": "access", "value": ["open"]}}]
    content += [{"op": "in", "content": {"field": k, "value": [v]}} for k, v in criteria.items()]
    body = {"filters": {"op": "and", "content": content}, "fields": ",".join(FIELDS),
            "format": "JSON", "size": 20000}
    hits = requests.post(f"{GDC}/files", json=body, timeout=120).json()["data"]["hits"]
    rows = []
    for h in hits:
        case = h["cases"][0]
        samples = case.get("samples") or [{}]
        rows.append({"file_id": h["file_id"], "file_name": h["file_name"], "file_size": h["file_size"],
                     "md5sum": h["md5sum"], "patient_id": case["submitter_id"],
                     "sample_id": samples[0].get("submitter_id", ""),
                     "sample_type": samples[0].get("sample_type", "")})
    return pd.DataFrame(rows)


def download(url: str, path: str, chunk=1 << 20):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    tmp = path + ".part"
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk):
                f.write(block)
    os.replace(tmp, path)
    return path


def download_gdc_files(manifest: pd.DataFrame, out_dir: str, workers: int = 8):
    os.makedirs(out_dir, exist_ok=True)
    jobs = {row.file_id: os.path.join(out_dir, f"{row.file_id}__{row.file_name}") for row in manifest.itertuples()}
    with cf.ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(download, f"{GDC}/data/{fid}", path): fid for fid, path in jobs.items()}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            fut.result()
            if i % 50 == 0 or i == len(futures):
                print(f"    {i}/{len(futures)} files")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="TCGA-BRCA")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", nargs="+", default=["clinical", "rna_seq"])
    ap.add_argument("--manifest-only", action="store_true", help="write manifests, download nothing")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    code = args.project.split("-")[1]

    for kind in args.kinds:
        print(f"[{kind}]")
        if kind == "clinical":
            download(CDR_URL, os.path.join(args.out, "TCGA-CDR-SupplementalTableS1.xlsx"))
            continue
        if kind == "copynumbervariation":
            url = FIREHOSE_GISTIC.format(code=code)
            try:
                download(url, os.path.join(args.out, f"{code}_gistic2.tar.gz"))
                continue
            except Exception as err:  # Firehose is frozen and sometimes offline
                print(f"    Firehose GISTIC2 unavailable ({err}); falling back to GDC gene-level copy number")
                kind = "copynumbervariation_gdc"
        manifest = query_files(args.project, QUERIES[kind])
        manifest.to_csv(os.path.join(args.out, f"manifest_{kind}.tsv"), sep="\t", index=False)
        print(f"    {len(manifest)} files, {manifest.file_size.sum() / 1e9:.1f} GB")
        if not args.manifest_only and kind != "slides":
            download_gdc_files(manifest, os.path.join(args.out, kind), args.workers)

    with open(os.path.join(args.out, "download_info.json"), "w") as f:
        json.dump({"project": args.project, "kinds": args.kinds, "gdc": GDC}, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
