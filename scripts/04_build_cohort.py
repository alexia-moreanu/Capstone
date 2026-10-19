"""Step 4 - join survival labels, slide features and genomic features into cohort.csv.

    python scripts/04_build_cohort.py --cdr data/raw/TCGA-BRCA/TCGA-CDR-SupplementalTableS1.xlsx \
        --cohort data/cohort/TCGA-BRCA --projects TCGA-BRCA --endpoint PFI

The paper's Kaplan-Meier plots use the progression-free interval (PFI) from the TCGA
Clinical Data Resource, and the released code reads PFI / PFI.time, so PFI is the
default. --endpoint OS switches to overall survival.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdr", required=True, help="TCGA-CDR-SupplementalTableS1.xlsx (or a .csv with the same columns)")
    ap.add_argument("--cohort", required=True, help="folder containing slides/ and omics/")
    ap.add_argument("--projects", nargs="+", default=["TCGA-BRCA"])
    ap.add_argument("--endpoint", default="PFI", choices=["PFI", "OS", "DSS", "DFI"])
    args = ap.parse_args()

    if args.cdr.endswith(".xlsx"):
        cdr = pd.read_excel(args.cdr, sheet_name="TCGA-CDR")
    else:
        cdr = pd.read_csv(args.cdr)
    cdr["project"] = "TCGA-" + cdr["type"]
    cdr = cdr[cdr["project"].isin(args.projects)]
    cdr = cdr.rename(columns={"bcr_patient_barcode": "patient_id", args.endpoint: "event",
                              f"{args.endpoint}.time": "time", "age_at_initial_pathologic_diagnosis": "age",
                              "ajcc_pathologic_tumor_stage": "stage"})
    cdr = cdr.dropna(subset=["event", "time"])
    cdr = cdr[cdr["time"] > 0]

    slides = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.cohort, "slides", "*.h5"))):
        slides[os.path.basename(path)[:12]].append(os.path.relpath(path, args.cohort))
    cdr["slide_files"] = cdr["patient_id"].map(lambda p: ";".join(slides[p]) if p in slides else None)

    keep = ["patient_id", "project", "time", "event", "slide_files", "age", "gender", "stage"]
    table = cdr[keep].reset_index(drop=True)
    table.to_csv(os.path.join(args.cohort, "cohort.csv"), index=False)
    print(f"{len(table)} patients with {args.endpoint}; {table.slide_files.notna().sum()} with slide features; "
          f"event rate {table.event.mean():.2f}")


if __name__ == "__main__":
    main()
