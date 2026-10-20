"""Step 2 - self-supervised TabAE per omics type, then export genomic tokens for CATfusion.

    python scripts/02_train_tabae.py --windows data/processed/windows --cohort data/cohort/TCGA-BRCA \
        --omics rna_seq mirna_seq --preset paper --epochs 100

Writes <cohort>/omics/<omic>.h5 (features [n, T, 1024], samplenames), plus the trained
weights and loss curve under <cohort>/tabae/. TabAE never sees survival labels,
so training it on every patient (as the paper does) does not leak outcome information.
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
from flax import serialization

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catfusion.config import TabAEConfig  # noqa: E402
from catfusion.tabae_train import extract_features, train_tabae  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--omics", nargs="+", default=["rna_seq"])
    ap.add_argument("--preset", choices=["paper", "tiny"], default="paper")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=0.2,
                    help="patients held out to monitor reconstruction error (paper: 4:1 split)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = TabAEConfig() if args.preset == "paper" else TabAEConfig.tiny()
    if args.epochs:
        cfg.epochs = args.epochs
    os.makedirs(os.path.join(args.cohort, "omics"), exist_ok=True)
    os.makedirs(os.path.join(args.cohort, "tabae"), exist_ok=True)

    for omic in args.omics:
        data = np.load(os.path.join(args.windows, f"{omic}.npz"), allow_pickle=True)
        windows, ids = data["windows"], data["patient_ids"]
        rng = np.random.default_rng(args.seed)
        is_val = rng.random(len(ids)) < args.val_fraction
        print(f"[{omic}] TabAE on {(~is_val).sum()} train / {is_val.sum()} val patients, {windows.shape[1]} tokens")

        model, params, history = train_tabae(windows[~is_val], cfg, windows[is_val], seed=args.seed)
        feats = extract_features(model, params, windows)

        with h5py.File(os.path.join(args.cohort, "omics", f"{omic}.h5"), "w") as f:
            f.create_dataset("features", data=feats.astype(np.float32), compression="gzip")
            f.create_dataset("samplenames", data=np.array(ids, dtype="S"))
        with open(os.path.join(args.cohort, "tabae", f"{omic}.msgpack"), "wb") as f:
            f.write(serialization.to_bytes(params))
        pd.DataFrame(history).to_csv(os.path.join(args.cohort, "tabae", f"{omic}_history.csv"), index=False)
        print(f"[{omic}] features {feats.shape}")


if __name__ == "__main__":
    main()
