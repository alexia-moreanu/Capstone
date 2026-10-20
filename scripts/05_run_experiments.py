"""Step 5 - cross-validated experiments. Each suite maps to a sub-question in the proposal.

    python scripts/05_run_experiments.py --cohort data/cohort/TCGA-BRCA --out results/brca \
        --suite replication --preset paper --epochs 100 --repeats 5

Suites
  replication     SQ1/SQ2  CATfusion, AttMIL-WSI, AttMIL-Genomic with the released training recipe
  fusion          Table 1  concatenation, Kronecker and low-rank fusion
  protocol        F2/F3    effect of the released code's model selection and Cox loss
  extension       SQ3/SQ4  modality dropout and distillation, tested at 0/50/100% missing genomics
  all             everything above

Each run writes <out>/<run>/{predictions.csv, history.csv, summary.json, config.json}.
Analyse them with scripts/07_evaluate.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catfusion.config import CATfusionConfig, TrainConfig  # noqa: E402
from catfusion.data.cohort import Cohort  # noqa: E402
from catfusion.train import run_cv  # noqa: E402

MISSINGNESS = (0.0, 0.5, 1.0)


def suite_runs(suite, base: TrainConfig, dropout_rates, distill_weights):
    """List of (run_name, model, train_cfg, test_missingness)."""
    r = dataclasses.replace
    runs = []
    if suite in ("replication", "all"):
        runs += [("catfusion", "catfusion", base, MISSINGNESS if suite == "all" else (0.0,)),
                 ("attmil_wsi", "attmil_wsi", base, (0.0,)),
                 ("attmil_genomic", "attmil_genomic", base, (0.0,))]
    if suite in ("fusion", "all"):
        runs += [(f"fusion_{op}", f"fusion_{op}", base, (0.0,)) for op in ("concat", "kronecker", "lowrank")]
    if suite in ("protocol", "all"):
        # compare against the "catfusion" run, which uses the base recipe (released code by default)
        runs += [("protocol_inner_val_official_cox", "catfusion", r(base, selection="inner_val", cox_loss="official"), (0.0,)),
                 ("protocol_inner_val_exact_cox", "catfusion", r(base, selection="inner_val", cox_loss="exact"), (0.0,))]
    if suite in ("extension", "all"):
        if suite == "extension":
            runs += [("catfusion", "catfusion", base, MISSINGNESS), ("attmil_wsi", "attmil_wsi", base, (0.0,))]
        runs += [(f"dropout_{p:g}", "catfusion", r(base, genomics_dropout=p), MISSINGNESS) for p in dropout_rates]
        runs += [(f"distill_{w:g}", "attmil_wsi", r(base, distill_weight=w), (0.0,)) for w in distill_weights]
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="replication", choices=["replication", "fusion", "protocol", "extension", "all"])
    ap.add_argument("--preset", default="paper", choices=["paper", "tiny"])
    ap.add_argument("--omics", nargs="+", default=["rna_seq", "mirna_seq", "copynumbervariation",
                                                   "dnamethylationvariation", "snp"])
    ap.add_argument("--projects", nargs="*", help="restrict to these TCGA projects")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None, help="paper: 1e-5")
    ap.add_argument("--weight-decay", type=float, default=None, help="paper: 0.7 (coupled L2 inside Adam)")
    ap.add_argument("--selection", default="official", choices=["official", "inner_val", "last"],
                    help="model selection for the base recipe (official = released code)")
    ap.add_argument("--cox-loss", default="official", choices=["official", "exact"])
    ap.add_argument("--dropout-rates", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--distill-weights", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--only", nargs="*", help="run only these run names")
    args = ap.parse_args()

    if args.preset == "paper":
        mcfg = CATfusionConfig.paper(args.omics)
    else:  # token counts come from the data
        probe = Cohort(args.cohort, args.omics)
        mcfg = CATfusionConfig.tiny({k: v.shape[1] for k, v in probe.omics.items()})
    base = TrainConfig(epochs=args.epochs, repeats=args.repeats, batch_size=args.batch_size,
                       selection=args.selection, cox_loss=args.cox_loss)
    if args.lr is not None:
        base = dataclasses.replace(base, lr=args.lr)
    if args.weight_decay is not None:
        base = dataclasses.replace(base, weight_decay=args.weight_decay)

    for name, model, tcfg, missing in suite_runs(args.suite, base, args.dropout_rates, args.distill_weights):
        if args.only and name not in args.only:
            continue
        out = os.path.join(args.out, name)
        if os.path.exists(os.path.join(out, "summary.json")):
            print(f"== {name}: already done, skipping")
            continue
        print(f"== {name}")
        run_cv(args.cohort, out, model, mcfg, tcfg, args.projects, missing)


if __name__ == "__main__":
    main()
