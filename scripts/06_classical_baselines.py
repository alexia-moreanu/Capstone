"""Step 6 - non-deep-learning baselines on the same folds (paper Table 2 and Supplementary Table 7).

    python scripts/06_classical_baselines.py --cohort data/cohort/TCGA-BRCA --out results/brca --repeats 5

  cox_clinical   Cox PH on age, gender and AJCC stage (the paper's clinical baseline)
  rsf            random survival forest on pooled slide + genomic features
  lasso_cox      L1-penalised Cox on the same features, penalty chosen by inner 3-fold CV

"Pooled features" = mean of the 100-mosaic CTransPath features and the mean TabAE token of
each omics type, so every model sees the same information as CATfusion (paper: "identical
input features"). Output has the same format as 05_run_experiments.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catfusion.data.cohort import Cohort, make_folds  # noqa: E402
from catfusion.metrics import cindex  # noqa: E402
from catfusion.train import summarise  # noqa: E402

STAGES = {"I": 1, "II": 2, "III": 3, "IV": 4}


def stage_number(s):
    s = str(s).upper().replace("STAGE", "").strip().rstrip("ABC")
    return STAGES.get(s, np.nan)


def clinical_matrix(table):
    x = pd.DataFrame({"age": pd.to_numeric(table["age"], errors="coerce"),
                      "male": (table["gender"].astype(str).str.upper() == "MALE").astype(float),
                      "stage": table["stage"].map(stage_number)})
    x = x.fillna(x.median())
    return x.loc[:, x.std() > 0].to_numpy(float)


def pooled_features(cohort):
    slide = np.stack([feats.mean(0) for feats, _ in cohort.slides])
    return np.concatenate([slide] + [v.mean(1) for v in cohort.omics.values()], axis=1)


def lasso_cox(x_train, y_train, x_test, seed, n_inner=3):
    """L1-penalised Cox. Each inner fold fits the whole penalty path once; the penalty with
    the best mean inner C-index is refitted on all training patients."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxnetSurvivalAnalysis

    def path_model(alphas=None):
        return CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01, n_alphas=30, alphas=alphas, max_iter=20000)

    scaler = StandardScaler().fit(x_train)
    xs = scaler.transform(x_train)
    alphas = path_model().fit(xs, y_train).alphas_
    scores = np.zeros(len(alphas))
    for tr, va in StratifiedKFold(n_inner, shuffle=True, random_state=seed).split(xs, y_train["event"]):
        fold = path_model(alphas).fit(xs[tr], y_train[tr])
        for i, a in enumerate(fold.alphas_):
            scores[i] += cindex(y_train["time"][va], y_train["event"][va], fold.predict(xs[va], alpha=a))
    best = alphas[int(np.argmax(scores))]
    final = path_model(alphas).fit(xs, y_train)
    return final.predict(scaler.transform(x_test), alpha=best)


def fit_predict(name, x_train, y_train, x_test, seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    if name == "cox_clinical":
        model = make_pipeline(StandardScaler(), CoxPHSurvivalAnalysis(alpha=1e-4))
    elif name == "rsf":
        model = RandomSurvivalForest(n_estimators=300, min_samples_leaf=15, max_features="sqrt",
                                     n_jobs=-1, random_state=seed)
    elif name == "lasso_cox":
        return lasso_cox(x_train, y_train, x_test, seed)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--omics", nargs="+", default=["rna_seq", "mirna_seq", "copynumbervariation",
                                                   "dnamethylationvariation", "snp"])
    ap.add_argument("--projects", nargs="*")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=24)
    ap.add_argument("--only", nargs="*", help="run only these baselines")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    from sksurv.util import Surv

    cohort = Cohort(args.cohort, args.omics, args.projects)
    y = Surv.from_arrays(cohort.event.astype(bool), cohort.time)
    ap_only = set(args.only or ["cox_clinical", "rsf", "lasso_cox"])
    inputs = {"cox_clinical": clinical_matrix(cohort.table), "rsf": pooled_features(cohort),
              "lasso_cox": pooled_features(cohort)}
    inputs = {k: v for k, v in inputs.items() if k in ap_only}

    for name, x in inputs.items():
        rows = []
        for rep in range(args.repeats):
            for k, (tr, te) in enumerate(make_folds(cohort.event, cohort.table["project"], args.folds,
                                                    args.split_seed, rep)):
                risk = fit_predict(name, x[tr], y[tr], x[te], seed=rep * 100 + k)
                rows.append(pd.DataFrame({"repeat": rep, "fold": k, "missingness": 0.0,
                                          "patient_id": cohort.table.patient_id.to_numpy()[te],
                                          "project": cohort.table.project.to_numpy()[te],
                                          "time": cohort.time[te], "event": cohort.event[te],
                                          "genomics_present": True, "risk": risk}))
        preds = pd.concat(rows, ignore_index=True)
        out = os.path.join(args.out, name)
        os.makedirs(out, exist_ok=True)
        preds.to_csv(os.path.join(out, "predictions.csv"), index=False)
        pd.Series(summarise(preds)).to_json(os.path.join(out, "summary.json"), indent=2)
        print(f"{name:13s} C-index {cindex(preds.time, preds.event, preds.risk):.3f}")


if __name__ == "__main__":
    main()
