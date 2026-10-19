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


def fit_predict(name, x_train, y_train, x_test, seed):
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis

    if name == "cox_clinical":
        model = make_pipeline(StandardScaler(), CoxPHSurvivalAnalysis(alpha=1e-4))
    elif name == "rsf":
        model = RandomSurvivalForest(n_estimators=300, min_samples_leaf=15, max_features="sqrt",
                                     n_jobs=-1, random_state=seed)
    elif name == "lasso_cox":
        path = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01, n_alphas=30).fit(
            StandardScaler().fit_transform(x_train), y_train).alphas_
        model = GridSearchCV(make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=1.0, max_iter=10000)),
                             {"coxnetsurvivalanalysis__alphas": [[a] for a in path]},
                             cv=StratifiedKFold(3, shuffle=True, random_state=seed).split(x_train, y_train["event"]),
                             error_score=0.5)
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
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    from sksurv.util import Surv

    cohort = Cohort(args.cohort, args.omics, args.projects)
    y = Surv.from_arrays(cohort.event.astype(bool), cohort.time)
    inputs = {"cox_clinical": clinical_matrix(cohort.table), "rsf": pooled_features(cohort),
              "lasso_cox": pooled_features(cohort)}

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
