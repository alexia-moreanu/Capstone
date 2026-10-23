"""
Immune-signature survival baseline for TCGA-LUAD.

Reusable pieces for the capstone (CATfusion in JAX):
    load_cohort()        -> one row per patient: OS time/event, age, stage, gene expression
    score_signatures()   -> per-patient immune signature scores (z-scored)
    fit_cox()            -> adjusted hazard ratios
    cv_cindex()          -> cross-validated C-index for any set of covariates
    harrell_c()          -> plain numpy C-index (same metric CATfusion is evaluated with)

Every function takes and returns plain pandas objects, so the same scores can later
be correlated with CATfusion risk predictions on the same patients.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from statsmodels.duration.hazard_regression import PHReg

# ---------------------------------------------------------------------------
# Immune signatures (each tied to a course session)
# ---------------------------------------------------------------------------
SIGNATURES: dict[str, list[str]] = {
    "cDC1":            ["CLEC9A", "XCR1", "BATF3"],                    # 3.1, 12.1
    "Cytolytic":       ["GZMA", "PRF1"],                               # 6.1 (Rooney et al., 2015)
    "TLS":             ["CXCL13", "CCL19", "CCL21", "LAMP3", "MS4A1"],  # 12.1
    "MHC-I":           ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1"],      # 4.1, 10.1
    "Treg":            ["FOXP3", "IL2RA", "CTLA4"],                    # 10.1
}

CLINICAL = ["age", "stage"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _open(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def load_expression(tpm_path, probemap_path, genes) -> pd.DataFrame:
    """Read only the rows we need from the 60k-gene GDC matrix (fast, low memory).

    Returns samples x genes, values in log2(TPM + 1).
    """
    probemap = pd.read_csv(probemap_path, sep="\t", usecols=[0, 1])
    probemap.columns = ["ensembl_id", "gene"]
    wanted = probemap[probemap["gene"].isin(genes)]
    id_to_gene = dict(zip(wanted["ensembl_id"], wanted["gene"]))

    rows = {}
    with _open(Path(tpm_path)) as f:
        samples = f.readline().rstrip("\n").split("\t")[1:]
        for line in f:
            ensembl_id, rest = line.split("\t", 1)
            if ensembl_id in id_to_gene:
                rows[id_to_gene[ensembl_id]] = np.array(rest.split("\t"), dtype=float)

    missing = sorted(set(genes) - set(rows))
    if missing:
        print(f"Warning: genes not found in expression matrix: {missing}")
    return pd.DataFrame(rows, index=samples)


def _find_column(df: pd.DataFrame, *patterns: str) -> str:
    for pattern in patterns:
        hits = [c for c in df.columns if pattern in c.lower()]
        if hits:
            return hits[0]
    raise KeyError(f"No column matching {patterns}. Columns: {list(df.columns)[:30]}...")


def _stage_to_number(stage) -> float:
    """'Stage IIB' -> 2. Ordinal coding keeps the Cox model small."""
    match = re.search(r"stage\s+(iv|iii|ii|i)", str(stage).lower())
    return {"i": 1, "ii": 2, "iii": 3, "iv": 4}[match.group(1)] if match else np.nan


def overall_survival_from_clinical(pheno: pd.DataFrame) -> pd.DataFrame:
    """Standard OS definition (as in the TCGA Pan-Cancer Clinical Data Resource):
    dead  -> time = days_to_death, event = 1
    alive -> time = days_to_last_follow_up, event = 0
    Takes the longest follow-up across a patient's rows (one row per sample).
    """
    num = lambda col: pd.to_numeric(pheno[_find_column(pheno, col)], errors="coerce")
    dead = pheno[_find_column(pheno, "vital_status")].eq("Dead")
    time = np.where(dead, num("days_to_death"), num("days_to_last_follow_up"))
    surv = pd.DataFrame({"patient": pheno["patient"], "time": time, "event": dead.astype(int)})
    return surv.dropna().groupby("patient").agg(time=("time", "max"), event=("event", "max"))


def load_cohort(expr: pd.DataFrame, survival_path, phenotype_path) -> pd.DataFrame:
    """One row per patient: primary tumour only (sample type 01), first vial if several.

    `expr` is the samples x genes table from load_expression().
    """
    expr = expr.copy()

    # Primary solid tumour = sample type code "01" (TCGA-XX-XXXX-01A)
    expr = expr[expr.index.str[13:15] == "01"].sort_index()
    expr["patient"] = expr.index.str[:12]
    expr = expr.drop_duplicates("patient").set_index("patient")

    pheno = pd.read_csv(phenotype_path, sep="\t", low_memory=False)
    pheno["patient"] = pheno.iloc[:, 0].astype(str).str[:12]

    if survival_path is not None:
        surv = pd.read_csv(survival_path, sep="\t")
        surv["patient"] = surv[_find_column(surv, "_patient", "sample")].str[:12]
        surv = (surv.rename(columns={"OS.time": "time", "OS": "event"})
                    .drop_duplicates("patient").set_index("patient")[["time", "event"]])
    else:
        surv = overall_survival_from_clinical(pheno)
    age_col = _find_column(pheno, "age_at_index", "age_at_initial", "age_at_diagnosis")
    stage_col = _find_column(pheno, "ajcc_pathologic_stage", "pathologic_stage", "tumor_stage")
    pheno = pheno.drop_duplicates("patient").set_index("patient")
    clinical = pd.DataFrame({
        "age": pd.to_numeric(pheno[age_col], errors="coerce"),
        "stage": pheno[stage_col].map(_stage_to_number),
    })
    # age_at_diagnosis is stored in days in some GDC exports
    if clinical["age"].median() > 200:
        clinical["age"] = clinical["age"] / 365.25

    cohort = surv.join(clinical, how="inner").join(expr, how="inner")
    cohort = cohort[cohort["time"] > 0].dropna(subset=["time", "event", "age", "stage"])
    cohort["event"] = cohort["event"].astype(int)
    return cohort


# ---------------------------------------------------------------------------
# Signature scores
# ---------------------------------------------------------------------------
def zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=0)


def score_signatures(cohort: pd.DataFrame, signatures=SIGNATURES) -> pd.DataFrame:
    """Mean of per-gene z-scores, then re-standardised so a Cox HR reads 'per 1 SD'.

    Z-scoring each gene first stops one highly expressed gene (e.g. B2M)
    from dominating its signature.
    """
    gene_z = zscore(cohort[[g for gs in signatures.values() for g in gs if g in cohort]])
    scores = pd.DataFrame({
        name: gene_z[[g for g in genes if g in gene_z]].mean(axis=1)
        for name, genes in signatures.items()
    })
    return zscore(scores)


# ---------------------------------------------------------------------------
# Survival models
# ---------------------------------------------------------------------------
def fit_cox(data: pd.DataFrame, covariates: list[str]):
    """Cox proportional hazards. Returns a tidy table of HR, 95% CI and p-value."""
    model = PHReg(data["time"], data[covariates], status=data["event"], ties="efron")
    result = model.fit()
    ci = np.exp(result.conf_int())
    return pd.DataFrame({
        "HR": np.exp(result.params),
        "CI_low": ci[:, 0],
        "CI_high": ci[:, 1],
        "p": result.pvalues,
    }, index=covariates)


def harrell_c(time, event, risk) -> float:
    """Harrell's concordance: of comparable pairs, how often the higher-risk patient dies first."""
    time, event, risk = map(np.asarray, (time, event, risk))
    earlier = time[:, None] < time[None, :]           # i died before j was last seen
    comparable = earlier & (event[:, None] == 1)
    concordant = (risk[:, None] > risk[None, :]) & comparable
    ties = (risk[:, None] == risk[None, :]) & comparable
    return (concordant.sum() + 0.5 * ties.sum()) / comparable.sum()


def cv_cindex(data: pd.DataFrame, covariates: list[str], n_splits=5, n_repeats=10,
              splits=None) -> np.ndarray:
    """Out-of-fold C-index. Pass `splits` (list of (train_idx, test_idx)) to reuse capstone folds."""
    if splits is None:
        splits = [
            fold
            for seed in range(n_repeats)
            for fold in StratifiedKFold(n_splits, shuffle=True, random_state=seed)
                        .split(data, data["event"])
        ]
    cindices = []
    for train_idx, test_idx in splits:
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        beta = PHReg(train["time"], train[covariates], status=train["event"],
                     ties="efron").fit().params
        risk = test[covariates].to_numpy() @ np.asarray(beta)
        cindices.append(harrell_c(test["time"], test["event"], risk))
    return np.array(cindices)
