"""Step 7 - turn run folders into the tables and figures for the paper.

    python scripts/07_evaluate.py --results results/brca --project TCGA-BRCA

Writes <results>/report.md, report.json and figures/:
  * C-index (aggregated out-of-fold, mean +/- SD over repeats), survival AUC, log-rank p,
    next to the published TCGA-BRCA numbers                                      SQ1
  * paired bootstrap of C-index differences on the same patients                SQ2
  * C-index at 0 / 50 / 100% missing genomics and the "gap recovered"            SQ3
  * Kaplan-Meier curves of predicted low- vs high-risk patients (median split)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from catfusion.metrics import (cindex, gap_recovered, kaplan_meier, logrank_test, paired_bootstrap,  # noqa: E402
                               risk_groups, survival_auc)

# Published TCGA-BRCA C-index (Hu et al. 2025, Tables 1 and 2). Unimodal values are only shown
# as box plots (Fig. 3b) in the paper, so they are left empty here.
PAPER_BRCA = {"catfusion": (0.724, 0.027), "fusion_concat": (0.576, 0.034), "fusion_kronecker": (0.607, 0.029),
              "fusion_lowrank": (0.651, 0.022), "rsf": (0.626, 0.040), "lasso_cox": (0.611, 0.055)}

# Light-mode chart tokens (validated categorical order: blue, orange, aqua, yellow, magenta).
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def load_runs(results_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "predictions.csv"))):
        runs[os.path.basename(os.path.dirname(path))] = pd.read_csv(path)
    return runs


def risk_matrix(preds, missingness=0.0):
    """Patients x repeats matrix of out-of-fold risks, plus time and event per patient."""
    d = preds[preds.missingness == missingness]
    wide = d.pivot_table(index="patient_id", columns="repeat", values="risk")
    info = d.drop_duplicates("patient_id").set_index("patient_id").loc[wide.index]
    return wide.to_numpy(), info.time.to_numpy(), info.event.to_numpy(), wide.index


def mean_cindex(time, event, risks):
    risks = np.atleast_2d(risks.T).T
    return float(np.mean([cindex(time, event, risks[:, j]) for j in range(risks.shape[1])]))


def mean_survival_auc(preds, missingness=0.0):
    """Fold-level survival AUC (training set = the other folds of the same repeat), averaged."""
    d = preds[preds.missingness == missingness]
    values = []
    for (rep, fold), test in d.groupby(["repeat", "fold"]):
        train = d[(d.repeat == rep) & (d.fold != fold)]
        values.append(survival_auc(train.time, train.event, test.time, test.event, test.risk))
    return float(np.nanmean(values)), float(np.nanstd(values))


def describe(name, preds, missingness=0.0):
    risks, time, event, _ = risk_matrix(preds, missingness)
    per_repeat = [cindex(time, event, risks[:, j]) for j in range(risks.shape[1])]
    auc, auc_sd = mean_survival_auc(preds, missingness)
    d = preds[preds.missingness == missingness]
    per_fold = [cindex(g.time, g.event, g.risk) for _, g in d.groupby(["repeat", "fold"])]
    pvals = [logrank_test(time, event, risk_groups(risks[:, j]))[1] for j in range(risks.shape[1])]
    return {"run": name, "missingness": missingness, "n": len(time), "events": int(event.sum()),
            "repeats": risks.shape[1], "cindex": float(np.mean(per_repeat)), "cindex_sd": float(np.std(per_repeat)) if len(per_repeat) > 1 else float("nan"),
            "cindex_fold_mean": float(np.nanmean(per_fold)), "cindex_fold_sd": float(np.nanstd(per_fold)),
            "survival_auc": auc, "survival_auc_sd": auc_sd, "logrank_p_median": float(np.median(pvals))}


def compare(runs, a, b, ma=0.0, mb=0.0, n_boot=1000):
    ra, time, event, ids = risk_matrix(runs[a], ma)
    rb_full, _, _, ids_b = risk_matrix(runs[b], mb)
    rb = pd.DataFrame(rb_full, index=ids_b).loc[ids].to_numpy()  # align patients
    res = paired_bootstrap(time, event, ra, rb, n_boot=n_boot, metric=mean_cindex)
    return {"a": a, "missingness_a": ma, "b": b, "missingness_b": mb, **res}


# ----------------------------------------------------------------------------- figures
def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def km_figure(preds, title, path):
    import matplotlib.pyplot as plt
    risks, time, event, _ = risk_matrix(preds)
    group = risk_groups(risks[:, 0])
    _, p = logrank_test(time, event, group)
    fig, ax = plt.subplots(figsize=(5, 3.6), facecolor=SURFACE)
    style(ax)
    for g, label, color in ((0, "Low risk", SERIES[0]), (1, "High risk", SERIES[1])):
        t, s = kaplan_meier(time[group == g], event[group == g])
        ax.step(t, s, where="post", color=color, linewidth=2, label=f"{label} (n={int((group == g).sum())})")
        ax.text(t[-1], s[-1], f" {label}", color=INK2, fontsize=9, va="center")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Days", color=INK2)
    ax.set_ylabel("Proportion without event", color=INK2)
    ax.set_title(f"{title}\nlog-rank p = {p:.2g} (median split, repeat 0)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def missingness_figure(rows, reference, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.8), facecolor=SURFACE)
    style(ax)
    ends = []
    for i, (name, d) in enumerate(rows.groupby("run", sort=False)):
        d = d.sort_values("missingness")
        color = SERIES[i % len(SERIES)]
        ax.plot(d.missingness * 100, d.cindex, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name)
        ends.append([d.cindex.iloc[-1], name])
    # direct labels at the line ends, nudged apart so they never overlap
    ends.sort()
    lo, hi = rows.cindex.min(), rows.cindex.max()
    gap = 0.045 * (hi - lo if hi > lo else 1)
    for j in range(1, len(ends)):
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + gap)
    for y, name in ends:
        ax.text(103, y, name, color=INK2, fontsize=8, va="center")
    if reference is not None:
        ax.axhline(reference, color=MUTED, linestyle="--", linewidth=1)
        ax.annotate("slide-only (AttMIL-WSI)", (88, reference), xytext=(0, 4), textcoords="offset points",
                    color=MUTED, fontsize=8, ha="center", va="bottom",
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
    ax.set_xlim(-5, 135)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Test patients without genomic data (%)", color=INK2)
    ax.set_ylabel("C-index", color=INK2)
    ax.set_title("Performance as genomics goes missing", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- report
def fmt(m, s):
    if not np.isfinite(m):
        return "–"
    return f"{m:.3f} ± {s:.3f}" if np.isfinite(s) else f"{m:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--project", default="TCGA-BRCA", help="used to pick published reference numbers")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--note", default="", help="text placed at the top of report.md")
    args = ap.parse_args()
    runs = load_runs(args.results)
    if not runs:
        sys.exit(f"no runs with predictions.csv under {args.results}")
    fig_dir = os.path.join(args.results, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    paper = PAPER_BRCA if args.project == "TCGA-BRCA" else {}

    table = [describe(name, p, m) for name, p in runs.items() for m in sorted(p.missingness.unique())]
    table = pd.DataFrame(table)

    comparisons = []
    if "catfusion" in runs:
        for other in ["attmil_wsi", "attmil_genomic", "fusion_concat", "fusion_kronecker", "fusion_lowrank",
                      "cox_clinical", "rsf", "lasso_cox"]:
            if other in runs:
                comparisons.append(compare(runs, "catfusion", other, n_boot=args.n_boot))

    gaps = []
    if {"catfusion", "attmil_wsi"} <= set(runs):
        c_fusion = table.query("run == 'catfusion' and missingness == 0").cindex.iloc[0]
        c_slide = table.query("run == 'attmil_wsi' and missingness == 0").cindex.iloc[0]
        for name in runs:
            if name == "catfusion" or name.startswith("dropout_") or name.startswith("distill_"):
                m = 0.0 if name.startswith("distill_") else 1.0
                row = table[(table.run == name) & (table.missingness == m)]
                if len(row):
                    c = row.cindex.iloc[0]
                    gaps.append({"run": name, "test_missingness": m, "cindex": c, "gap_recovered":
                                 gap_recovered(c, c_slide, c_fusion)})
                    if name != "catfusion":
                        comparisons.append(compare(runs, name, "attmil_wsi", m, 0.0, args.n_boot))

    if "catfusion" in runs:
        km_figure(runs["catfusion"], "CATfusion", os.path.join(fig_dir, "km_catfusion.png"))
    if "attmil_wsi" in runs:
        km_figure(runs["attmil_wsi"], "AttMIL-WSI (slide only)", os.path.join(fig_dir, "km_attmil_wsi.png"))
    miss_rows = table[table.run.apply(lambda r: r == "catfusion" or r.startswith("dropout_"))]
    if miss_rows.missingness.nunique() > 1:
        ref = table.query("run == 'attmil_wsi'").cindex.iloc[0] if "attmil_wsi" in runs else None
        missingness_figure(miss_rows, ref, os.path.join(fig_dir, "missingness.png"))

    # ---- markdown
    lines = ["# Results", ""]
    if args.note:
        lines += [args.note, ""]
    lines += ["## C-index and survival AUC", "",
              "C-index: aggregated out-of-fold predictions, mean ± SD over repeats of 5-fold CV (the paper's "
              "reporting; no SD with a single repeat). Per-fold: mean ± SD of the five test-fold C-indices. "
              "Survival AUC: Uno's cumulative/dynamic AUC, mean ± SD over folds.", "",
              "| Run | Test missingness | n (events) | C-index | Per-fold C-index | Published (BRCA) | Survival AUC | Log-rank p (median) |",
              "|---|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        pub = fmt(*paper[r.run]) if r.run in paper and r.missingness == 0 else "–"
        lines.append(f"| {r.run} | {r.missingness:.0%} | {r.n} ({r.events}) | {fmt(r.cindex, r.cindex_sd)} | {fmt(r.cindex_fold_mean, r.cindex_fold_sd)} | {pub} | "
                     f"{fmt(r.survival_auc, r.survival_auc_sd)} | {r.logrank_p_median:.2g} |")
    if comparisons:
        lines += ["", "## Paired bootstrap of C-index differences (same patients)", "",
                  f"{args.n_boot} bootstrap resamples of patients; the difference is averaged over repeats.", "",
                  "| A | B | ΔC-index (A − B) | 95% CI | p |", "|---|---|---|---|---|"]
        for c in comparisons:
            a = c["a"] + (f" @ {c['missingness_a']:.0%} missing" if c["missingness_a"] else "")
            lines.append(f"| {a} | {c['b']} | {c['diff']:+.3f} | [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | "
                         f"{c['p_two_sided']:.3f} |")
    if gaps:
        lines += ["", "## Gap recovered when genomics is missing (SQ3)", "",
                  "(C_robust − C_slide-only) / (C_fusion − C_slide-only); fusion and slide-only at 0% missing.", "",
                  "| Run | Test missingness | C-index | Gap recovered |", "|---|---|---|---|"]
        for g in gaps:
            lines.append(f"| {g['run']} | {g['test_missingness']:.0%} | {g['cindex']:.3f} | {g['gap_recovered']:.2f} |")
    figs = sorted(os.listdir(fig_dir))
    if figs:
        lines += ["", "## Figures", ""] + [f"![{f}](figures/{f})" for f in figs]

    with open(os.path.join(args.results, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.results, "report.json"), "w") as f:
        json.dump({"table": table.to_dict("records"), "comparisons": comparisons, "gap_recovered": gaps}, f, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
