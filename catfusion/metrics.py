"""Evaluation: C-index, survival AUC, Kaplan-Meier / log-rank, paired bootstrap, gap recovered.

Convention everywhere: higher risk = expected to fail sooner.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def cindex(time, event, risk) -> float:
    """Harrell's concordance index (= lifelines.concordance_index(time, -risk, event),
    which is what the official code reports).

    A pair (i, j) is comparable when i had the event and time_i < time_j. It is
    concordant when risk_i > risk_j; tied risks count one half.
    """
    time, event, risk = map(np.asarray, (time, event, risk))
    comparable = (event[:, None] == 1) & (time[:, None] < time[None, :])
    diff = risk[:, None] - risk[None, :]
    score = (diff > 0) + 0.5 * (diff == 0)
    n = comparable.sum()
    return float((score * comparable).sum() / n) if n else float("nan")


def survival_auc(train_time, train_event, test_time, test_event, risk, quantiles=(0.1, 0.9), n_times=10):
    """Mean time-dependent cumulative/dynamic AUC (Uno, IPCW), the paper's "survival AUC".

    Evaluated on an even grid between the given quantiles of test event times,
    clipped to the training follow-up so the censoring weights are defined.
    """
    from sksurv.metrics import cumulative_dynamic_auc
    from sksurv.util import Surv

    y_train = Surv.from_arrays(np.asarray(train_event, bool), train_time)
    y_test = Surv.from_arrays(np.asarray(test_event, bool), test_time)
    event_times = np.asarray(test_time)[np.asarray(test_event, bool)]
    if len(event_times) < 2:
        return float("nan")
    lo, hi = np.quantile(event_times, quantiles)
    hi = min(hi, np.max(train_time) - 1e-6, np.max(test_time) - 1e-6)
    if hi <= lo:
        return float("nan")
    times = np.linspace(lo, hi, n_times)
    try:
        _, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, times)
    except ValueError:
        return float("nan")
    return float(mean_auc)


def logrank_test(time, event, group) -> tuple[float, float]:
    """Two-group log-rank test. Returns (chi-square statistic, p-value)."""
    time, event, group = map(np.asarray, (time, event, group))
    observed_minus_expected, variance = 0.0, 0.0
    for t in np.unique(time[event == 1]):
        at_risk = time >= t
        n, n1 = at_risk.sum(), (at_risk & (group == 1)).sum()
        d = ((time == t) & (event == 1)).sum()
        d1 = ((time == t) & (event == 1) & (group == 1)).sum()
        observed_minus_expected += d1 - d * n1 / n
        if n > 1:
            variance += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    if variance == 0:
        return 0.0, 1.0
    chi2 = observed_minus_expected ** 2 / variance
    return float(chi2), float(stats.chi2.sf(chi2, df=1))


def risk_groups(risk, cutoff=None):
    """1 = high risk (above the median, as in the paper), 0 = low risk."""
    risk = np.asarray(risk)
    cutoff = np.median(risk) if cutoff is None else cutoff
    return (risk > cutoff).astype(int)


def kaplan_meier(time, event):
    """Kaplan-Meier survival curve: returns (times, survival probability)."""
    time, event = np.asarray(time), np.asarray(event)
    times = np.unique(time)
    surv, s = [], 1.0
    for t in times:
        n = (time >= t).sum()
        d = ((time == t) & (event == 1)).sum()
        s *= 1 - d / n
        surv.append(s)
    return np.concatenate([[0], times]), np.concatenate([[1.0], surv])


def paired_bootstrap(time, event, risk_a, risk_b, n_boot=1000, seed=0, metric=cindex):
    """Bootstrap the difference metric(a) - metric(b) on the *same* resampled patients.

    Returns dict(diff, ci_low, ci_high, p_two_sided). Proposal B5.6 / SQ2.
    """
    rng = np.random.default_rng(seed)
    time, event, risk_a, risk_b = map(np.asarray, (time, event, risk_a, risk_b))
    observed = metric(time, event, risk_a) - metric(time, event, risk_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(time), len(time))
        d = metric(time[idx], event[idx], risk_a[idx]) - metric(time[idx], event[idx], risk_b[idx])
        if np.isfinite(d):
            diffs.append(d)
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(observed), "ci_low": float(np.quantile(diffs, 0.025)),
            "ci_high": float(np.quantile(diffs, 0.975)), "p_two_sided": float(min(p, 1.0))}


def gap_recovered(c_robust, c_slide_only, c_fusion) -> float:
    """(C_robust - C_slide) / (C_fusion - C_slide). 0 = no help from genomics in training,
    1 = slide-only prediction as good as full fusion. Proposal B5.6."""
    gap = c_fusion - c_slide_only
    return float((c_robust - c_slide_only) / gap) if gap != 0 else float("nan")
