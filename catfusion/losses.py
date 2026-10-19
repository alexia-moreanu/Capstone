"""Cox partial-likelihood losses (paper Eq. 4-5) and the distillation loss.

All losses take a `valid` mask so batches can be padded to a fixed size (one
compiled function per model instead of one per batch length).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import logsumexp


def cox_loss_exact(risk, time, event, valid):
    """Negative Cox log partial likelihood, averaged over observed events.

    Risk set of patient i = everyone still at risk at time_i (time_j >= time_i),
    which handles tied times the Breslow way.
    """
    at_risk = (time[None, :] >= time[:, None]) & valid[None, :]
    log_risk_set = logsumexp(jnp.where(at_risk, risk[None, :], -jnp.inf), axis=1)
    observed = event * valid
    return -jnp.sum((risk - log_risk_set) * observed) / jnp.maximum(observed.sum(), 1.0)


def cox_loss_official(risk, time, event, valid):
    """The released CoxLoss2, kept for replication.

    It builds the risk set with a lower-triangular matrix of ones, i.e. patient i's
    risk set is "everyone earlier in the batch", regardless of survival time, and
    divides by (number of events + 1). Batches are not sorted by time first, so
    this is not the textbook Cox likelihood. See REPLICATION.md, finding F3.
    """
    n = risk.shape[0]
    tril = jnp.tril(jnp.ones((n, n))) * valid[None, :]
    risk_set_sum = tril @ jnp.exp(risk)
    diff = risk - jnp.log(jnp.where(valid, risk_set_sum, 1.0))
    observed = event * valid
    return -jnp.sum(diff * observed) / (observed.sum() + 1.0)


COX_LOSSES = {"exact": cox_loss_exact, "official": cox_loss_official}


def distillation_loss(student_risk, teacher_risk, valid):
    """Mean squared error between student and teacher risk scores (Hinton-style, on the risk scale)."""
    return jnp.sum(((student_risk - teacher_risk) ** 2) * valid) / jnp.maximum(valid.sum(), 1.0)
