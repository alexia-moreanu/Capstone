"""Training and cross-validation for every survival model in the repo.

One function, `run_cv`, covers the paper's experiments and the extensions:

    model="catfusion"                              full fusion (paper)
    model="attmil_wsi" / "attmil_genomic"          unimodal baselines (paper Fig. 3)
    model="fusion_concat|kronecker|lowrank"        fusion-operator ablation (paper Table 1)
    model="catfusion", genomics_dropout>0          Extension A: modality dropout
    model="attmil_wsi", distill_weight>0           Extension B: slide-only student of a CATfusion teacher

It writes one row per (repeat, fold, patient, test-time missingness) to predictions.csv.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time as clock
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from catfusion.config import CATfusionConfig, TrainConfig, to_json
from catfusion.data.cohort import Cohort, make_folds
from catfusion.losses import COX_LOSSES, distillation_loss
from catfusion.metrics import cindex
from catfusion.models import build_model


def make_optimizer(cfg: TrainConfig, steps_per_epoch: int):
    """torch Adam(lr, weight_decay) + StepLR(step_size, gamma), written with optax.
    torch's Adam weight decay is L2 added to the gradient, hence add_decayed_weights first."""
    schedule = optax.exponential_decay(cfg.lr, transition_steps=cfg.lr_step_epochs * steps_per_epoch,
                                       decay_rate=cfg.lr_gamma, staircase=True)
    return optax.chain(optax.add_decayed_weights(cfg.weight_decay), optax.scale_by_adam(),
                       optax.scale_by_learning_rate(schedule))


class Trainer:
    """Holds a model, its parameters and compiled train/predict steps."""

    def __init__(self, model_name: str, mcfg: CATfusionConfig, tcfg: TrainConfig, steps_per_epoch: int, seed: int):
        self.model = build_model(model_name, mcfg)
        self.mcfg, self.tcfg = mcfg, tcfg
        self.cox = COX_LOSSES[tcfg.cox_loss]
        self.optimizer = make_optimizer(tcfg, steps_per_epoch)
        self.rng = jax.random.PRNGKey(seed)
        self.params = None

    def init(self, batch):
        self.rng, key = jax.random.split(self.rng)
        self.params = self.model.init({"params": key}, batch.wsi, batch.omics, batch.genomics_present)["params"]
        self.opt_state = self.optimizer.init(self.params)

    @partial(jax.jit, static_argnums=0)
    def _train_step(self, params, opt_state, rng, wsi, omics, present, time, event, valid, teacher):
        def loss_fn(p):
            risk = self.model.apply({"params": p}, wsi, omics, present, deterministic=False, rngs={"dropout": rng})
            loss = self.cox(risk, time, event, valid)
            w = self.tcfg.distill_weight
            if w > 0:
                loss = (1 - w) * loss + w * distillation_loss(risk, teacher, valid)
            return loss
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @partial(jax.jit, static_argnums=0)
    def _predict(self, params, wsi, omics, present):
        return self.model.apply({"params": params}, wsi, omics, present, deterministic=True)

    def train_epoch(self, cohort, idx, np_rng, teacher_risk=None):
        losses = []
        for b in cohort.batches(idx, self.tcfg.batch_size, self.mcfg.wsi_tokens, np_rng, shuffle=True):
            present = b.genomics_present
            if self.tcfg.genomics_dropout > 0:  # Extension A
                present = present & (np_rng.random(len(present)) >= self.tcfg.genomics_dropout)
            teacher = np.zeros(len(b.index), np.float32)
            if teacher_risk is not None:
                teacher = np.where(b.valid, teacher_risk[np.maximum(b.index, 0)], 0).astype(np.float32)
            self.rng, key = jax.random.split(self.rng)
            self.params, self.opt_state, loss = self._train_step(
                self.params, self.opt_state, key, b.wsi, b.omics, present, b.time, b.event, b.valid, teacher)
            losses.append(float(loss))
        return float(np.mean(losses))

    def predict(self, cohort, idx, seed, genomics_present=None, params=None):
        """Risk for each patient in idx (one fixed random draw of 100 patches per patient)."""
        params = self.params if params is None else params
        rng = np.random.default_rng(seed)
        out = []
        for b in cohort.batches(idx, self.tcfg.batch_size, self.mcfg.wsi_tokens, rng, genomics_present):
            out.append(np.asarray(self._predict(params, b.wsi, b.omics, b.genomics_present))[b.valid])
        return np.concatenate(out)

    def held_out_loss(self, cohort, idx, seed):
        risk = self.predict(cohort, idx, seed)
        valid = np.ones(len(idx), bool)
        return float(self.cox(jnp.asarray(risk), cohort.time[idx], cohort.event[idx], valid))


def split_inner(cohort, train_idx, fraction, seed):
    """Hold out part of the training fold for model selection (stratified by event)."""
    from sklearn.model_selection import train_test_split
    return train_test_split(train_idx, test_size=fraction, random_state=seed,
                            stratify=cohort.event[train_idx].astype(int))


def fit(cohort, train_idx, select_idx, model_name, mcfg, tcfg, seed, teacher_risk=None, log=print):
    """Train one model; keep the parameters with the lowest loss on `select_idx`
    (or the last epoch when select_idx is None). Returns (trainer, history)."""
    steps = int(np.ceil(len(train_idx) / tcfg.batch_size))
    trainer = Trainer(model_name, mcfg, tcfg, steps, seed)
    np_rng = np.random.default_rng(seed)
    trainer.init(next(cohort.batches(train_idx[:tcfg.batch_size], tcfg.batch_size, mcfg.wsi_tokens, np_rng)))

    best, best_params, history = np.inf, trainer.params, []
    for epoch in range(tcfg.epochs):
        train_loss = trainer.train_epoch(cohort, train_idx, np_rng, teacher_risk)
        row = {"epoch": epoch, "train_loss": train_loss}
        if select_idx is not None:
            row["select_loss"] = trainer.held_out_loss(cohort, select_idx, seed)
            if row["select_loss"] < best:
                best, best_params, row["best"] = row["select_loss"], trainer.params, True
        history.append(row)
        if epoch % max(1, tcfg.epochs // 5) == 0 or epoch == tcfg.epochs - 1:
            log(f"    epoch {epoch:4d}  train {train_loss:.4f}  select {row.get('select_loss', float('nan')):.4f}")
    if select_idx is not None:
        trainer.params = best_params
    return trainer, history


_TEACHERS: dict = {}


def _teacher_risk(cohort_dir, cohort, fit_idx, select_idx, mcfg, teacher_cfg, seed, log):
    """Train (once per fold, then cached) a CATfusion teacher and return its risk on the training patients."""
    key = (os.path.abspath(cohort_dir), seed, tuple(fit_idx), to_json(mcfg), to_json(teacher_cfg))
    if key not in _TEACHERS:
        log("    training CATfusion teacher")
        teacher, _ = fit(cohort, fit_idx, select_idx, "catfusion", mcfg, teacher_cfg, seed, log=log)
        risk = np.zeros(len(cohort), np.float32)
        risk[fit_idx] = teacher.predict(cohort, fit_idx, seed)
        _TEACHERS[key] = risk
    return _TEACHERS[key]


def run_cv(cohort_dir, out_dir, model_name, mcfg: CATfusionConfig, tcfg: TrainConfig, projects=None,
           test_missingness=(0.0,), teacher_cfg: TrainConfig | None = None, log=print):
    """K-fold cross-validation, repeated `tcfg.repeats` times. Writes predictions.csv,
    history.csv, summary.json and the configs to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        f.write(json.dumps({"model": model_name, "model_cfg": json.loads(to_json(mcfg)),
                            "train_cfg": json.loads(to_json(tcfg)),
                            "teacher_cfg": json.loads(to_json(teacher_cfg)) if teacher_cfg else None}, indent=2))
    cohort = Cohort(cohort_dir, mcfg.omics_names, projects)
    log(f"[{model_name}] {len(cohort)} complete-case patients, event rate {cohort.event.mean():.2f}")

    rows, histories = [], []
    for rep in range(tcfg.repeats):
        folds = make_folds(cohort.event, cohort.table["project"], tcfg.folds, tcfg.split_seed, rep)
        for k, (train_idx, test_idx) in enumerate(folds):
            seed = tcfg.seed + 1000 * rep + k
            t0 = clock.time()
            if tcfg.selection == "official":
                fit_idx, select_idx = train_idx, test_idx  # the released code selects on the test fold
            elif tcfg.selection == "inner_val":
                fit_idx, select_idx = split_inner(cohort, train_idx, tcfg.inner_val_fraction, seed)
            else:
                fit_idx, select_idx = train_idx, None

            teacher_risk = None
            if tcfg.distill_weight > 0:  # Extension B: a fusion teacher trained on the same patients
                teacher_risk = _teacher_risk(cohort_dir, cohort, fit_idx, select_idx, mcfg,
                                             teacher_cfg or dataclasses.replace(tcfg, distill_weight=0.0), seed, log)

            log(f"  repeat {rep} fold {k}: train {len(fit_idx)}  test {len(test_idx)}")
            trainer, history = fit(cohort, fit_idx, select_idx, model_name, mcfg, tcfg, seed, teacher_risk, log)
            histories += [{**h, "repeat": rep, "fold": k} for h in history]

            for miss in test_missingness:
                present = np.ones(len(cohort), bool)
                drop = np.random.default_rng(seed).permutation(test_idx)[: int(round(miss * len(test_idx)))]
                present[drop] = False
                risk = trainer.predict(cohort, test_idx, seed + 7, genomics_present=present)
                rows.append(pd.DataFrame({
                    "repeat": rep, "fold": k, "missingness": miss,
                    "patient_id": cohort.table["patient_id"].to_numpy()[test_idx],
                    "project": cohort.table["project"].to_numpy()[test_idx],
                    "time": cohort.time[test_idx], "event": cohort.event[test_idx],
                    "genomics_present": present[test_idx], "risk": risk}))
            last = rows[-len(test_missingness)]
            log(f"  repeat {rep} fold {k}: test C-index {cindex(last.time, last.event, last.risk):.3f}"
                f"  ({clock.time() - t0:.0f}s)")

    preds = pd.concat(rows, ignore_index=True)
    preds.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    pd.DataFrame(histories).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    summary = summarise(preds)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return preds, summary


def summarise(preds: pd.DataFrame) -> dict:
    """Paper-style summary: C-index of the aggregated out-of-fold predictions per repeat,
    then mean and SD over repeats; plus the mean and SD over individual folds."""
    out = {}
    for miss, d in preds.groupby("missingness"):
        per_repeat = [cindex(g.time, g.event, g.risk) for _, g in d.groupby("repeat")]
        per_fold = [cindex(g.time, g.event, g.risk) for _, g in d.groupby(["repeat", "fold"])]
        out[f"missingness={miss:g}"] = {
            "cindex_aggregated_mean": float(np.mean(per_repeat)),
            "cindex_aggregated_sd": float(np.std(per_repeat)),
            "cindex_fold_mean": float(np.nanmean(per_fold)),
            "cindex_fold_sd": float(np.nanstd(per_fold)),
            "n_patients": int(d.patient_id.nunique()),
        }
    return out
