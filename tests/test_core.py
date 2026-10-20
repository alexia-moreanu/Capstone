"""Fast checks of the pieces the replication depends on. Run: pytest -q"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from catfusion.config import CATfusionConfig, TabAEConfig
from catfusion.losses import cox_loss_exact, cox_loss_official
from catfusion.metrics import cindex, logrank_test, paired_bootstrap, gap_recovered
from catfusion.models import MODELS, build_model
from catfusion.models.tabae import TabAE, sliding_windows

rng = np.random.default_rng(0)


def brute_force_cox(risk, time, event):
    total = 0.0
    for i in range(len(risk)):
        if event[i]:
            total += risk[i] - np.log(np.sum(np.exp(risk[time >= time[i]])))
    return -total / max(event.sum(), 1)


def test_cox_exact_matches_definition():
    risk, time = rng.normal(size=20), rng.integers(1, 10, 20).astype(float)
    event = rng.integers(0, 2, 20).astype(float)
    got = cox_loss_exact(jnp.asarray(risk), jnp.asarray(time), jnp.asarray(event), jnp.ones(20, bool))
    assert np.isclose(got, brute_force_cox(risk, time, event), atol=1e-5)


def test_cox_exact_ignores_padding():
    risk, time, event = rng.normal(size=8), rng.uniform(1, 9, 8), np.ones(8)
    full = cox_loss_exact(risk[:5], time[:5], event[:5], np.ones(5, bool))
    padded = cox_loss_exact(np.r_[risk[:5], 0, 0, 0], np.r_[time[:5], 0, 0, 0], np.r_[event[:5], 0, 0, 0],
                            np.r_[np.ones(5), np.zeros(3)].astype(bool))
    assert np.isclose(full, padded, atol=1e-5)


def test_cox_official_matches_torch_formula():
    # CoxLoss2: R = tril(ones); cost = -(sum((h - log(R @ exp(h))) * c)) / (sum(c) + 1)
    h, c = rng.normal(size=6), np.array([1, 0, 1, 1, 0, 1.0])
    ref = -np.sum((h - np.log(np.tril(np.ones((6, 6))) @ np.exp(h))) * c) / (c.sum() + 1)
    got = cox_loss_official(jnp.asarray(h), jnp.zeros(6), jnp.asarray(c), jnp.ones(6, bool))
    assert np.isclose(got, ref, atol=1e-5)


def test_cindex_matches_sksurv():
    from sksurv.metrics import concordance_index_censored
    t, e, r = rng.uniform(1, 100, 200), rng.integers(0, 2, 200), rng.normal(size=200)
    assert np.isclose(cindex(t, e, r), concordance_index_censored(e.astype(bool), t, r)[0], atol=1e-9)


def test_logrank_matches_sksurv():
    from sksurv.compare import compare_survival
    from sksurv.util import Surv
    t, e, g = rng.uniform(1, 100, 150), rng.integers(0, 2, 150), rng.integers(0, 2, 150)
    chi2, p = logrank_test(t, e, g)
    ref_chi2, ref_p = compare_survival(Surv.from_arrays(e.astype(bool), t), g)
    assert np.isclose(chi2, ref_chi2, rtol=1e-6) and np.isclose(p, ref_p, rtol=1e-6)


def test_bootstrap_and_gap():
    t, e = rng.uniform(1, 100, 100), rng.integers(0, 2, 100)
    good, bad = -t + rng.normal(0, 5, 100), rng.normal(size=100)
    res = paired_bootstrap(t, e, good, bad, n_boot=200)
    assert res["ci_low"] > 0
    assert gap_recovered(0.65, 0.60, 0.70) == pytest.approx(0.5)


def tiny_inputs(cfg, b=3):
    wsi = jnp.asarray(rng.normal(size=(b, cfg.wsi_tokens, cfg.wsi_dim)), jnp.float32)
    omics = {n: jnp.asarray(rng.normal(size=(b, t, cfg.omics_dim)), jnp.float32) for n, t in cfg.omics_tokens.items()}
    return wsi, omics


@pytest.mark.parametrize("name", sorted(MODELS))
def test_models_output_bounded_risk(name):
    cfg = CATfusionConfig.tiny({"rna_seq": 3, "mirna_seq": 2})
    wsi, omics = tiny_inputs(cfg)
    model = build_model(name, cfg)
    params = model.init(jax.random.PRNGKey(0), wsi, omics)["params"]
    risk = model.apply({"params": params}, wsi, omics, None, deterministic=True)
    assert risk.shape == (3,) and np.all(np.abs(risk) <= 4)


def test_paper_config_parameter_count():
    # 768-wide, 12 blocks (2 cross + 10 self), 5 omics types: roughly ViT-Base sized.
    cfg = CATfusionConfig.paper()
    wsi, omics = tiny_inputs(cfg, b=1)
    params = build_model("catfusion", cfg).init(jax.random.PRNGKey(0), wsi, omics)["params"]
    n = sum(x.size for x in jax.tree_util.tree_leaves(params))
    assert 90e6 < n < 120e6, n


def test_missing_token_only_changes_absent_patients():
    cfg = CATfusionConfig.tiny({"rna_seq": 3})
    wsi, omics = tiny_inputs(cfg)
    model = build_model("catfusion", cfg)
    p = model.init(jax.random.PRNGKey(0), wsi, omics)["params"]
    # give the head non-zero weights so risks differ
    p = jax.tree_util.tree_map(lambda x: x + 0.01, p)
    all_present = model.apply({"params": p}, wsi, omics, jnp.array([True, True, True]))
    one_missing = model.apply({"params": p}, wsi, omics, jnp.array([True, False, True]))
    assert np.allclose(np.asarray(all_present)[[0, 2]], np.asarray(one_missing)[[0, 2]], atol=1e-5)
    assert not np.isclose(all_present[1], one_missing[1])


def test_sliding_windows_official_shape():
    x = rng.normal(size=(4, 2500)).astype(np.float32)
    w = sliding_windows(x, 1024)
    assert w.shape == (4, 3, 1024)  # 0-1024, 1024-2048, 1476-2500 (last window end-aligned)
    assert np.allclose(w[:, 2] * x[:, 1476:].std(1, keepdims=True) + x[:, 1476:].mean(1, keepdims=True), x[:, 1476:], atol=1e-4)


def test_tabae_reconstruction_loss_on_masked_tokens():
    cfg = TabAEConfig.tiny()
    x = jnp.asarray(rng.normal(size=(2, 10, 1024)), jnp.float32)
    model = TabAE(cfg, seq_len=10)
    params = model.init(jax.random.PRNGKey(0), x, jax.random.PRNGKey(1))["params"]
    loss, pred, mask = model.apply({"params": params}, x, jax.random.PRNGKey(2))
    assert pred.shape == x.shape and mask.sum() == 2 * 4  # 40% of 10 tokens hidden
    feats = model.apply({"params": params}, x, method=TabAE.features)
    assert feats.shape == (2, 10, 1024)
