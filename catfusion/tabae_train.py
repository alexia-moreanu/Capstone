"""Self-supervised TabAE training and feature export (official single_omic_training/train.py)."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from catfusion.config import TabAEConfig
from catfusion.models.tabae import TabAE


def cosine_epoch_schedule(cfg: TabAEConfig, steps_per_epoch: int):
    """Official LambdaLR: lr * ((1 + cos(pi * epoch / E)) / 2 * (1 - lrf) + lrf), stepped per epoch."""
    def lr(step):
        epoch = step // steps_per_epoch
        return cfg.lr * ((1 + jnp.cos(epoch * jnp.pi / cfg.epochs)) / 2 * (1 - cfg.lrf) + cfg.lrf)
    return lr


def train_tabae(windows: np.ndarray, cfg: TabAEConfig, val_windows: np.ndarray | None = None, seed: int = 0, log=print):
    """windows: [n_samples, L, 1024] from sliding_windows(). Returns (model, params, history)."""
    model = TabAE(cfg, seq_len=windows.shape[1])
    rng = jax.random.PRNGKey(seed)
    rng, init_key, mask_key = jax.random.split(rng, 3)
    params = model.init(init_key, jnp.asarray(windows[:2]), mask_key)["params"]

    steps = int(np.ceil(len(windows) / cfg.batch_size))
    optimizer = optax.chain(optax.add_decayed_weights(cfg.weight_decay),
                            optax.sgd(cosine_epoch_schedule(cfg, steps), momentum=0.9))
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, x, key):
        loss, grads = jax.value_and_grad(lambda p: model.apply({"params": p}, x, key)[0])(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def eval_loss(params, x, key):
        return model.apply({"params": params}, x, key)[0]

    np_rng = np.random.default_rng(seed)
    history = []
    for epoch in range(cfg.epochs):
        order = np_rng.permutation(len(windows))
        losses = []
        for s in range(0, len(order), cfg.batch_size):
            rng, key = jax.random.split(rng)
            params, opt_state, loss = step(params, opt_state, jnp.asarray(windows[order[s:s + cfg.batch_size]]), key)
            losses.append(float(loss))
        row = {"epoch": epoch, "train_mse": float(np.mean(losses))}
        if val_windows is not None and len(val_windows):
            rng, key = jax.random.split(rng)
            row["val_mse"] = float(eval_loss(params, jnp.asarray(val_windows), key))
        history.append(row)
        log(f"    TabAE epoch {epoch:3d}  train {row['train_mse']:.4f}  val {row.get('val_mse', float('nan')):.4f}")
    return model, params, history


def extract_features(model: TabAE, params, windows: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Encoder output with no masking, tokens kept in their original order -> [n, L, 1024].

    (The official export calls forward_encoder(mask_ratio=0), which still shuffles the
    token order per patient; we keep the order fixed. See REPLICATION.md, finding F5.)
    """
    fn = jax.jit(lambda p, x: model.apply({"params": p}, x, method=TabAE.features))
    return np.concatenate([np.asarray(fn(params, jnp.asarray(windows[s:s + batch_size])))
                           for s in range(0, len(windows), batch_size)])
