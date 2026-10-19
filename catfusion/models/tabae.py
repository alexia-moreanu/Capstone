"""TabAE: masked autoencoder for one omics type (paper Fig. 1b, official mae_model.py).

A patient's feature vector (e.g. 10,240 genes) is cut into 1024-long windows,
giving a short sequence of "tokens". 40% of the tokens are hidden; a 4-block
transformer encodes the rest and a 2-block decoder reconstructs the hidden ones.
After training, the encoder runs with no masking and its output tokens
([L, 1024] per patient) are the genomic input to CATfusion.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from catfusion.config import TabAEConfig

TORCH_LN_EPS = 1e-5  # mae_model.py uses nn.LayerNorm defaults
xavier = nn.initializers.xavier_uniform()


def sinusoidal_positions(length: int, dim: int) -> jnp.ndarray:
    """Fixed sin/cos table, as registered as a buffer in the official model."""
    pos = np.arange(length)[:, None]
    div = np.exp(np.arange(0, dim, 2) * (-np.log(10000.0) / dim))
    table = np.zeros((length, dim), np.float32)
    table[:, 0::2], table[:, 1::2] = np.sin(pos * div), np.cos(pos * div)
    return jnp.asarray(table[None])


class Block(nn.Module):
    dim: int
    heads: int
    mlp_ratio: float

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(TORCH_LN_EPS)(x)
        x = x + nn.MultiHeadDotProductAttention(self.heads, kernel_init=xavier)(y, y)
        y = nn.LayerNorm(TORCH_LN_EPS)(x)
        y = nn.Dense(int(self.dim * self.mlp_ratio), kernel_init=xavier)(y)
        return x + nn.Dense(self.dim, kernel_init=xavier)(nn.gelu(y, approximate=False))


class TabAE(nn.Module):
    cfg: TabAEConfig
    seq_len: int

    def setup(self):
        c = self.cfg
        self.cls_token = self.param("cls_token", nn.initializers.normal(0.02), (1, 1, c.embed_dim))
        self.mask_token = self.param("mask_token", nn.initializers.normal(0.02), (1, 1, c.decoder_dim))
        self.pos = sinusoidal_positions(self.seq_len + 1, c.embed_dim)
        self.dec_pos = sinusoidal_positions(self.seq_len + 1, c.decoder_dim)
        self.blocks = [Block(c.embed_dim, c.heads, c.mlp_ratio) for _ in range(c.depth)]
        self.norm = nn.LayerNorm(TORCH_LN_EPS)
        self.decoder_embed = nn.Dense(c.decoder_dim, kernel_init=xavier)
        self.decoder_blocks = [Block(c.decoder_dim, c.decoder_heads, c.mlp_ratio) for _ in range(c.decoder_depth)]
        self.decoder_norm = nn.LayerNorm(TORCH_LN_EPS)
        self.decoder_pred = nn.Dense(c.embed_dim, kernel_init=xavier)

    def encode(self, x, keep_idx=None):
        """x: [B, L, D]. keep_idx: [B, K] indices of visible tokens (None = keep all, in order)."""
        x = x + self.pos[:, 1:]
        if keep_idx is not None:
            x = jnp.take_along_axis(x, keep_idx[..., None], axis=1)
        cls = jnp.broadcast_to(self.cls_token + self.pos[:, :1], (x.shape[0], 1, x.shape[2]))
        x = jnp.concatenate([cls, x], axis=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def decode(self, latent, restore_idx):
        x = self.decoder_embed(latent)
        b, n_vis = x.shape[0], x.shape[1] - 1
        masks = jnp.broadcast_to(self.mask_token, (b, self.seq_len - n_vis, x.shape[2]))
        x_ = jnp.concatenate([x[:, 1:], masks], axis=1)
        x_ = jnp.take_along_axis(x_, restore_idx[..., None], axis=1)  # unshuffle
        x = jnp.concatenate([x[:, :1], x_], axis=1) + self.dec_pos
        for blk in self.decoder_blocks:
            x = blk(x)
        return self.decoder_pred(self.decoder_norm(x))[:, 1:]

    def __call__(self, x, rng):
        """Returns (loss, prediction, mask) with mask=1 on hidden tokens, as in the official code."""
        b, length, _ = x.shape
        n_keep = int(length * (1 - self.cfg.mask_ratio))
        shuffle = jnp.argsort(jax.random.uniform(rng, (b, length)), axis=1)
        restore = jnp.argsort(shuffle, axis=1)
        latent = self.encode(x, shuffle[:, :n_keep])
        pred = self.decode(latent, restore)
        mask = jnp.take_along_axis(
            jnp.concatenate([jnp.zeros((b, n_keep)), jnp.ones((b, length - n_keep))], axis=1), restore, axis=1)
        per_token = ((pred - x) ** 2).mean(axis=-1)
        loss = (per_token * mask).sum() / mask.sum()  # Eq. 1, masked positions only
        return loss, pred, mask

    def features(self, x):
        """Encoder output without masking, tokens in their original order: [B, L, D]."""
        return self.encode(x)[:, 1:]


def sliding_windows(matrix: np.ndarray, window: int = 1024) -> np.ndarray:
    """Official getData(): cut each row into windows of `window` features, the last
    window aligned to the end (so it overlaps), and standardise each window.

    matrix: [n_samples, n_features] (already z-scored per feature) -> [n_samples, L, window]
    """
    n_feat = matrix.shape[1]
    if n_feat < window:  # pad short profiles (e.g. tiny cohorts) with zeros
        matrix = np.pad(matrix, ((0, 0), (0, window - n_feat)))
        n_feat = window
    starts = [k if k + window <= n_feat else n_feat - window for k in range(0, n_feat, window)]
    windows = np.stack([matrix[:, s:s + window] for s in starts], axis=1).astype(np.float32)
    mean = windows.mean(axis=2, keepdims=True)
    std = windows.std(axis=2, keepdims=True)
    return (windows - mean) / np.where(std > 0, std, 1.0)
