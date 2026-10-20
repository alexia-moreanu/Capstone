"""Comparison models from the paper, sharing CATfusion's call signature and head.

* AttMIL-WSI      slide-only gated-attention MIL (Ilse et al., 2018)   -> Fig. 3, SQ2 lower bound
* AttMIL-Genomic  the same pooling over genomic tokens                 -> Fig. 3
* Concat / Kronecker / LowRank fusion of the two pooled vectors         -> Table 1

The paper does not publish code for these, so their widths are our choices
(listed in REPLICATION.md). Every model returns a risk score per patient.
"""

from __future__ import annotations

import jax.numpy as jnp
import flax.linen as nn

from catfusion.config import CATfusionConfig
from catfusion.models.catfusion import ScaledSigmoidHead


class GatedAttentionPool(nn.Module):
    """Ilse et al. gated attention: a = softmax(w^T (tanh(Vh) * sigmoid(Uh)))."""
    dim: int = 256
    dropout: float = 0.25

    @nn.compact
    def __call__(self, tokens, deterministic: bool):
        h = nn.Dropout(self.dropout)(nn.relu(nn.Dense(self.dim)(tokens)), deterministic)
        gate = nn.tanh(nn.Dense(self.dim)(h)) * nn.sigmoid(nn.Dense(self.dim)(h))
        weights = nn.softmax(nn.Dense(1)(gate)[..., 0], axis=-1)  # [B, N]
        self.sow("intermediates", "attention", weights)
        return jnp.einsum("bn,bnd->bd", weights, h)


def genomic_tokens(cfg: CATfusionConfig, omics: dict, dim: int):
    """Project every omics type to a shared width and stack the tokens."""
    return jnp.concatenate([nn.Dense(dim, name=f"proj_{n}")(omics[n]) for n in cfg.omics_names], axis=1)


class AttMILWSI(nn.Module):
    cfg: CATfusionConfig
    dim: int = 256

    @nn.compact
    def __call__(self, wsi, omics=None, genomics_present=None, deterministic: bool = True):
        return ScaledSigmoidHead(self.cfg)(GatedAttentionPool(self.dim)(wsi, deterministic), deterministic)


class AttMILGenomic(nn.Module):
    cfg: CATfusionConfig
    dim: int = 256

    @nn.compact
    def __call__(self, wsi, omics, genomics_present=None, deterministic: bool = True):
        tokens = genomic_tokens(self.cfg, omics, self.dim)
        return ScaledSigmoidHead(self.cfg)(GatedAttentionPool(self.dim)(tokens, deterministic), deterministic)


class PooledFusion(nn.Module):
    """Pool each modality with gated attention, then fuse with one of three operators.

    concat     [h_w ; h_g]
    kronecker  vec([h_w;1] (x) [h_g;1])                      (Pathomic Fusion, Chen et al.)
    lowrank    sum_r ([h_w;1] W_w^r) * ([h_g;1] W_g^r)       (Liu et al., 2018, rank 4)
    """
    cfg: CATfusionConfig
    operator: str = "concat"
    dim: int = 256
    small: int = 32  # width before the Kronecker product keeps it at 33*33 features
    rank: int = 4

    @nn.compact
    def __call__(self, wsi, omics, genomics_present=None, deterministic: bool = True):
        h_w = GatedAttentionPool(self.dim, name="pool_wsi")(wsi, deterministic)
        h_g = GatedAttentionPool(self.dim, name="pool_gen")(genomic_tokens(self.cfg, omics, self.dim), deterministic)

        if self.operator == "concat":
            fused = jnp.concatenate([h_w, h_g], axis=-1)
        else:
            one = jnp.ones((h_w.shape[0], 1))
            a = jnp.concatenate([nn.relu(nn.Dense(self.small)(h_w)), one], -1)
            b = jnp.concatenate([nn.relu(nn.Dense(self.small)(h_g)), one], -1)
            if self.operator == "kronecker":
                fused = jnp.einsum("bi,bj->bij", a, b).reshape(a.shape[0], -1)
            elif self.operator == "lowrank":
                wa = self.param("factor_wsi", nn.initializers.xavier_uniform(), (self.rank, a.shape[1], self.dim))
                wb = self.param("factor_gen", nn.initializers.xavier_uniform(), (self.rank, b.shape[1], self.dim))
                fused = (jnp.einsum("bi,rio->bro", a, wa) * jnp.einsum("bj,rjo->bro", b, wb)).sum(axis=1)
            else:
                raise ValueError(f"unknown fusion operator {self.operator!r}")
        fused = nn.Dropout(0.25)(fused, deterministic)
        return ScaledSigmoidHead(self.cfg)(fused, deterministic)
