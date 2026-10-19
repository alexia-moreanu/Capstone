"""CATfusion in JAX/Flax: a line-by-line port of the authors' PyTorch model.

Data flow for one patient (paper Fig. 1a, official models/*.py):

    100 slide-patch features [100, 768] --Linear--> + cls token + pos-emb --> image tokens
    each omics type [T_k, 1024]       --Linear--> + its own pos-emb      --> genomic tokens
    2 x bidirectional cross-attention blocks   (image <-> genomic)
    concat(image, genomic) -> 10 x self-attention blocks -> LayerNorm
    risk = MLP([cls ; mean(other tokens)]) -> sigmoid * 8 - 4

Extension (proposal B5.2 A): `genomics_present` lets a patient's genomic tokens be
replaced by a learned "missing" embedding. With every patient present the model
is identical to the original.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn

from catfusion.config import CATfusionConfig

LN_EPS = 1e-6  # official: LayerNorm(hidden, eps=1e-6)
xavier = nn.initializers.xavier_uniform()
tiny_normal = nn.initializers.normal(1e-6)


class Mlp(nn.Module):
    cfg: CATfusionConfig

    @nn.compact
    def __call__(self, x, deterministic: bool):
        x = nn.Dense(self.cfg.mlp_dim, kernel_init=xavier, bias_init=tiny_normal)(x)
        x = nn.gelu(x, approximate=False)  # torch.nn.functional.gelu is exact
        x = nn.Dropout(self.cfg.dropout)(x, deterministic)
        x = nn.Dense(self.cfg.hidden, kernel_init=xavier, bias_init=tiny_normal)(x)
        return nn.Dropout(self.cfg.dropout)(x, deterministic)


def split_heads(x, heads):
    b, n, d = x.shape
    return x.reshape(b, n, heads, d // heads).transpose(0, 2, 1, 3)  # [B, H, N, Dh]


def merge_heads(x):
    b, h, n, dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, n, h * dh)


class Attend(nn.Module):
    """softmax(Q K^T / sqrt(d)) V with dropout on the attention probabilities."""
    rate: float

    @nn.compact
    def __call__(self, q, k, v, deterministic: bool):
        scores = q @ k.swapaxes(-1, -2) / jnp.sqrt(q.shape[-1])
        probs = jax.nn.softmax(scores, axis=-1)
        dropped = nn.Dropout(self.rate)(probs, deterministic)
        return dropped @ v, probs


class SelfAttention(nn.Module):
    cfg: CATfusionConfig

    @nn.compact
    def __call__(self, x, deterministic: bool):
        c = self.cfg
        q, k, v = (split_heads(nn.Dense(c.hidden, name=n)(x), c.heads) for n in ("query", "key", "value"))
        ctx, probs = Attend(c.attention_dropout)(q, k, v, deterministic)
        self.sow("intermediates", "attention", probs)
        out = nn.Dense(c.hidden, name="out")(merge_heads(ctx))
        return nn.Dropout(c.attention_dropout)(out, deterministic)


class BidirectionalAttention(nn.Module):
    """Paper Eq. 2-3. Each modality's queries attend to its own keys/values and to
    the other modality's; the two contexts are averaged (official code divides by 2)
    and passed through one output projection shared by both modalities."""
    cfg: CATfusionConfig

    @nn.compact
    def __call__(self, img, gen, deterministic: bool):
        c = self.cfg
        h = lambda name, x: split_heads(nn.Dense(c.hidden, name=name)(x), c.heads)
        qi, ki, vi = h("query", img), h("key", img), h("value", img)
        qg, kg, vg = h("query_text", gen), h("key_text", gen), h("value_text", gen)

        img_img, probs = Attend(c.attention_dropout)(qi, ki, vi, deterministic)
        gen_gen, _ = Attend(c.attention_dropout)(qg, kg, vg, deterministic)
        img_gen, _ = Attend(c.attention_dropout)(qi, kg, vg, deterministic)  # image asks genomics
        gen_img, _ = Attend(c.attention_dropout)(qg, ki, vi, deterministic)  # genomics asks image
        self.sow("intermediates", "attention", probs)

        out = nn.Dense(c.hidden, name="out")  # shared, as in the official code
        img = out(merge_heads((img_img + img_gen) / 2))
        gen = out(merge_heads((gen_gen + gen_img) / 2))
        img = nn.Dropout(c.attention_dropout)(img, deterministic)
        gen = nn.Dropout(c.attention_dropout)(gen, deterministic)
        return img, gen


class SelfBlock(nn.Module):
    """Pre-norm transformer block (paper Fig. 1e)."""
    cfg: CATfusionConfig

    @nn.compact
    def __call__(self, x, deterministic: bool):
        x = x + SelfAttention(self.cfg)(nn.LayerNorm(LN_EPS)(x), deterministic)
        return x + Mlp(self.cfg)(nn.LayerNorm(LN_EPS)(x), deterministic)


class CrossBlock(nn.Module):
    """Bidirectional multimodal attention block (paper Fig. 1c)."""
    cfg: CATfusionConfig

    @nn.compact
    def __call__(self, img, gen, deterministic: bool):
        a_img, a_gen = BidirectionalAttention(self.cfg)(
            nn.LayerNorm(LN_EPS, name="norm_img")(img), nn.LayerNorm(LN_EPS, name="norm_gen")(gen), deterministic)
        img, gen = img + a_img, gen + a_gen
        img = img + Mlp(self.cfg, name="mlp_img")(nn.LayerNorm(LN_EPS, name="ffn_norm_img")(img), deterministic)
        gen = gen + Mlp(self.cfg, name="mlp_gen")(nn.LayerNorm(LN_EPS, name="ffn_norm_gen")(gen), deterministic)
        return img, gen


class ScaledSigmoidHead(nn.Module):
    """Official survival head: Linear-ReLU-Dropout x2, then sigmoid * 8 - 4 (zero-initialised)."""
    cfg: CATfusionConfig

    @nn.compact
    def __call__(self, h, deterministic: bool):
        for width in self.cfg.head_dims:
            h = nn.Dropout(self.cfg.head_dropout)(nn.relu(nn.Dense(width)(h)), deterministic)
        logit = nn.Dense(1, kernel_init=nn.initializers.zeros)(h)[..., 0]  # zero_head=True
        return jax.nn.sigmoid(logit) * self.cfg.output_range + self.cfg.output_shift


class GenomicEmbedding(nn.Module):
    """Linear projection + positional embedding for each omics type, concatenated."""
    cfg: CATfusionConfig

    @nn.compact
    def __call__(self, omics: dict, genomics_present, deterministic: bool):
        c = self.cfg
        tokens = []
        for name in c.omics_names:
            n_tok = c.omics_tokens[name]
            emb = nn.Dense(c.hidden, name=f"embed_{name}")(omics[name])
            # Extension A: learned stand-in for a patient without genomic data.
            missing = self.param(f"missing_{name}", nn.initializers.normal(0.02), (n_tok, c.hidden))
            emb = jnp.where(genomics_present[:, None, None], emb, missing[None])
            emb = emb + self.param(f"pos_{name}", nn.initializers.zeros, (1, n_tok, c.hidden))
            tokens.append(nn.Dropout(c.dropout)(emb, deterministic))
        return jnp.concatenate(tokens, axis=1)


class CATfusion(nn.Module):
    cfg: CATfusionConfig

    @nn.compact
    def __call__(self, wsi, omics: dict, genomics_present=None, deterministic: bool = True):
        """wsi: [B, 100, 768]; omics: {name: [B, T, 1024]}; returns risk [B] (higher = worse)."""
        c = self.cfg
        b = wsi.shape[0]
        if genomics_present is None:
            genomics_present = jnp.ones((b,), bool)

        img = nn.Dense(c.hidden, name="patch_embed")(wsi)
        cls = self.param("cls_token", nn.initializers.zeros, (1, 1, c.hidden))
        img = jnp.concatenate([jnp.broadcast_to(cls, (b, 1, c.hidden)), img], axis=1)
        img = img + self.param("pos_img", nn.initializers.zeros, (1, img.shape[1], c.hidden))
        img = nn.Dropout(c.dropout)(img, deterministic)
        gen = GenomicEmbedding(c)(omics, genomics_present, deterministic)

        for i in range(c.num_cross_layers):
            img, gen = CrossBlock(c, name=f"cross_{i}")(img, gen, deterministic)
        x = jnp.concatenate([img, gen], axis=1)
        for i in range(c.num_layers - c.num_cross_layers):
            x = SelfBlock(c, name=f"self_{i}")(x, deterministic)
        x = nn.LayerNorm(LN_EPS, name="encoder_norm")(x)

        pooled = jnp.concatenate([x[:, 0], x[:, 1:].mean(axis=1)], axis=-1)
        return ScaledSigmoidHead(c)(pooled, deterministic)
