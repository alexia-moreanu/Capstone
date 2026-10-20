"""Model and training configurations.

Every default below is copied from the authors' released code
(github.com/Wanglabsmu/CATfusion) unless the comment says otherwise, so the
`paper()` presets are the replication settings and anything else is a choice
we made and document in REPLICATION.md.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

# Order matters: it fixes the order in which omics tokens are concatenated.
OMICS_ORDER = ("rna_seq", "mirna_seq", "copynumbervariation", "dnamethylationvariation", "snp")

# Tokens per omics type in the official CATfusion config (models/configs.py).
# Each token is one 1024-gene window produced by the TabAE encoder.
PAPER_OMICS_TOKENS = {
    "rna_seq": 10,  # 10,240 most variable genes
    "mirna_seq": 2,  # ~1,900 expressed miRNAs -> 2 overlapping windows
    "copynumbervariation": 27,  # ~27k genes (GISTIC2 by-gene)
    "dnamethylationvariation": 8,  # 8,192 most variable CpGs
    "snp": 10,  # 10,240 gene x variant-type mutation counts
}


@dataclass
class CATfusionConfig:
    wsi_dim: int = 768  # CTransPath feature size
    wsi_tokens: int = 100  # mosaics sampled per slide
    omics_dim: int = 1024  # TabAE latent size per token
    omics_tokens: dict = field(default_factory=lambda: dict(PAPER_OMICS_TOKENS))
    hidden: int = 768
    mlp_dim: int = 3072
    heads: int = 12
    num_layers: int = 12  # total transformer blocks
    num_cross_layers: int = 2  # first blocks are bidirectional cross-attention
    attention_dropout: float = 0.2
    dropout: float = 0.3
    head_dims: tuple = (64, 16)
    head_dropout: float = 0.5
    output_range: float = 8.0  # risk = sigmoid(.) * 8 - 4, as in the official head
    output_shift: float = -4.0

    @property
    def omics_names(self) -> tuple:
        return tuple(name for name in OMICS_ORDER if name in self.omics_tokens)

    @classmethod
    def paper(cls, omics=OMICS_ORDER) -> "CATfusionConfig":
        return cls(omics_tokens={k: PAPER_OMICS_TOKENS[k] for k in omics})

    @classmethod
    def tiny(cls, omics_tokens: dict) -> "CATfusionConfig":
        """Small model for CPU smoke tests. Same architecture, fewer/narrower layers."""
        return cls(omics_tokens=dict(omics_tokens), hidden=64, mlp_dim=128, heads=4,
                   num_layers=3, num_cross_layers=2)


@dataclass
class TabAEConfig:
    window: int = 1024  # sliding-window length = token size (official getData gap)
    mask_ratio: float = 0.4
    embed_dim: int = 1024  # must equal window: windows are fed in without projection
    depth: int = 4
    heads: int = 16
    decoder_dim: int = 512
    decoder_depth: int = 2
    decoder_heads: int = 16
    mlp_ratio: float = 4.0
    # optimiser: SGD(momentum=0.9, wd=5e-5), cosine schedule lr -> lr*lrf
    lr: float = 1e-3
    lrf: float = 0.01
    weight_decay: float = 5e-5
    batch_size: int = 64
    epochs: int = 100  # official default is 1 on the CLI; real value not reported

    @classmethod
    def tiny(cls) -> "TabAEConfig":
        return cls(depth=1, heads=4, decoder_dim=64, decoder_depth=1, decoder_heads=4,
                   epochs=5, lr=1e-2)


@dataclass
class TrainConfig:
    model: str = "catfusion"
    epochs: int = 100  # not reported in the paper; official CLI default is 1
    batch_size: int = 64
    lr: float = 1e-5
    weight_decay: float = 0.7  # coupled L2, like torch.optim.Adam(weight_decay=0.7)
    lr_step_epochs: int = 50  # StepLR(step_size=50, gamma=1.2) -- lr *grows* every 50 epochs
    lr_gamma: float = 1.2
    folds: int = 5
    repeats: int = 1
    split_seed: int = 24  # StratifiedKFold(random_state=24)
    seed: int = 1
    # "official": keep the epoch with the lowest loss on the *test* fold (what the
    #              released code does; optimistic).
    # "inner_val": hold out part of the training fold for model selection.
    # "last":      keep the final epoch.
    selection: str = "inner_val"
    inner_val_fraction: float = 0.15
    # "official": risk sets from batch order (released CoxLoss2)
    # "exact":    risk sets from survival times (textbook Cox partial likelihood)
    cox_loss: str = "exact"
    # Extensions (Part B5.2 of the proposal)
    genomics_dropout: float = 0.0  # P(drop all genomics for a patient) during training
    distill_weight: float = 0.0  # lambda in (1-lambda)*Cox + lambda*MSE(student, teacher)

    @classmethod
    def paper(cls, **overrides) -> "TrainConfig":
        """Exactly what the released training script does."""
        base = dict(selection="official", cox_loss="official")
        base.update(overrides)
        return cls(**base)


def to_json(obj) -> str:
    return json.dumps(dataclasses.asdict(obj), indent=2, default=list)


def from_dict(cls, d: dict):
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})
