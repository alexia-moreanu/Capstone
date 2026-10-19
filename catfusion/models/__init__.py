from catfusion.models.catfusion import CATfusion
from catfusion.models.baselines import AttMILWSI, AttMILGenomic, PooledFusion
from catfusion.models.tabae import TabAE, sliding_windows

# name -> (constructor, which inputs the model actually uses)
MODELS = {
    "catfusion": (lambda cfg: CATfusion(cfg), "wsi+genomics"),
    "attmil_wsi": (lambda cfg: AttMILWSI(cfg), "wsi"),
    "attmil_genomic": (lambda cfg: AttMILGenomic(cfg), "genomics"),
    "fusion_concat": (lambda cfg: PooledFusion(cfg, "concat"), "wsi+genomics"),
    "fusion_kronecker": (lambda cfg: PooledFusion(cfg, "kronecker"), "wsi+genomics"),
    "fusion_lowrank": (lambda cfg: PooledFusion(cfg, "lowrank"), "wsi+genomics"),
}


def build_model(name: str, cfg):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name][0](cfg)
