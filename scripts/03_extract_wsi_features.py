"""Step 3 - whole-slide images -> SISH mosaics -> CTransPath features (PyTorch; run on your Mac or Colab).

    # one slide file
    python scripts/03_extract_wsi_features.py --slides path/to/*.svs --out data/cohort/TCGA-BRCA/slides \
        --weights checkpoints/ctranspath.pth
    # download from the GDC in batches, extract, delete the image (proposal A3/A4)
    python scripts/03_extract_wsi_features.py --manifest data/raw/TCGA-BRCA/manifest_slides.tsv \
        --out data/cohort/TCGA-BRCA/slides --weights checkpoints/ctranspath.pth --batch 10 --delete

Output: one <slide>.h5 per slide with `features` [n_mosaics, 768] and `coords`
[n_mosaics, 2] (level-0 pixels), the same format as the official code reads.
Already-finished slides are skipped, so the job can be stopped and restarted.

Pipeline (paper: "follows SISH", Chen et al. 2022):
  1. tissue mask on a thumbnail (HSV saturation, median blur, Otsu)
  2. 1024 x 1024 patches at 20x on the tissue grid
  3. mosaic = two-stage k-means: 9 colour clusters on RGB histograms, then within each
     cluster k-means on (x, y) choosing 5% of patches (min 1), nearest patch to each centre
  4. each mosaic resized to 224 x 224, ImageNet-normalised, encoded by CTransPath (768-d)

CTransPath weights: download ctranspath.pth from the TransPath repository
(github.com/Xiyue-Wang/TransPath) and pass --weights. Without --weights the encoder
is randomly initialised; that is only useful for testing the pipeline.
Device: --device mps uses the Apple-silicon GPU.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

import h5py
import numpy as np
import requests
import torch
import torch.nn as nn

GDC = "https://api.gdc.cancer.gov"


# ----------------------------------------------------------------------------- CTransPath
class ConvStem(nn.Module):
    """CTransPath's convolutional patch embedding (TransPath repo), compatible with timm >= 0.9."""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None,
                 flatten=True, output_fmt=None, **kwargs):
        super().__init__()
        img_size = tuple(img_size) if isinstance(img_size, (tuple, list)) else (img_size, img_size)
        patch_size = patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size
        self.img_size, self.patch_size = img_size, (patch_size, patch_size)
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.dynamic_img_pad = False
        self.strict_img_size = True
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.output_fmt = output_fmt
        self.flatten = flatten and output_fmt is None
        stem, c_in, c_out = [], in_chans, embed_dim // 8
        for _ in range(2):
            stem += [nn.Conv2d(c_in, c_out, 3, 2, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            c_in, c_out = c_out, c_out * 2
        stem.append(nn.Conv2d(c_in, embed_dim, 1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.output_fmt == "NHWC" or str(self.output_fmt).endswith("NHWC"):
            x = x.permute(0, 2, 3, 1)
        elif self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


def _remap_old_timm_keys(state: dict, model_keys: set) -> dict:
    """ctranspath.pth was saved with timm 0.5.4, where PatchMerging sat at the *end* of
    stage i; newer timm puts it at the *start* of stage i+1. Shift those keys if needed."""
    old_style = any(k.startswith("layers.0.downsample.") for k in state)
    new_style = not any(k.startswith("layers.0.downsample.") for k in model_keys)
    if not (old_style and new_style):
        return state
    out = {}
    for k, v in state.items():
        m = re.match(r"layers\.(\d+)\.downsample\.(.*)", k)
        out[f"layers.{int(m.group(1)) + 1}.downsample.{m.group(2)}" if m else k] = v
    return out


def ctranspath(weights: str | None):
    import timm
    model = timm.create_model("swin_tiny_patch4_window7_224", embed_layer=ConvStem, pretrained=False, num_classes=0)
    if weights:
        state = torch.load(weights, map_location="cpu")
        state = state.get("model", state)
        state = {k: v for k, v in state.items() if not k.startswith("head")}
        state = _remap_old_timm_keys(state, set(model.state_dict()))
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if "relative_position_index" not in k and "attn_mask" not in k]
        if missing or unexpected:
            raise RuntimeError(f"CTransPath weights did not load cleanly. missing={missing[:5]} unexpected={unexpected[:5]}")
    else:
        print("WARNING: no --weights given, CTransPath is randomly initialised (pipeline test only)")
    return model.eval()


# ----------------------------------------------------------------------------- slide -> mosaics
def tissue_mask(slide, thumb_size=2048):
    from skimage.color import rgb2hsv
    from skimage.filters import median, threshold_otsu
    thumb = np.asarray(slide.get_thumbnail((thumb_size, thumb_size)).convert("RGB"))
    sat = (rgb2hsv(thumb)[..., 1] * 255).astype(np.uint8)
    sat = median(sat, np.ones((7, 7), bool))
    return sat > threshold_otsu(sat), slide.dimensions[0] / thumb.shape[1]


def level0_patch_size(slide, patch_20x=1024):
    """Patch size in level-0 pixels that corresponds to `patch_20x` pixels at 20x."""
    mag = slide.properties.get("openslide.objective-power") or slide.properties.get("aperio.AppMag") or 20
    return int(round(patch_20x * float(mag) / 20))


def tissue_grid(slide, size0, min_tissue=0.5):
    mask, scale = tissue_mask(slide)
    w, h = slide.dimensions
    coords = []
    for y in range(0, h - size0 + 1, size0):
        for x in range(0, w - size0 + 1, size0):
            m = mask[int(y / scale):int((y + size0) / scale) + 1, int(x / scale):int((x + size0) / scale) + 1]
            if m.size and m.mean() >= min_tissue:
                coords.append((x, y))
    return np.array(coords, dtype=np.int64).reshape(-1, 2)


def read_patch(slide, xy, size0, out_px):
    level = slide.get_best_level_for_downsample(size0 / out_px)
    ds = slide.level_downsamples[level]
    region = slide.read_region((int(xy[0]), int(xy[1])), level, (int(size0 / ds), int(size0 / ds))).convert("RGB")
    return region.resize((out_px, out_px))


def sish_mosaic(slide, coords, size0, n_color_clusters=9, fraction=0.05, seed=0):
    """Two-stage k-means mosaic selection (SISH). Returns the indices of the chosen patches."""
    from sklearn.cluster import KMeans
    if len(coords) <= n_color_clusters:
        return np.arange(len(coords))
    hists = []
    for xy in coords:
        small = np.asarray(read_patch(slide, xy, size0, 64))
        hists.append(np.concatenate([np.histogram(small[..., c], bins=16, range=(0, 255))[0] for c in range(3)]))
    colour = KMeans(n_color_clusters, n_init=10, random_state=seed).fit_predict(np.array(hists, float))
    chosen = []
    for c in range(n_color_clusters):
        members = np.where(colour == c)[0]
        if len(members) == 0:
            continue
        k = max(1, int(round(fraction * len(members))))
        km = KMeans(k, n_init=10, random_state=seed).fit(coords[members].astype(float))
        for centre in km.cluster_centers_:
            chosen.append(members[np.argmin(((coords[members] - centre) ** 2).sum(1))])
    return np.unique(chosen)


@torch.no_grad()
def encode(model, slide, coords, size0, device, batch=32):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    feats = []
    for s in range(0, len(coords), batch):
        imgs = np.stack([np.asarray(read_patch(slide, xy, size0, 224)) for xy in coords[s:s + batch]])
        x = (torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255 - mean) / std
        feats.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(feats).astype(np.float32)


def process_slide(path, out_dir, model, device):
    import openslide
    name = os.path.basename(path).split(".")[0]
    out = os.path.join(out_dir, f"{name}.h5")
    if os.path.exists(out):
        return out
    t0 = time.time()
    slide = openslide.OpenSlide(path)
    size0 = level0_patch_size(slide)
    grid = tissue_grid(slide, size0)
    if len(grid) == 0:
        print(f"  {name}: no tissue found, skipped")
        return None
    mosaic = grid[sish_mosaic(slide, grid, size0)]
    feats = encode(model, slide, mosaic, size0, device)
    with h5py.File(out + ".part", "w") as f:
        f.create_dataset("features", data=feats)
        f.create_dataset("coords", data=mosaic)
    os.replace(out + ".part", out)
    print(f"  {name}: {len(grid)} tissue patches -> {len(mosaic)} mosaics  ({time.time() - t0:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slides", nargs="*", default=[])
    ap.add_argument("--manifest", help="manifest_slides.tsv from 00_download_gdc.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=10, help="slides downloaded at a time")
    ap.add_argument("--download-dir", default="data/tmp_slides")
    ap.add_argument("--delete", action="store_true", help="delete each slide after extraction")
    ap.add_argument("--limit", type=int, help="process only the first N slides (pilot run)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = ctranspath(args.weights).to(args.device)

    paths = [p for pattern in args.slides for p in glob.glob(pattern)]
    for p in paths[: args.limit]:
        process_slide(p, args.out, model, args.device)

    if args.manifest:
        import pandas as pd
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        download = __import__("00_download_gdc").download
        manifest = pd.read_csv(args.manifest, sep="\t")
        manifest = manifest[manifest.file_name.str.contains("DX")]  # FFPE diagnostic slides
        if args.limit:
            manifest = manifest.head(args.limit)
        os.makedirs(args.download_dir, exist_ok=True)
        todo = [r for r in manifest.itertuples()
                if not os.path.exists(os.path.join(args.out, r.file_name.split(".")[0] + ".h5"))]
        print(f"{len(todo)} slides to process")
        for s in range(0, len(todo), args.batch):
            for r in todo[s:s + args.batch]:
                local = os.path.join(args.download_dir, r.file_name)
                download(f"{GDC}/data/{r.file_id}", local)
                process_slide(local, args.out, model, args.device)
                if args.delete:
                    os.remove(local)


if __name__ == "__main__":
    main()
