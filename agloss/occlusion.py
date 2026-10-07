"""Second explainer that was NOT optimised during training: occlusion sensitivity.

For every saved checkpoint, slides a patch over each in-domain test image, records the
drop in P(TB), and scores the resulting map against the lung mask with the same three
alignment metrics used for Grad-CAM. Because no loss ever touched these maps, they test
whether the classifier's decision actually depends on the lung region.

  python -m agloss.occlusion --out-root results --shenzhen-root ... --montgomery-root ...

Writes results/occlusion.csv (one row per method, dataset, seed).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import GpuData, load_arrays, make_split
from .metrics import alignment_metrics
from .train import data_root, load_checkpoint


@torch.no_grad()
def occlusion_maps(model, x, patch: int = 32, stride: int = 16, chunk: int = 256):
    """x: B x 3 x H x W (normalised). Returns B x 1 x H x W maps in [0, 1].

    The occluding value is 0 in normalised space, i.e. the ImageNet mean intensity.
    """
    b, _, h, w = x.shape
    base = torch.softmax(model(x)[0], dim=1)[:, 1]
    tops = list(range(0, h - patch + 1, stride))
    lefts = list(range(0, w - patch + 1, stride))
    heat = torch.zeros(b, 1, h, w, device=x.device)
    count = torch.zeros(1, 1, h, w, device=x.device)
    for i in range(b):
        occluded, boxes = [], []
        for t in tops:
            for l in lefts:
                xi = x[i].clone()
                xi[:, t:t + patch, l:l + patch] = 0.0
                occluded.append(xi)
                boxes.append((t, l))
        p = torch.cat([torch.softmax(model(torch.stack(occluded[j:j + chunk]))[0], dim=1)[:, 1]
                       for j in range(0, len(occluded), chunk)])
        drop = F.relu(base[i] - p)                    # evidence FOR TB removed by the patch
        for (t, l), d in zip(boxes, drop):
            heat[i, 0, t:t + patch, l:l + patch] += d
            if i == 0:
                count[0, 0, t:t + patch, l:l + patch] += 1
    heat = heat / count.clamp_min(1)
    hmin = heat.amin(dim=(2, 3), keepdim=True)
    hmax = heat.amax(dim=(2, 3), keepdim=True)
    return (heat - hmin) / (hmax - hmin + 1e-8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--shenzhen-root", required=True)
    ap.add_argument("--montgomery-root", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots = dict(shenzhen=args.shenzhen_root, montgomery=args.montgomery_root)
    cache, rows = {}, []
    for ck in sorted(Path(args.out_root).glob(f"*/*/{args.arch}/seed_*_best.pt")):
        model, cfg = load_checkpoint(ck, device)
        if cfg.dataset not in cache:
            cache[cfg.dataset] = GpuData(load_arrays(cfg.dataset, roots[cfg.dataset], cfg.image_size,
                                                     args.cache_dir, verbose=False), device)
        data = cache[cfg.dataset]
        _, _, test_idx = make_split(data.arrays.labels, cfg.seed, cfg.test_fraction, cfg.n_folds, cfg.used_fold)
        maps, masks = [], []
        for b0 in range(0, len(test_idx), 16):
            x, _, m = data.batch(test_idx[b0:b0 + 16], augment=False, masked_input=cfg.masked_input)
            maps.append(occlusion_maps(model, x, args.patch, args.stride))
            masks.append(m)
        maps, masks = torch.cat(maps), torch.cat(masks)
        flat = maps.flatten(1).amax(dim=1) > 0          # images where no patch lowered P(TB) carry no map
        a = alignment_metrics(maps[flat], masks[flat]) if flat.any() else dict(energyRatio=np.nan, pointingGame=np.nan, iou=np.nan)
        rows.append(dict(method=cfg.method, dataset=cfg.dataset, seed=cfg.seed, nImages=int(flat.sum()),
                         occER=a["energyRatio"], occPG=a["pointingGame"], occIoU=a["iou"]))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out_root) / "occlusion.csv", index=False)
    print(df.groupby(["dataset", "method"])[["occER", "occPG", "occIoU"]].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
