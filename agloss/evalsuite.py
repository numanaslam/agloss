"""Evaluation-only analyses on saved checkpoints (no training).

  python -m agloss.evalsuite --out-root results --shenzhen-root ... --montgomery-root ...

For every checkpoint it writes one row per image to results/eval/per_image_<arch>.csv:

  both splits (in-domain test, transfer = the whole other dataset)
    p_tb                       predicted TB probability
    er, pg, iou                Grad-CAM alignment for that image
    p_lungs_zero / _blur       prediction when everything OUTSIDE the lungs is removed
    p_outside_zero / _blur     prediction when the LUNGS are removed
  in-domain test split only
    occ_er_16 / _32 / _64      Energy Ratio of occlusion maps at three patch sizes
    del_cam, ins_cam           deletion / insertion AUC following the Grad-CAM ranking
    del_rand, ins_rand         the same with a random pixel ranking (reference)

Completed checkpoints are recorded and skipped on re-run. Summaries: `python -m agloss.evalstats`.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import expand_methods
from .data import GpuData, load_arrays, make_split
from .metrics import alignment_per_image
from .model import grad_cam
from .occlusion import occlusion_maps
from .train import load_checkpoint

FIELDS = ["arch", "method", "dataset", "seed", "split", "eval_dataset", "name", "label", "p_tb",
          "er", "pg", "iou", "p_lungs_zero", "p_outside_zero", "p_lungs_blur", "p_outside_blur",
          "occ_er_16", "occ_er_32", "occ_er_64", "del_cam", "ins_cam", "del_rand", "ins_rand"]
OTHER = {"shenzhen": "montgomery", "montgomery": "shenzhen"}
OCC = {16: 16, 32: 16, 64: 32}            # patch size -> stride
STEPS = 20


def _amp(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


@torch.no_grad()
def p_tb(model, x):
    with _amp(x.device):
        return torch.softmax(model(x)[0].float(), dim=1)[:, 1]


def blur(x):
    for _ in range(2):
        x = F.avg_pool2d(x, 31, stride=1, padding=15, count_include_pad=False)
    return x


@torch.no_grad()
def deletion_insertion(model, x, order_scores, steps: int = STEPS):
    """Deletion and insertion AUC for one image batch.  x: B x 3 x H x W, order_scores: B x 1 x H x W.

    Deletion: the most salient pixels are replaced first by the mean intensity (0 after
    normalisation); a faithful map makes P(TB) fall quickly (low AUC).
    Insertion: starting from a blurred image, the most salient pixels are restored first;
    a faithful map makes P(TB) rise quickly (high AUC).
    """
    b, _, h, w = x.shape
    rank = order_scores.flatten(1).argsort(dim=1, descending=True).argsort(dim=1).view(b, 1, h, w)
    xb = blur(x)
    dele, ins = [], []
    for s in range(steps + 1):
        top = (rank < int(round(s / steps * h * w))).float()
        dele.append(p_tb(model, x * (1 - top)))
        ins.append(p_tb(model, x * top + xb * (1 - top)))
    dele, ins = torch.stack(dele, 1), torch.stack(ins, 1)             # B x (steps+1)
    return torch.trapz(dele, dx=1.0 / steps, dim=1), torch.trapz(ins, dx=1.0 / steps, dim=1)


def analyse(model, cfg, data, idx, full: bool, gen):
    out = {k: [] for k in FIELDS[8:]}
    for b0 in range(0, len(idx), 32):
        x, _, m = data.batch(idx[b0:b0 + 32], augment=False, masked_input=cfg.masked_input)
        with torch.enable_grad():
            logits, feats = model(x)
            cam = grad_cam(logits, feats, x.shape[-2:], target_class=1).detach()
        out["p_tb"].append(torch.softmax(logits.detach(), dim=1)[:, 1])
        er, pg, iou = alignment_per_image(cam, m)
        out["er"].append(er); out["pg"].append(pg); out["iou"].append(iou)
        xb = blur(x)
        out["p_lungs_zero"].append(p_tb(model, x * m))
        out["p_outside_zero"].append(p_tb(model, x * (1 - m)))
        out["p_lungs_blur"].append(p_tb(model, x * m + xb * (1 - m)))
        out["p_outside_blur"].append(p_tb(model, x * (1 - m) + xb * m))
        if full:
            for patch, stride in OCC.items():
                with _amp(x.device):
                    om = occlusion_maps(model, x, patch, stride).float()
                e, _, _ = alignment_per_image(om, m)
                e[om.flatten(1).amax(dim=1) <= 0] = float("nan")       # no patch lowered P(TB)
                out[f"occ_er_{patch}"].append(e)
            d, i = deletion_insertion(model, x, cam)
            dr, ir = deletion_insertion(model, x, torch.rand(cam.shape, device=x.device, generator=gen))
            out["del_cam"].append(d); out["ins_cam"].append(i); out["del_rand"].append(dr); out["ins_rand"].append(ir)
    n = len(idx)
    return {k: (torch.cat(v).float().cpu().numpy() if v else np.full(n, np.nan)) for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--shenzhen-root", required=True)
    ap.add_argument("--montgomery-root", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--methods", nargs="+", default=["main", "full_noclamp", "masked_input", "variants"],
                    help="methods or groups to analyse (checkpoints that do not exist are skipped)")
    ap.add_argument("--no-occlusion", action="store_true",
                    help="skip occlusion and deletion/insertion (the slow part); predictions only")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots = dict(shenzhen=args.shenzhen_root, montgomery=args.montgomery_root)
    data = {n: GpuData(load_arrays(n, roots[n], 224, args.cache_dir, verbose=False), device) for n in roots}
    out_dir = Path(args.out_root) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, done_path = out_dir / f"per_image_{args.arch}.csv", out_dir / f"done_{args.arch}.txt"
    done = set(done_path.read_text().split()) if done_path.is_file() else set()
    if not csv_path.is_file():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(FIELDS)

    ckpts = []
    for m in expand_methods(args.methods):
        ckpts += sorted((Path(args.out_root) / m).glob(f"*/{args.arch}/seed_*_best.pt"))
    todo = [c for c in ckpts if "/".join(c.parts[-4:]) not in done]
    print(f"{len(ckpts)} checkpoints, {len(todo)} to analyse ({args.arch})", flush=True)
    t0 = time.time()
    for k, ck in enumerate(todo):
        model, cfg = load_checkpoint(ck, device)
        gen = torch.Generator(device=device).manual_seed(cfg.seed)
        labels = data[cfg.dataset].arrays.labels
        _, _, test_idx = make_split(labels, cfg.seed, cfg.test_fraction, cfg.n_folds, cfg.used_fold)
        target = OTHER[cfg.dataset]
        jobs = [("test", cfg.dataset, test_idx, not args.no_occlusion),
                ("transfer", target, np.arange(len(data[target].arrays.labels)), False)]
        rows = []
        for split, ds, idx, full in jobs:
            r = analyse(model, cfg, data[ds], idx, full, gen)
            for j, i in enumerate(idx):
                rows.append([cfg.arch, cfg.method, cfg.dataset, cfg.seed, split, ds, data[ds].arrays.names[i],
                             int(data[ds].arrays.labels[i])] + [f"{r[c][j]:.5g}" for c in FIELDS[8:]])
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerows(rows)
        with open(done_path, "a") as f:
            f.write("/".join(ck.parts[-4:]) + "\n")
        el = (time.time() - t0) / 60
        print(f"[{k + 1}/{len(todo)}] {cfg.method} {cfg.dataset} seed {cfg.seed} | "
              f"{el:.0f} min elapsed, about {el / (k + 1) * (len(todo) - k - 1):.0f} min left", flush=True)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
