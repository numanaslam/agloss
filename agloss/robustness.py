"""Robustness and control checks on saved checkpoints (no training).

  python -m agloss.robustness --out-root results --shenzhen-root DATA/Shenzhen --montgomery-root DATA/Montgomery

Writes into results/robustness/ (one row per checkpoint and split unless noted):
  thresholds_<arch>.csv   CAM-IoU of the Grad-CAM at thresholds 0.3-0.7
  layer_<arch>.csv        Energy Ratio, Pointing Game, CAM-IoU of the Grad-CAM taken one stage
                          earlier (14 x 14 instead of 7 x 7) next to the default layer
  ablation_<arch>.csv     AUC with the lungs removed by zeroing or by smooth diffusion inpainting
                          (no sharp mask edge), and the same for a CONTROL region of equal area
                          outside the lungs, to separate the effect of the lungs from the effect
                          of removing that much image
  activity_<arch>.csv     share of images on which the guidance term is still active (> 0) for the
                          final model, on the training and test images; first and last epoch with
                          a non-zero term from the training history
  compute_<arch>.csv      time per training step and peak GPU memory for each configuration
                          (one row per configuration), plus mean minutes and epochs per run
  duplicates.csv          candidate duplicate image pairs (grey-scale correlation >= 0.95 at 64 x 64,
                          re-checked at full resolution) within and across the two datasets, and how
                          many seeds put the two images of a within-dataset pair on different sides
                          of the train/test split

Use --parts to run a subset (thresholds layer ablation activity compute duplicates).
Completed checkpoints are recorded and skipped on re-run.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import expand_methods, make_config
from .data import GpuData, load_arrays, make_split, perturb_masks
from .losses import (anatomical_guidance_loss, focal_loss, inside_outside_means, outside_topk_mean,
                     saliency_mass_loss, tversky_loss)
from .metrics import alignment_per_image, roc_auc_score
from .model import CamNet, grad_cam
from .train import load_checkpoint

OTHER = {"shenzhen": "montgomery", "montgomery": "shenzhen"}
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
EARLY_SPLIT = {"resnet50": 7, "convnext_tiny": 6}     # features[:k] ends at the 14 x 14 stage
PARTS = ("thresholds", "layer", "ablation", "activity", "compute", "duplicates")
BENCH_METHODS = ["vanilla", "focal", "focal_tversky", "anat_only", "full", "anat_only_topk", "anat_only_mass"]


def forward_split(model, x, k):
    """Forward pass that also returns the activations after features[:k]."""
    early = model.features[:k](x)
    feats = model.features[k:](early)
    pooled = F.adaptive_avg_pool2d(feats, 1)
    if model.arch == "convnext_tiny":
        pooled = model.head_norm(pooled)
    return model.fc(torch.flatten(pooled, 1)), feats, early


@torch.no_grad()
def p_tb(model, x):
    return torch.softmax(model(x)[0].float(), dim=1)[:, 1]


def inpaint(x, hole, coarse_iters=60, fine_iters=20):
    """Fill `hole` (B x 1 x H x W, 1 = fill) smoothly from the surrounding pixels (diffusion
    inpainting): a fill at 1/4 resolution, upsampled, then a short pass at full resolution."""
    known = 1.0 - hole
    h, w = x.shape[-2:]
    ks = F.avg_pool2d(known, 4)
    xs = F.avg_pool2d(x * known, 4) / ks.clamp_min(1e-6)
    holes = (ks < 0.5).float()
    mean = (xs * (1 - holes)).sum(dim=(2, 3), keepdim=True) / (1 - holes).sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    ys = xs * (1 - holes) + mean * holes
    for _ in range(coarse_iters):
        ys = xs * (1 - holes) + F.avg_pool2d(ys, 5, stride=1, padding=2, count_include_pad=False) * holes
    y = x * known + F.interpolate(ys, size=(h, w), mode="bilinear", align_corners=False) * hole
    for _ in range(fine_iters):
        y = x * known + F.avg_pool2d(y, 5, stride=1, padding=2, count_include_pad=False) * hole
    return y


def control_region(m, gen, block=16):
    """Random blocks lying (almost) entirely outside the lungs, of total area about that of the lungs."""
    b, _, h, w = m.shape
    outside = (1.0 - m)
    count = (F.avg_pool2d(outside, block, stride=block) * block * block)[:, 0].cpu().numpy()   # B x h/b x w/b
    need = m.flatten(1).sum(1).cpu().numpy()
    sel = np.zeros_like(count)
    rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen)))
    for i in range(b):
        cand = np.flatnonzero(count[i].ravel() >= 0.9 * block * block)
        got = 0.0
        for c in rng.permutation(cand):
            if got >= need[i]:
                break
            sel[i].flat[c] = 1.0
            got += count[i].flat[c]
    sel = torch.from_numpy(sel).to(m.device).unsqueeze(1)
    return F.interpolate(sel, scale_factor=block, mode="nearest") * outside


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def analyse(model, cfg, data, idx, parts, gen):
    """All per-checkpoint measurements for one set of images."""
    acc = {k: [] for k in ("iou_t", "e7", "g7", "i7", "e14", "g14", "i14", "p_full", "p_lz", "p_li",
                           "p_cz", "p_ci", "lung_frac", "ctrl_frac", "active")}
    k = EARLY_SPLIT.get(cfg.arch)
    for b0 in range(0, len(idx), 32):
        x, _, m = data.batch(idx[b0:b0 + 32], augment=False, masked_input=cfg.masked_input)
        with torch.enable_grad():
            if k is not None and "layer" in parts:
                logits, feats, early = forward_split(model, x, k)
                cam14 = grad_cam(logits, early, x.shape[-2:], target_class=1).detach()
            else:
                logits, feats = model(x)
            cam, cam_raw = grad_cam(logits, feats, x.shape[-2:], target_class=1, return_raw=True)
            cam, cam_raw = cam.detach(), cam_raw.detach()
        if "thresholds" in parts:
            acc["iou_t"].append(torch.stack([alignment_per_image(cam, m, t)[2] for t in THRESHOLDS], 1))
        if "layer" in parts and k is not None:
            for tag, c in (("7", cam), ("14", cam14)):
                e, g, i = alignment_per_image(c, m)
                acc["e" + tag].append(e); acc["g" + tag].append(g); acc["i" + tag].append(i)
        if "ablation" in parts:
            ctrl = control_region(m, gen)
            acc["p_full"].append(p_tb(model, x))
            acc["p_lz"].append(p_tb(model, x * (1 - m)))
            acc["p_li"].append(p_tb(model, inpaint(x, m)))
            acc["p_cz"].append(p_tb(model, x * (1 - ctrl)))
            acc["p_ci"].append(p_tb(model, inpaint(x, ctrl)))
            acc["lung_frac"].append(m.flatten(1).mean(1)); acc["ctrl_frac"].append(ctrl.flatten(1).mean(1))
        if "activity" in parts and cfg.use_anat:
            mm = perturb_masks(m, cfg.mask_perturb, cfg.mask_perturb_px)
            if cfg.cam_norm == "mass":
                total = cam_raw.sum(dim=(1, 2, 3))
                outside = (cam_raw * (1.0 - mm)).sum(dim=(1, 2, 3)) / (total + 1e-8)
                act = outside > cfg.mass_tau
            else:
                r, p = inside_outside_means(cam, mm)
                if cfg.outside_stat == "topk":
                    p = outside_topk_mean(cam, mm, cfg.topk_frac)
                act = (p - cfg.rho * r) > 0
            acc["active"].append(act.float())
    return {key: torch.cat(v).float().cpu().numpy() for key, v in acc.items() if v}


def history_activity(path):
    """First and last epoch with a non-zero guidance term, and the share of epochs after the ramp-in."""
    h = json.load(open(path))["history"]
    on = [e for e in h if e.get("s", 0) > 0]
    act = [e["epoch"] for e in on if abs(e.get("anat", 0.0)) > 1e-6]
    return (act[0] if act else "", act[-1] if act else "", len(act) / len(on) if on else float("nan"), len(h))


def run_checkpoints(args, data, parts, device):
    out = Path(args.out_root) / "robustness"
    files = {"thresholds": ["arch", "method", "dataset", "seed", "split"] + [f"iou_{t}" for t in THRESHOLDS],
             "layer": ["arch", "method", "dataset", "seed", "split", "er_7x7", "pg_7x7", "iou_7x7",
                       "er_14x14", "pg_14x14", "iou_14x14"],
             "ablation": ["arch", "method", "dataset", "seed", "split", "auc_full", "auc_lungs_zero",
                          "auc_lungs_inpaint", "auc_control_zero", "auc_control_inpaint",
                          "lung_area", "control_area"],
             "activity": ["arch", "method", "dataset", "seed", "split", "share_active",
                          "first_active_epoch", "last_active_epoch", "share_epochs_active", "epochs_run"]}
    for p in parts:
        if p in files and not (out / f"{p}_{args.arch}.csv").is_file():
            with open(out / f"{p}_{args.arch}.csv", "w", newline="") as f:
                csv.writer(f).writerow(files[p])
    done_path = out / f"done_{args.arch}.txt"
    done = set(done_path.read_text().split()) if done_path.is_file() else set()
    ckpts = []
    for m in expand_methods(args.methods):
        ckpts += sorted((Path(args.out_root) / m).glob(f"*/{args.arch}/seed_*_best.pt"))
    tag = "+".join(sorted(p for p in parts if p in files))
    todo = [c for c in ckpts if f"{tag}:{'/'.join(c.parts[-4:])}" not in done]
    print(f"{len(ckpts)} checkpoints, {len(todo)} to analyse ({args.arch}; {tag})", flush=True)
    t0 = time.time()
    for n, ck in enumerate(todo):
        model, cfg = load_checkpoint(ck, device)
        gen = torch.Generator().manual_seed(cfg.seed)
        labels = data[cfg.dataset].arrays.labels
        train_idx, _, test_idx = make_split(labels, cfg.seed, cfg.test_fraction, cfg.n_folds, cfg.used_fold)
        target = OTHER[cfg.dataset]
        jobs = [("test", data[cfg.dataset], test_idx, parts),
                ("transfer", data[target], np.arange(len(data[target].arrays.labels)), parts)]
        if "activity" in parts and cfg.use_anat:
            jobs.append(("train", data[cfg.dataset], train_idx, {"activity"}))
        hist = history_activity(ck.with_name(ck.name.replace("_best.pt", "_history.json"))) \
            if cfg.use_anat and ck.with_name(ck.name.replace("_best.pt", "_history.json")).is_file() else ("", "", "", "")
        rows = {p: [] for p in files}
        for split, d, idx, job_parts in jobs:
            r = analyse(model, cfg, d, idx, job_parts, gen)
            y = d.arrays.labels[idx]
            key = [cfg.arch, cfg.method, cfg.dataset, cfg.seed, split]
            if split != "train":
                if "iou_t" in r:
                    rows["thresholds"].append(key + [f"{v:.4f}" for v in np.nanmean(r["iou_t"], 0)])
                if "e7" in r:
                    rows["layer"].append(key + [f"{np.nanmean(r[c]):.4f}" for c in ("e7", "g7", "i7", "e14", "g14", "i14")])
                if "p_full" in r:
                    rows["ablation"].append(key + [f"{auc(y, r[c]):.4f}" for c in ("p_full", "p_lz", "p_li", "p_cz", "p_ci")]
                                            + [f"{r['lung_frac'].mean():.4f}", f"{r['ctrl_frac'].mean():.4f}"])
            if "active" in r:
                rows["activity"].append(key + [f"{r['active'].mean():.4f}"] + [str(v) for v in hist])
        for p, rr in rows.items():
            if p in parts and rr:
                with open(out / f"{p}_{args.arch}.csv", "a", newline="") as f:
                    csv.writer(f).writerows(rr)
        with open(done_path, "a") as f:
            f.write(f"{tag}:{'/'.join(ck.parts[-4:])}\n")
        el = (time.time() - t0) / 60
        print(f"[{n + 1}/{len(todo)}] {cfg.method} {cfg.dataset} seed {cfg.seed} | {el:.0f} min, "
              f"about {el / (n + 1) * (len(todo) - n - 1):.0f} min left", flush=True)


def run_compute(args, data, device, steps=30, warmup=5):
    """Time per training step and peak memory for each configuration, measured with the real step."""
    out = Path(args.out_root) / "robustness" / f"compute_{args.arch}.csv"
    d = data["shenzhen"]
    train_idx, _, _ = make_split(d.arrays.labels, 42)
    rows = []
    for method in BENCH_METHODS:
        cfg = make_config(method, arch=args.arch)
        torch.manual_seed(0)
        model = CamNet(cfg.arch, pretrained=False).to(device).train()
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        gen = torch.Generator(device=device).manual_seed(0)
        need_cam = cfg.use_tversky or cfg.use_anat
        if device.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times = []
        for it in range(warmup + steps):
            idx = train_idx[(it * cfg.batch_size) % (len(train_idx) - cfg.batch_size):][:cfg.batch_size]
            x, y, m = d.batch(idx, augment=cfg.augment, generator=gen, masked_input=cfg.masked_input)
            t = time.perf_counter()
            logits, feats = model(x)
            total = focal_loss(logits, y, cfg.focal_gamma, cfg.focal_alpha, cfg.label_smoothing)
            if need_cam:
                cam, cam_raw = grad_cam(logits, feats, x.shape[-2:], target_class=1, create_graph=True, return_raw=True)
                if cfg.use_tversky:
                    total = total + cfg.tv_lambda * tversky_loss(cam, m, cfg.tv_alpha, cfg.tv_beta, cfg.tv_eps)
                if cfg.use_anat:
                    l_ag = (saliency_mass_loss(cam_raw, m, cfg.mass_tau) if cfg.cam_norm == "mass" else
                            anatomical_guidance_loss(cam, m, cfg.rho, cfg.positivity_clamp, cfg.outside_stat, cfg.topk_frac))[0]
                    total = total + cfg.anat_lambda * l_ag
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            if it >= warmup:
                times.append(time.perf_counter() - t)
        peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else float("nan")
        mins, eps = [], []
        for ps in (Path(args.out_root) / method).glob(f"*/{args.arch}/per_seed.csv"):
            for r in csv.DictReader(open(ps)):
                mins.append(float(r["minutes"])); eps.append(float(r["epochsRun"]))
        rows.append([args.arch, method, f"{1000 * np.median(times):.1f}", f"{peak:.0f}",
                     f"{np.mean(mins):.1f}" if mins else "", f"{np.mean(eps):.1f}" if eps else "", len(mins)])
        print(f"{method:16s} {1000 * np.median(times):7.1f} ms/step  peak {peak:7.0f} MB", flush=True)
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "method", "ms_per_step", "peak_mem_mb", "minutes_per_run", "epochs_per_run", "runs"])
        w.writerows(rows)
    print(f"wrote {out}")


def dhash(images):
    """64-bit difference hash of each grey-scale image (N x H x W uint8). Too coarse for chest
    X-rays, which share their layout: kept only for reference; run_duplicates uses correlation."""
    g = torch.from_numpy(np.ascontiguousarray(images)).float().unsqueeze(1)
    s = F.adaptive_avg_pool2d(g, (8, 9))[:, 0]
    return (s[:, :, 1:] > s[:, :, :-1]).reshape(len(images), -1).numpy()


def unit_vectors(images, size):
    """Grey-scale images (N x H x W, any dtype) as zero-mean, unit-norm vectors at `size` x `size`."""
    g = F.adaptive_avg_pool2d(images.float().unsqueeze(1), (size, size)).flatten(1)
    g = g - g.mean(dim=1, keepdim=True)
    return g / g.norm(dim=1, keepdim=True).clamp_min(1e-8)


def run_duplicates(args, data, threshold=0.95):
    """Candidate duplicates: Pearson correlation >= `threshold` between 64 x 64 grey-scale versions,
    re-checked at full resolution. Distinct patients rarely exceed 0.99; copies of one image do."""
    out = Path(args.out_root) / "robustness" / "duplicates.csv"
    grey = {n: d.images[:, 0] for n, d in data.items()}                       # N x H x W uint8 on the device
    small = {n: unit_vectors(g, 64) for n, g in grey.items()}
    full = {n: unit_vectors(g, g.shape[-1]) for n, g in grey.items()}
    test_sets = {n: [set(make_split(d.arrays.labels, s)[2].tolist()) for s in range(42, 50)] for n, d in data.items()}
    rows, maxima = [], {}
    for a, b in (("shenzhen", "shenzhen"), ("montgomery", "montgomery"), ("shenzhen", "montgomery")):
        c = small[a] @ small[b].T
        if a == b:
            c = torch.triu(c, diagonal=1)
        maxima[f"{a}-{b}"] = float(c.max())
        for i, j in torch.nonzero(c >= threshold).tolist():
            c_full = float(full[a][i] @ full[b][j])
            mad = float((grey[a][i].float() - grey[b][j].float()).abs().mean())
            cross = sum((i in te) != (j in te) for te in test_sets[a]) if a == b else ""
            rows.append([a, data[a].arrays.names[i], int(data[a].arrays.labels[i]), b, data[b].arrays.names[j],
                         int(data[b].arrays.labels[j]), f"{float(c[i, j]):.4f}", f"{c_full:.4f}", f"{mad:.2f}", cross])
    rows.sort(key=lambda r: -float(r[7]))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset_a", "name_a", "label_a", "dataset_b", "name_b", "label_b",
                    "corr_64", "corr_full", "mean_abs_diff_0_255", "seeds_crossing_train_test"])
        w.writerows(rows)
    for t in (0.95, 0.98, 0.99, 0.995):
        print(f"pairs with full-resolution correlation >= {t}: {sum(float(r[7]) >= t for r in rows)}")
    print("highest 64 x 64 correlation per comparison: " + ", ".join(f"{k} {v:.4f}" for k, v in maxima.items()))
    print(f"wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--shenzhen-root", required=True)
    ap.add_argument("--montgomery-root", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--methods", nargs="+", default=["main", "full_noclamp", "ablations", "variants", "maskperturb"])
    ap.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots = dict(shenzhen=args.shenzhen_root, montgomery=args.montgomery_root)
    data = {n: GpuData(load_arrays(n, roots[n], 224, args.cache_dir, verbose=False), device) for n in roots}
    (Path(args.out_root) / "robustness").mkdir(parents=True, exist_ok=True)
    parts = set(args.parts)
    if "duplicates" in parts:
        run_duplicates(args, data)
    ck_parts = parts & {"thresholds", "layer", "ablation", "activity"}
    if ck_parts:
        run_checkpoints(args, data, ck_parts, device)
    if "compute" in parts:
        run_compute(args, data, device)


if __name__ == "__main__":
    main()
