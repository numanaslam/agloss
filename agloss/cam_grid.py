"""Qualitative Grad-CAM grid (no image appears twice).

Selects distinct test images from one seed's Shenzhen test split: `--n-normal` Normal,
`--n-tb` TB, and `--n-disagree` cases with the largest between-method spread in P(TB).
Duplicates are impossible by construction, and the true label is printed on each row.

  python -m agloss.cam_grid --out-root results --shenzhen-root ... --montgomery-root ... --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import DISPLAY
from .data import GpuData, load_arrays, make_split
from .train import evaluate, load_checkpoint

COLUMNS = ["vanilla", "focal_tversky", "sym_noclamp", "anat_only", "full"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--shenzhen-root", required=True)
    ap.add_argument("--montgomery-root", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--dataset", default="shenzhen")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-normal", type=int, default=2)
    ap.add_argument("--n-tb", type=int, default=2)
    ap.add_argument("--n-disagree", type=int, default=2)
    ap.add_argument("--output", default="fig06_cam_grids.pdf")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = args.shenzhen_root if args.dataset == "shenzhen" else args.montgomery_root
    data = GpuData(load_arrays(args.dataset, root, 224, args.cache_dir, verbose=False), device)
    labels, names = data.arrays.labels, data.arrays.names
    _, _, test_idx = make_split(labels, args.seed)

    probs, cams = {}, {}
    for m in COLUMNS:
        ck = Path(args.out_root) / m / args.dataset / args.arch / f"seed_{args.seed}_best.pt"
        model, cfg = load_checkpoint(ck, device)
        out = evaluate(model, data, test_idx, cfg, return_arrays=True)
        probs[m], cams[m] = out["p_tb"], out["cams"][:, 0].numpy()

    y = labels[test_idx]
    spread = np.stack([probs[m] for m in COLUMNS]).std(axis=0)
    rng = np.random.RandomState(args.seed)
    chosen = []

    def take(candidates, k):
        for j in candidates:
            if k == 0:
                break
            if j not in chosen:
                chosen.append(int(j))
                k -= 1

    by_spread = np.argsort(-spread)                                  # highest disagreement first,
    half = args.n_disagree // 2                                      # split evenly between the classes
    take([j for j in by_spread if y[j] == 1], args.n_disagree - half)
    take([j for j in by_spread if y[j] == 0], half)
    take(rng.permutation(np.flatnonzero(y == 0)), args.n_normal)
    take(rng.permutation(np.flatnonzero(y == 1)), args.n_tb)
    assert len(set(chosen)) == len(chosen)
    chosen.sort(key=lambda j: (y[j], names[test_idx[j]]))

    imgs = data.arrays.images[test_idx][:, 0]
    masks = data.arrays.masks[test_idx]
    nr, nc = len(chosen), len(COLUMNS) + 1
    fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.7 * nr), squeeze=False)
    for r, j in enumerate(chosen):
        tag = "TB" if y[j] == 1 else "Normal"
        for c in range(nc):
            ax = axes[r, c]
            ax.imshow(imgs[j], cmap="gray")
            ax.contour(masks[j], levels=[0.5], colors="yellow", linewidths=1.0)
            if c > 0:
                m = COLUMNS[c - 1]
                ax.imshow(cams[m][j], cmap="jet", alpha=0.40, vmin=0, vmax=1)
                ax.set_xlabel(f"P(TB) = {probs[m][j]:.2f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title("Input" if c == 0 else DISPLAY[COLUMNS[c - 1]], fontsize=11, fontweight="bold")
        axes[r, 0].set_ylabel(f"{names[test_idx[j]]}\n({tag})", fontsize=9)
    fig.tight_layout()
    Path(args.output).resolve().parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"wrote {args.output}; rows: " + ", ".join(f"{names[test_idx[j]]} ({'TB' if y[j] else 'Normal'})" for j in chosen))


if __name__ == "__main__":
    main()
