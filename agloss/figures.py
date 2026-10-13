"""Regenerate the manuscript's result figures and LaTeX tables from results/.

  python -m agloss.figures --out-root results

Writes into results/figures/:
  fig02_shenzhen_results.pdf, fig03_montgomery_results.pdf   in-domain bars (mean +/- SD)
  fig04_tradeoff.pdf                                         AUC vs CAM-IoU
  fig05_transfer_slopes.pdf                                  in-domain -> transfer AUC
  fig07_clamp_training.pdf                                   per-epoch loss, R and P
  fig08_rho_sensitivity.pdf                                  (if the rho sweep was run)
  tables.tex                                                 all result tables, mean +/- SD
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DISPLAY, GROUPS
from .stats import load_all

MAIN = GROUPS["main"]
COLORS = dict(vanilla="#7f7f7f", focal="#bcbd22", focal_tversky="#1f77b4", sym_noclamp="#d62728",
              anat_only="#2ca02c", full="#9467bd", full_noclamp="#ff7f0e")
OTHER = dict(shenzhen="montgomery", montgomery="shenzhen")


def _ms(df, method, ds, col):
    v = df[(df.method == method) & (df.dataset == ds)][col].dropna()
    return (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0) if len(v) else (np.nan, np.nan)


def bars(df, ds, path):
    metrics = [("testAUC", "Test AUC", "AUC"), ("energyRatio", "Energy Ratio", "Energy Ratio"),
               ("pointingGame", "Pointing Game", "Pointing Game score"), ("camIoU", "CAM-IoU", "CAM-IoU")]
    methods = [m for m in MAIN if m in set(df.method)]
    # 2 x 2 panels so that labels stay legible at page width
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for k, (ax, (col, title, ylabel)) in enumerate(zip(axes.flat, metrics)):
        ms = [_ms(df, m, ds, col) for m in methods]
        ax.bar(range(len(methods)), [a for a, _ in ms], yerr=[b for _, b in ms], capsize=4,
               color=[COLORS[m] for m in methods])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([DISPLAY[m] for m in methods], rotation=30, ha="right", fontsize=11)
        ax.set_title(f"({'abcd'[k]}) {title}", fontweight="bold", fontsize=13)
        ax.set_ylabel(f"{ylabel} (mean $\\pm$ SD)", fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in methods]
    fig.legend(handles, [DISPLAY[m] for m in methods], loc="upper center", ncol=len(methods),
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def tradeoff(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, ds in zip(axes, ("shenzhen", "montgomery")):
        for m in [m for m in MAIN if m in set(df.method)]:
            (a, sa), (i, si) = _ms(df, m, ds, "testAUC"), _ms(df, m, ds, "camIoU")
            ax.errorbar(a, i, xerr=sa, yerr=si, fmt="o", color=COLORS[m], capsize=2, label=DISPLAY[m])
        ax.set_xlabel("Test AUC"); ax.set_ylabel("CAM-IoU"); ax.set_title(ds.capitalize(), fontweight="bold")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def slopes(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, ds in zip(axes, ("shenzhen", "montgomery")):
        for m in [m for m in MAIN if m in set(df.method)]:
            a, t = _ms(df, m, ds, "testAUC")[0], _ms(df, m, ds, "transferAUC")[0]
            ax.plot([0, 1], [a, t], "-o", color=COLORS[m], label=DISPLAY[m])
        ax.axhline(0.5, ls="--", color="k", lw=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["In-domain", "Transfer"])
        ax.set_title(f"{ds.capitalize()} -> {OTHER[ds].capitalize()}", fontweight="bold")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AUC"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def clamp_training(out_root, arch, ds, path):
    methods = [m for m in ("full", "sym_noclamp", "full_noclamp")
               if list((Path(out_root) / m / ds / arch).glob("seed_*_history.json"))]
    if not methods:
        return False
    panels = [("total", "Training loss"), ("anat", "Anatomical term (unweighted)"),
              ("R", "Inside-lung mean R"), ("P", "Outside-lung mean P")]
    fig, axes = plt.subplots(len(methods), 4, figsize=(16, 3.2 * len(methods)), squeeze=False)
    for r, m in enumerate(methods):
        for f in sorted((Path(out_root) / m / ds / arch).glob("seed_*_history.json")):
            h = pd.DataFrame(json.load(open(f))["history"])
            for c, (col, _) in enumerate(panels):
                axes[r, c].plot(h["epoch"], h[col], color=COLORS[m], alpha=0.6)
        for c, (_, title) in enumerate(panels):
            ax = axes[r, c]
            ax.axhline(0, ls="--", color="k", lw=1); ax.grid(alpha=0.3)
            ax.set_title(f"{DISPLAY[m]}: {title}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Epoch")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return True


def rho_sensitivity(df, path):
    pts = [(0.25, "anat_only_rho025"), (0.50, "anat_only_rho050"), (0.75, "anat_only"), (1.00, "anat_only_rho100")]
    pts = [(r, m) for r, m in pts if m in set(df.method)]
    if len(pts) < 3:
        return False
    cols = [("testAUC", "In-domain AUC"), ("transferAUC", "Transfer AUC"), ("energyRatio", "Energy Ratio"), ("camIoU", "CAM-IoU")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, (col, title) in zip(axes, cols):
        for ds, color in (("shenzhen", "#1f77b4"), ("montgomery", "#d62728")):
            ms = [_ms(df, m, ds, col) for _, m in pts]
            ax.errorbar([r for r, _ in pts], [a for a, _ in ms], yerr=[b for _, b in ms], fmt="-o",
                        capsize=3, color=color, label=f"train: {ds.capitalize()}")
        ax.set_xlabel(r"$\rho$"); ax.set_title(title, fontweight="bold"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return True


def _pm(df, m, ds, col):
    a, s = _ms(df, m, ds, col)
    return "--" if np.isnan(a) else f"${a:.3f} \\pm {s:.3f}$"


def tables(df, path):
    methods = [m for m in GROUPS["all"] if m in set(df.method)]
    lines = ["% Generated by agloss.figures. Mean +/- SD over seeds.\n"]
    for ds in ("shenzhen", "montgomery"):
        n = int(df[df.dataset == ds].groupby("method").seed.nunique().max())
        lines += [f"% ---- In-domain: {ds} (up to {n} seeds) ----",
                  "\\begin{tabular}{l c c c c}", "\\toprule",
                  "\\textbf{Method} & \\textbf{Test AUC} & \\textbf{Energy Ratio} & \\textbf{Pointing Game} & \\textbf{CAM-IoU} \\\\",
                  "\\midrule"]
        for m in methods:
            if not df[(df.method == m) & (df.dataset == ds)].empty:
                lines.append(f"{DISPLAY[m]} & " + " & ".join(_pm(df, m, ds, c) for c in
                             ("testAUC", "energyRatio", "pointingGame", "camIoU")) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", ""]
        lines += [f"% ---- Transfer: {ds} -> {OTHER[ds]} ----",
                  "\\begin{tabular}{l c c c c c}", "\\toprule",
                  "\\textbf{Method} & \\textbf{In-AUC} & \\textbf{Transfer AUC} & \\textbf{$\\Delta$AUC} & \\textbf{Transfer ER} & \\textbf{Transfer IoU} \\\\",
                  "\\midrule"]
        for m in methods:
            sub = df[(df.method == m) & (df.dataset == ds)]
            if sub.empty or sub.transferAUC.isna().all():
                continue
            d = (sub.transferAUC - sub.testAUC)
            lines.append(f"{DISPLAY[m]} & {_pm(df, m, ds, 'testAUC')} & {_pm(df, m, ds, 'transferAUC')} & "
                         f"${d.mean():+.3f} \\pm {d.std(ddof=1):.3f}$ & {_pm(df, m, ds, 'transferER')} & "
                         f"{_pm(df, m, ds, 'transferIoU')} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", ""]
    Path(path).write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--arch", default="resnet50")
    args = ap.parse_args(argv)
    df = load_all(args.out_root, args.arch)
    fig_dir = Path(args.out_root) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    bars(df, "shenzhen", fig_dir / "fig02_shenzhen_results.pdf")
    bars(df, "montgomery", fig_dir / "fig03_montgomery_results.pdf")
    tradeoff(df, fig_dir / "fig04_tradeoff.pdf")
    slopes(df, fig_dir / "fig05_transfer_slopes.pdf")
    clamp_training(args.out_root, args.arch, "montgomery", fig_dir / "fig07_clamp_training.pdf")
    rho_sensitivity(df, fig_dir / "fig08_rho_sensitivity.pdf")
    tables(df, fig_dir / "tables.tex")
    print(f"wrote figures and tables.tex to {fig_dir}")


if __name__ == "__main__":
    main()
