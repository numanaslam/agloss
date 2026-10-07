"""Summaries of the per-image analyses written by agloss.evalsuite.

  python -m agloss.evalstats --out-root results --arch resnet50

Writes into results/eval/ (one set per architecture):
  clinical_<arch>.csv      AUC, AUPRC, sensitivity at 90% specificity, specificity at 90% sensitivity, Brier
  byclass_<arch>.csv       Grad-CAM alignment separately for TB / Normal and correct / incorrect predictions
  reliance_<arch>.csv      region-ablation AUCs, occlusion Energy Ratio at three patch sizes,
                           deletion / insertion AUC (TB images), each as mean and SD over seeds
  bootstrap_<arch>.csv     image-level bootstrap 95% CIs: AUC per method, and paired AUC differences
                           (AG-only vs each other method; Full vs Full (no clamp))
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_curve

from .config import DISPLAY, GROUPS

ORDER = GROUPS["all"]
N_BOOT = 2000
MARGIN = 0.05           # non-inferiority margin for AUC (stated in the paper as post hoc)


def auc(y, p):
    """Fast AUC from ranks (ties handled); NaN if only one class is present."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def clinical(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return dict(AUC=auc(y, p), AUPRC=average_precision_score(y, p),
                sens_at_spec90=float(tpr[fpr <= 0.10].max()),
                spec_at_sens90=float(1.0 - fpr[tpr >= 0.90].min()),
                Brier=float(np.mean((p - y) ** 2)))


def cond_label(dataset, split):
    other = "montgomery" if dataset == "shenzhen" else "shenzhen"
    return f"In-domain {dataset.capitalize()}" if split == "test" else f"{dataset.capitalize()} -> {other.capitalize()}"


def mean_sd(df, keys, cols):
    """Per-seed values first, then mean and SD over seeds."""
    per_seed = df.groupby(keys + ["seed"])[cols].mean(numeric_only=True).reset_index()
    g = per_seed.groupby(keys)[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).join(g.size().rename("seeds"))
    return out.reset_index()


def bootstrap(df, rng):
    """Image-level bootstrap. Transfer: the target set is the same for every seed and method, so
    images are resampled once per replicate and shared (paired). In-domain: each seed has its own
    test split, resampled independently; methods stay paired within a seed."""
    rows = []
    for (dataset, split), d in df.groupby(["dataset", "split"]):
        methods = [m for m in ORDER if m in set(d.method)]
        seeds = sorted(d.seed.unique())
        tab = {}
        for (m, s), g in d.groupby(["method", "seed"]):
            g = g.sort_values("name")
            tab[(m, s)] = (g.label.to_numpy(), g.p_tb.to_numpy())
        n = {s: len(tab[(methods[0], s)][0]) for s in seeds if (methods[0], s) in tab}
        point = {m: np.nanmean([auc(*tab[(m, s)]) for s in seeds if (m, s) in tab]) for m in methods}
        boot = {m: np.empty(N_BOOT) for m in methods}
        for b in range(N_BOOT):
            shared = rng.integers(0, next(iter(n.values())), next(iter(n.values()))) if split == "transfer" else None
            acc = {m: [] for m in methods}
            for s in seeds:
                if s not in n:
                    continue
                idx = shared if shared is not None else rng.integers(0, n[s], n[s])
                for m in methods:
                    if (m, s) in tab and len(tab[(m, s)][0]) == n[s]:
                        y, p = tab[(m, s)]
                        acc[m].append(auc(y[idx], p[idx]))
            for m in methods:
                boot[m][b] = np.nanmean(acc[m]) if acc[m] else np.nan
        cond = cond_label(dataset, split)
        for m in methods:
            lo, hi = np.nanpercentile(boot[m], [2.5, 97.5])
            rows.append(dict(condition=cond, quantity="AUC", item=DISPLAY.get(m, m), estimate=point[m], ci_low=lo, ci_high=hi))
        pairs = [("anat_only", m) for m in methods if m != "anat_only"] + [("full", "full_noclamp")]
        for a, c in pairs:
            if a in boot and c in boot:
                diff = boot[a] - boot[c]
                lo, hi = np.nanpercentile(diff, [2.5, 97.5])
                rows.append(dict(condition=cond, quantity="AUC difference", item=f"{DISPLAY.get(a, a)} - {DISPLAY.get(c, c)}",
                                 estimate=point[a] - point[c], ci_low=lo, ci_high=hi,
                                 noninferior_at_margin=bool(lo > -MARGIN)))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--arch", default="resnet50")
    args = ap.parse_args(argv)
    ev = Path(args.out_root) / "eval"
    df = pd.read_csv(ev / f"per_image_{args.arch}.csv")
    df = df.drop_duplicates(subset=["method", "dataset", "seed", "split", "name"], keep="last")
    df["condition"] = [cond_label(d, s) for d, s in zip(df["dataset"], df["split"])]
    df["method_name"] = df.method.map(lambda m: DISPLAY.get(m, m))
    df["rank"] = df.method.map({m: i for i, m in enumerate(ORDER)})
    keys = ["condition", "rank", "method_name"]

    # 1. clinical metrics (per seed, then mean/SD)
    rows = []
    for (cond, rk, name, seed), g in df.groupby(keys + ["seed"]):
        rows.append(dict(condition=cond, rank=rk, method_name=name, seed=seed, **clinical(g.label.to_numpy(), g.p_tb.to_numpy())))
    c = pd.DataFrame(rows)
    cols = ["AUC", "AUPRC", "sens_at_spec90", "spec_at_sens90", "Brier"]
    g = c.groupby(keys)[cols]
    clin = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).reset_index().drop(columns="rank")
    clin.to_csv(ev / f"clinical_{args.arch}.csv", index=False)

    # 2. alignment by class and by correctness
    df["class"] = np.where(df.label == 1, "TB", "Normal")
    df["correct"] = np.where((df.p_tb >= 0.5) == (df.label == 1), "correct", "incorrect")
    parts = []
    for by in ("class", "correct"):
        t = mean_sd(df, keys + [by], ["er", "pg", "iou"]).rename(columns={by: "subset"})
        parts.append(t)
    byclass = pd.concat(parts).sort_values(["condition", "rank", "subset"]).drop(columns="rank")
    byclass.to_csv(ev / f"byclass_{args.arch}.csv", index=False)

    # 3. reliance: region ablation (AUC per seed), occlusion ER, deletion / insertion on TB images
    rows = []
    for (cond, rk, name, seed), g in df.groupby(keys + ["seed"]):
        y = g.label.to_numpy()
        r = dict(condition=cond, rank=rk, method_name=name, seed=seed, auc_full=auc(y, g.p_tb.to_numpy()))
        for col in ("p_lungs_zero", "p_outside_zero", "p_lungs_blur", "p_outside_blur"):
            r["auc" + col[1:]] = auc(y, g[col].to_numpy())
        for col in ("occ_er_16", "occ_er_32", "occ_er_64"):
            r[col] = g[col].mean()
        tb = g[g.label == 1]
        for col in ("del_cam", "del_rand", "ins_cam", "ins_rand"):
            r[col] = tb[col].mean()
        r["del_gain"] = r["del_rand"] - r["del_cam"]          # > 0: Grad-CAM ranks decisive pixels better than chance
        r["ins_gain"] = r["ins_cam"] - r["ins_rand"]
        rows.append(r)
    r = pd.DataFrame(rows)
    cols = [c_ for c_ in r.columns if c_ not in keys + ["seed"]]
    g = r.groupby(keys)[cols]
    rel = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).reset_index().drop(columns="rank")
    rel = rel.dropna(axis=1, how="all")
    rel.to_csv(ev / f"reliance_{args.arch}.csv", index=False)

    # 4. bootstrap
    boot = bootstrap(df, np.random.default_rng(0))
    boot.to_csv(ev / f"bootstrap_{args.arch}.csv", index=False)

    with pd.option_context("display.width", 260, "display.max_rows", None, "display.max_columns", None,
                           "display.float_format", "{:.3f}".format):
        for title, t in (("CLINICAL METRICS", clin), ("ALIGNMENT BY SUBSET", byclass), ("RELIANCE", rel), ("BOOTSTRAP", boot)):
            print(f"\n===== {title} =====")
            print(t.to_string(index=False))
    print(f"\nwrote clinical/byclass/reliance/bootstrap CSVs for {args.arch} to {ev}")


if __name__ == "__main__":
    main()
