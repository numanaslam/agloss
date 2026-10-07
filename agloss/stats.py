"""Summary tables and seed-paired statistics for the manuscript.

  python -m agloss.stats --out-root results

Writes results/summary.csv (mean and SD per method and condition),
results/paired_tests.csv (planned seed-paired comparisons) and prints both.

Planned comparisons, Holm-corrected within each (condition, metric) family:
  A. AG-only vs each other main configuration        (guidance question)
  B. Full vs Sym-AG, and Full vs Full (no clamp)     (clamp/asymmetry and clamp-only)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from .config import DISPLAY, GROUPS

CONDITIONS = [  # label, train dataset, columns for AUC / ER / PG / IoU
    ("In-domain Shenzhen", "shenzhen", dict(AUC="testAUC", ER="energyRatio", PG="pointingGame", IoU="camIoU")),
    ("In-domain Montgomery", "montgomery", dict(AUC="testAUC", ER="energyRatio", PG="pointingGame", IoU="camIoU")),
    ("Shenzhen -> Montgomery", "shenzhen", dict(AUC="transferAUC", ER="transferER", PG="transferPG", IoU="transferIoU")),
    ("Montgomery -> Shenzhen", "montgomery", dict(AUC="transferAUC", ER="transferER", PG="transferPG", IoU="transferIoU")),
]


def load_all(out_root: str, arch: str) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in Path(out_root).glob(f"*/*/{arch}/per_seed.csv")]
    if not frames:
        raise SystemExit(f"No per_seed.csv found under {out_root}/<method>/<dataset>/{arch}/")
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["method", "dataset", "seed"], keep="last")


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def paired(df, ds, col, a, b):
    da = df[(df.method == a) & (df.dataset == ds)].set_index("seed")[col]
    db = df[(df.method == b) & (df.dataset == ds)].set_index("seed")[col]
    seeds = da.index.intersection(db.index)
    d = (da.loc[seeds] - db.loc[seeds]).dropna().to_numpy()
    n = len(d)
    if n < 2:
        return None
    mean, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    tc = sps.t.ppf(0.975, n - 1)
    try:
        p = sps.wilcoxon(d, alternative="two-sided", method="exact").pvalue if np.any(d != 0) else 1.0
    except ValueError:                       # ties: fall back to the normal approximation
        p = sps.wilcoxon(d, alternative="two-sided").pvalue
    return dict(n=n, diff=mean, lo=mean - tc * se, hi=mean + tc * se, p=p)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--arch", default="resnet50")
    args = ap.parse_args(argv)
    df = load_all(args.out_root, args.arch)
    present = [m for m in GROUPS["all"] if m in set(df.method)]

    rows = []
    for label, ds, cols in CONDITIONS:
        for m in present:
            sub = df[(df.method == m) & (df.dataset == ds)]
            if sub.empty:
                continue
            r = dict(condition=label, method=DISPLAY.get(m, m), n=len(sub))
            for k, c in cols.items():
                r[f"{k}_mean"], r[f"{k}_sd"] = sub[c].mean(), sub[c].std(ddof=1)
            rows.append(r)
    summary = pd.DataFrame(rows)

    tests = []
    for label, ds, cols in CONDITIONS:
        for k, c in cols.items():
            fam_a = [("anat_only", m) for m in GROUPS["main"] if m != "anat_only"]
            fam_b = [("full", "sym_noclamp"), ("full", "full_noclamp")]
            for fam_name, fam in (("A: AG-only vs others", fam_a), ("B: clamp", fam_b)):
                res = [(a, b, paired(df, ds, c, a, b)) for a, b in fam]
                res = [(a, b, r) for a, b, r in res if r is not None]
                if not res:
                    continue
                padj = holm([r["p"] for _, _, r in res])
                for (a, b, r), q in zip(res, padj):
                    tests.append(dict(condition=label, metric=k, family=fam_name,
                                      comparison=f"{DISPLAY[a]} - {DISPLAY[b]}", pairs=r["n"],
                                      diff=r["diff"], ci_low=r["lo"], ci_high=r["hi"],
                                      wilcoxon_p=r["p"], holm_p=q))
    tests = pd.DataFrame(tests)

    out = Path(args.out_root)
    summary.to_csv(out / "summary.csv", index=False)
    tests.to_csv(out / "paired_tests.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", None, "display.max_columns", None,
                           "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
        print()
        print(tests.to_string(index=False))
    print(f"\nwrote {out / 'summary.csv'} and {out / 'paired_tests.csv'}")


if __name__ == "__main__":
    main()
