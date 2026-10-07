"""Command-line runner for the experiment grid.

Examples
--------
Time a single run first:
  python -m agloss.run --methods full --datasets shenzhen --seeds 42 \
      --shenzhen-root D:/data/Shenzhen --montgomery-root D:/data/Montgomery

Main grid (6 methods x 2 datasets x 8 seeds = 96 runs, transfer evaluated inline):
  python -m agloss.run --methods main --seeds 42-49 --shenzhen-root ... --montgomery-root ...

Extra experiments:
  python -m agloss.run --methods ablations masked maskperturb --seeds 42-49 ...

Runs already present in per_seed.csv are skipped, so the command can be re-run after an interruption.
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import torch

from .config import GROUPS, METHODS, expand_methods, make_config
from .train import already_done, train_single_seed

OTHER = {"shenzhen": "montgomery", "montgomery": "shenzhen"}


def parse_seeds(tokens):
    seeds = []
    for t in tokens:
        if "-" in t:
            a, b = t.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(t))
    return seeds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=["main"],
                    help=f"method names or groups {sorted(GROUPS)}; methods: {sorted(METHODS)}")
    ap.add_argument("--datasets", nargs="+", default=["shenzhen", "montgomery"], choices=["shenzhen", "montgomery"])
    ap.add_argument("--seeds", nargs="+", default=["42-49"], help="e.g. 42-49 or 42 43 44")
    ap.add_argument("--shenzhen-root", required=True)
    ap.add_argument("--montgomery-root", required=True)
    ap.add_argument("--out-root", default="results")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "densenet121", "convnext_tiny"])
    ap.add_argument("--no-transfer", action="store_true", help="skip evaluation on the other dataset")
    ap.add_argument("--no-checkpoint", action="store_true", help="do not save model weights (~95 MB per run)")
    ap.add_argument("--max-epochs", type=int, help="override (smoke tests only)")
    ap.add_argument("--tv-lambda", type=float, help="override the Tversky loss weight")
    ap.add_argument("--anat-lambda", type=float, help="override the anatomical loss weight")
    ap.add_argument("--mask-perturb-px", type=int, help="erosion/dilation radius for the maskperturb group")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    methods, seeds = expand_methods(args.methods), parse_seeds(args.seeds)
    overrides = dict(shenzhen_root=args.shenzhen_root, montgomery_root=args.montgomery_root,
                     out_root=args.out_root, cache_dir=args.cache_dir, arch=args.arch,
                     save_checkpoint=not args.no_checkpoint, verbose=not args.quiet)
    if args.max_epochs:
        overrides.update(max_epochs=args.max_epochs, min_epochs=min(25, args.max_epochs),
                         select_from_epoch=min(25, args.max_epochs))
    for k in ("tv_lambda", "anat_lambda", "mask_perturb_px"):
        if getattr(args, k) is not None:
            overrides[k] = getattr(args, k)

    total = len(methods) * len(args.datasets) * len(seeds)
    print(f"{total} runs: methods={methods} datasets={args.datasets} seeds={seeds}")
    data_cache, done, t0 = {}, 0, time.time()
    for ds in args.datasets:
        for method in methods:
            for seed in seeds:
                cfg = make_config(method, dataset=ds, seed=seed,
                                  transfer_dataset=None if args.no_transfer else OTHER[ds], **overrides)
                done += 1
                if already_done(cfg):
                    print(f"[{done}/{total}] skip (already in per_seed.csv): {method} {ds} seed {seed}")
                    continue
                print(f"[{done}/{total}] {method} {ds} seed {seed} | elapsed {(time.time() - t0) / 3600:.2f} h")
                try:
                    train_single_seed(cfg, device, data_cache)
                except Exception:                                   # keep the grid going
                    Path(args.out_root).mkdir(parents=True, exist_ok=True)
                    with open(Path(args.out_root) / "errors.log", "a") as f:
                        f.write(f"{method} {ds} seed {seed}\n{traceback.format_exc()}\n")
                    traceback.print_exc()
    print(f"finished in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
