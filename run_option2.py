"""Second round of experiments, in one command (resumable):

  python run_option2.py --data-root C:\\data

  1. loss variants on ResNet-50: AG-only with a top-k outside penalty, AG-only on saliency mass
  2. second and third backbones (DenseNet-121, ConvNeXt-Tiny): Focal, Focal+Tversky, AG-only, Full
  3. evaluation-only analyses on the saved checkpoints of every backbone (agloss.evalsuite)
  4. summaries (agloss.stats, agloss.evalstats) for every backbone

Options: --quick (3 epochs, one seed, to test the pipeline), --skip-backbones, --skip-eval,
         --archs densenet121 (choose backbones), --seeds 42-45
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def step(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def guarded(name, fn):
    try:
        fn()
    except BaseException as e:
        print(f"!! {name} failed: {type(e).__name__}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", default=str(HERE / "results"))
    ap.add_argument("--seeds", nargs="+", default=["42-49"])
    ap.add_argument("--archs", nargs="+", default=["densenet121", "convnext_tiny"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-backbones", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    from agloss.prepare_data import prepare
    sz, mo = prepare(args.data_root)
    if not args.skip_tests:
        step("Unit tests")
        if subprocess.run([sys.executable, "-m", "pytest", "-q", str(HERE / "tests")], cwd=HERE).returncode != 0:
            sys.exit("Tests failed. Send the output; do not start training.")

    out_root = args.out_root + ("_quick" if args.quick else "")
    cache = str(HERE / "cache")
    data_args = ["--shenzhen-root", sz, "--montgomery-root", mo, "--cache-dir", cache]
    common = [*data_args, "--out-root", out_root]
    seeds = ["42"] if args.quick else args.seeds
    extra = ["--max-epochs", "3"] if args.quick else []
    archs = [] if args.skip_backbones else args.archs

    from agloss import evalstats, evalsuite, run, stats
    step("1/4  Loss variants (ResNet-50)")
    run.main(["--methods", "variants", "--seeds", *seeds, *common, *extra])
    for arch in archs:
        step(f"2/4  Backbone: {arch}")
        run.main(["--methods", "core4", "--arch", arch, "--seeds", *seeds, *common, *extra])

    if not args.skip_eval:
        for arch in ["resnet50", *archs]:
            step(f"3/4  Evaluation-only analyses: {arch}")
            guarded(f"evalsuite {arch}", lambda a=arch: evalsuite.main(["--out-root", out_root, *data_args, "--arch", a]))

    step("4/4  Summaries")
    for arch in ["resnet50", *archs]:
        guarded(f"stats {arch}", lambda a=arch: stats.main(["--out-root", out_root, "--arch", a]))
        if arch != "resnet50":            # keep the per-backbone summaries apart
            for f in ("summary.csv", "paired_tests.csv"):
                src = Path(out_root) / f
                if src.is_file():
                    src.replace(Path(out_root) / f.replace(".csv", f"_{arch}.csv"))
        if not args.skip_eval:
            guarded(f"evalstats {arch}", lambda a=arch: evalstats.main(["--out-root", out_root, "--arch", a]))
    guarded("stats resnet50", lambda: stats.main(["--out-root", out_root, "--arch", "resnet50"]))

    print(f"\nAll done in {(time.time() - t0) / 3600:.2f} h. Results: {out_root}")
    if (Path(out_root) / "errors.log").is_file():
        print(f"!! Some runs failed; see {Path(out_root) / 'errors.log'}")


if __name__ == "__main__":
    main()
