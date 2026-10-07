"""Run everything with one command.

  python run_all.py --data-root D:/data

Steps (each is skipped or resumed if its outputs already exist):
  1. download and arrange the datasets            (agloss.prepare_data)
  2. unit tests                                   (pytest, about a minute)
  3. main grid: 6 methods x 2 datasets x 8 seeds  (agloss.run)
  4. extra experiments: clamp-only, rho sweep, masked input, mask perturbation
  5. statistics, figures, tables, Grad-CAM grid, occlusion explainer

Useful options:
  --quick          3-epoch run of one seed per method, to check the whole pipeline end to end (about 15 min)
  --skip-extras    stop after the main grid and its analysis
  --seeds 42-45    use fewer seeds
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
    """Analysis steps must not stop the pipeline; report and carry on."""
    try:
        fn()
    except BaseException as e:                       # includes SystemExit from a sub-command
        print(f"!! {name} failed: {type(e).__name__}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="folder that holds (or will hold) Shenzhen/ and Montgomery/")
    ap.add_argument("--out-root", default=str(HERE / "results"))
    ap.add_argument("--seeds", nargs="+", default=["42-49"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-extras", action="store_true")
    ap.add_argument("--skip-occlusion", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    import torch
    assert torch.cuda.is_available() or args.quick, \
        "No CUDA GPU visible to PyTorch. Install the CUDA build (see README) or use --quick to test on CPU."

    step("1/5  Datasets")
    from agloss.prepare_data import prepare
    sz, mo = prepare(args.data_root)

    if not args.skip_tests:
        step("2/5  Unit tests")
        if subprocess.run([sys.executable, "-m", "pytest", "-q", str(HERE / "tests")], cwd=HERE).returncode != 0:
            sys.exit("Tests failed. Fix this before training (or re-run with --skip-tests).")

    out_root = args.out_root + ("_quick" if args.quick else "")
    cache = str(HERE / "cache")
    common = ["--shenzhen-root", sz, "--montgomery-root", mo, "--out-root", out_root, "--cache-dir", cache]
    seeds = ["42"] if args.quick else args.seeds
    train_extra = ["--max-epochs", "3"] if args.quick else []

    from agloss import run
    step("3/5  Main grid")
    run.main(["--methods", "main", "--seeds", *seeds, *common, *train_extra])
    if not args.skip_extras:
        step("4/5  Extra experiments")
        run.main(["--methods", "ablations", "masked", "maskperturb", "--seeds", *seeds, *common, *train_extra])

    step("5/5  Analysis")
    from agloss import cam_grid, figures, occlusion, stats
    data_args = ["--shenzhen-root", sz, "--montgomery-root", mo, "--cache-dir", cache]
    guarded("stats", lambda: stats.main(["--out-root", out_root]))
    guarded("figures", lambda: figures.main(["--out-root", out_root]))
    guarded("cam grid", lambda: cam_grid.main(["--out-root", out_root, *data_args, "--seed", "42", "--output",
                                               str(Path(out_root) / "figures" / "fig06_cam_grids.pdf")]))
    if not args.skip_occlusion:
        guarded("occlusion", lambda: occlusion.main(["--out-root", out_root, *data_args]))

    err = Path(out_root) / "errors.log"
    print(f"\nAll done in {(time.time() - t0) / 3600:.2f} h. Results: {out_root}")
    if err.is_file():
        print(f"!! Some runs failed; see {err}")


if __name__ == "__main__":
    main()
