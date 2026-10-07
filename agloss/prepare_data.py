"""Download and arrange the two datasets.

  python -m agloss.prepare_data --data-root D:/data

Creates <data-root>/Shenzhen and <data-root>/Montgomery in the layout the loaders expect.
The chest X-rays come from the U.S. National Library of Medicine. The Shenzhen lung masks
(Stirenko et al.) are hosted on Kaggle and need either the `kaggle` command-line tool with
an API token, or a manual download of the zip into <data-root>.

If a download URL has moved, download the zip by hand into <data-root> and re-run:
the script picks up any zip already there.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

NLM = {
    "shenzhen": ("ChinaSet_AllFiles.zip", "https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip"),
    "montgomery": ("NLM-MontgomeryCXRSet.zip", "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"),
}
KAGGLE_MASKS = "yoctoman/shcxr-lung-mask"
MASK_ZIP_NAMES = ["shcxr-lung-mask.zip", "archive.zip"]


def _download(url: str, dest: Path):
    print(f"downloading {url}\n  -> {dest}")

    def hook(blocks, block_size, total):
        if total > 0 and blocks % 2000 == 0:
            print(f"  {min(100.0, 100.0 * blocks * block_size / total):5.1f}%", end="\r", flush=True)

    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp, hook)
    tmp.rename(dest)
    print("  done      ")


def _extract(zip_path: Path, into: Path):
    print(f"extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(into)


def _real_pngs(d: Path, pattern: str = "*.png"):
    return [p for p in d.glob(pattern) if p.is_file() and not p.name.startswith("._")]


def _best_dir(src_root: Path, wanted: str, pattern: str = "*.png"):
    """The folder called `wanted` (case-insensitive) that holds the most PNG files."""
    cands = [d for d in src_root.rglob("*") if d.is_dir() and d.name.lower() == wanted.lower()
             and "__MACOSX" not in d.parts]
    cands.sort(key=lambda d: len(_real_pngs(d, pattern)), reverse=True)
    return cands[0] if cands and _real_pngs(cands[0], pattern) else None


def _collect(src_root: Path, pattern: str, dest: Path, parent: str | None = None) -> int:
    """Copy matching files into dest. With `parent`, only from the folder of that name
    (the Montgomery mask folders reuse the image filenames, so this matters)."""
    if dest.is_dir():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    if parent is not None:
        src = _best_dir(src_root, parent, pattern)
        files = _real_pngs(src, pattern) if src else []
    else:
        files = [p for p in src_root.rglob(pattern) if p.is_file() and "__MACOSX" not in p.parts
                 and not p.name.startswith("._")]
    for p in files:
        shutil.copy2(p, dest / p.name)
    return len(files)


def _count(d: Path, pattern="*.png") -> int:
    return len(list(d.glob(pattern))) if d.is_dir() else 0


def prepare(data_root: str) -> tuple[str, str]:
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "_raw"
    sz, mo = root / "Shenzhen", root / "Montgomery"

    # ---- images (NLM) ------------------------------------------------------
    if _count(sz / "CXR_png", "CHNCXR_*.png") < 600:
        name, url = NLM["shenzhen"]
        if not (root / name).is_file():
            _download(url, root / name)
        _extract(root / name, raw / "shenzhen")
        n = _collect(raw / "shenzhen", "CHNCXR_*_[01].png", sz / "CXR_png", parent="CXR_png")
        print(f"Shenzhen: {n} images")
    marker = mo / ".prepared_v2"          # v1 mixed mask files into CXR_png; rebuild if absent
    if not marker.is_file():
        name, url = NLM["montgomery"]
        if not (raw / "montgomery").is_dir():
            if not (root / name).is_file():
                _download(url, root / name)
            _extract(root / name, raw / "montgomery")
        n = _collect(raw / "montgomery", "MCUCXR_*_[01].png", mo / "CXR_png", parent="CXR_png")
        nl = _collect(raw / "montgomery", "MCUCXR_*.png", mo / "ManualMask" / "leftMask", parent="leftMask")
        nr = _collect(raw / "montgomery", "MCUCXR_*.png", mo / "ManualMask" / "rightMask", parent="rightMask")
        print(f"Montgomery: {n} images, {nl} left masks, {nr} right masks")
        if not (n == nl == nr == 138):
            print("  folders found in the archive:")
            for d in sorted(p for p in (raw / "montgomery").rglob("*") if p.is_dir() and "__MACOSX" not in p.parts):
                print(f"    {d.relative_to(raw / 'montgomery')}  ({len(_real_pngs(d))} png)")
            sys.exit("Montgomery: expected 138 images, 138 left masks and 138 right masks. "
                     "Send the folder listing above so the script can be fixed.")
        marker.write_text("ok")

    # ---- Shenzhen lung masks (Kaggle) --------------------------------------
    if _count(sz / "mask") < 500:
        zips = [root / n for n in MASK_ZIP_NAMES if (root / n).is_file()]
        if not zips and shutil.which("kaggle"):
            print("downloading Shenzhen masks with the kaggle CLI ...")
            try:
                subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_MASKS, "-p", str(root)], check=False)
            except OSError as e:              # e.g. blocked by a Windows Application Control policy
                print(f"  kaggle CLI could not be started ({e}); falling back to manual download.")
            zips = [root / n for n in MASK_ZIP_NAMES if (root / n).is_file()]
        if not zips:                          # any zip in the folder that contains the masks
            for z in sorted(root.glob("*.zip")):
                try:
                    with zipfile.ZipFile(z) as zf:
                        if sum(n.endswith("_mask.png") and "CHNCXR_" in n for n in zf.namelist()) > 500:
                            zips = [z]
                            break
                except zipfile.BadZipFile:
                    pass
        if not zips:
            sys.exit("\nShenzhen lung masks are missing.\n"
                     f"  1. Download https://www.kaggle.com/datasets/{KAGGLE_MASKS} (free account)\n"
                     f"  2. Put the downloaded zip in {root} (any file name)\n"
                     "  3. Re-run this command.\n")
        _extract(zips[0], raw / "shenzhen_masks")
        n = _collect(raw / "shenzhen_masks", "CHNCXR_*_mask.png", sz / "mask")
        print(f"Shenzhen: {n} masks")

    print(f"\nShenzhen:   {_count(sz / 'CXR_png')} images, {_count(sz / 'mask')} masks   (expected 662 and 566)")
    print(f"Montgomery: {_count(mo / 'CXR_png')} images, {_count(mo / 'ManualMask' / 'leftMask')} left and "
          f"{_count(mo / 'ManualMask' / 'rightMask')} right masks   (expected 138 each)")
    return str(sz), str(mo)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    prepare(ap.parse_args(argv).data_root)


if __name__ == "__main__":
    main()
