"""Dataset scanning, caching, seed-deterministic splits, and GPU mini-batches.

Expected on-disk layout:

  Shenzhen:    <root>/CXR_png/CHNCXR_NNNN_C.png      C: 0 = Normal, 1 = TB
               <root>/mask/CHNCXR_NNNN_C_mask.png    (Stirenko et al. masks)

  Montgomery:  <root>/CXR_png/MCUCXR_NNNN_C.png
               <root>/mask/*.png                     pre-combined masks, OR
               <root>/ManualMask/leftMask/*.png + rightMask/*.png  (merged here)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------
# Scanning
# ----------------------------------------------------------------------------
def _first_existing(mask_dir: Path, candidates, mask_names, id_key: str):
    for c in candidates:
        if (mask_dir / c).is_file():
            return mask_dir / c
    hits = sorted(n for n in mask_names if id_key in n)
    return mask_dir / hits[0] if hits else None


def scan_shenzhen(root: str):
    root = Path(root)
    img_dir, mask_dir = root / "CXR_png", root / "mask"
    assert img_dir.is_dir(), f"Missing image dir: {img_dir}"
    assert mask_dir.is_dir(), f"Missing mask dir: {mask_dir}"
    mask_names = [p.name for p in mask_dir.glob("*.png")]
    items = []
    for p in sorted(img_dir.glob("CHNCXR_*.png")):
        m = re.fullmatch(r"CHNCXR_(\d+)_([01])\.png", p.name)
        assert m, f"Unexpected filename: {p.name}"
        i, c = m.group(1), m.group(2)
        cands = [f"CHNCXR_{i}_{c}_mask.png", f"CHNCXR_{i}_{c}.png", f"mask_CHNCXR_{i}_{c}.png",
                 f"CHNCXR_{i}_{c}_msk.png", f"CHNCXR_{i}_mask.png", f"CHNCXR_{i}.png"]
        mp = _first_existing(mask_dir, cands, mask_names, f"CHNCXR_{i}")
        if mp is not None:                       # images without a mask are dropped
            items.append((str(p), str(mp), int(c)))   # label: 0 = Normal, 1 = TB
    assert items, f"No Shenzhen image could be matched to a mask in {mask_dir}"
    return items


def scan_montgomery(root: str):
    root = Path(root)
    img_dir = root / "CXR_png"
    assert img_dir.is_dir(), f"Missing image dir: {img_dir}"
    combined = root / "mask"
    left, right = root / "ManualMask" / "leftMask", root / "ManualMask" / "rightMask"
    if combined.is_dir():
        layout, mask_names = "combined", [p.name for p in combined.glob("*.png")]
    elif left.is_dir() and right.is_dir():
        layout, mask_names = "split", []
    else:
        raise FileNotFoundError(f"Montgomery masks not found under {root} (mask/ or ManualMask/)")
    items = []
    for p in sorted(img_dir.glob("MCUCXR_*.png")):
        m = re.fullmatch(r"MCUCXR_(\d+)_([01])\.png", p.name)
        assert m, f"Unexpected filename: {p.name}"
        i, c = m.group(1), m.group(2)
        if layout == "combined":
            cands = [f"MCUCXR_{i}_{c}.png", f"MCUCXR_{i}_{c}_mask.png", f"MCUCXR_{i}.png", f"MCUCXR_{i}_mask.png"]
            mp = _first_existing(combined, cands, mask_names, f"MCUCXR_{i}")
            if mp is not None:
                items.append((str(p), str(mp), int(c)))
        else:
            lp, rp = left / p.name, right / p.name
            if lp.is_file() and rp.is_file():
                items.append((str(p), f"{lp}|{rp}", int(c)))   # merged at load time
    assert items, "No Montgomery image could be matched to a mask"
    return items


# ----------------------------------------------------------------------------
# Loading + caching (images are resized once and kept in memory)
# ----------------------------------------------------------------------------
def _read_image(path: str, size: int) -> np.ndarray:
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.dtype != np.uint8:                      # 16-bit PNGs -> 8-bit
        arr = (arr.astype(np.float64) / np.iinfo(arr.dtype).max * 255.0).round().astype(np.uint8)
        img = Image.fromarray(arr)
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)      # 3 x H x W


def _read_mask_one(path: str) -> Image.Image:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return Image.fromarray((arr > 0).astype(np.uint8) * 255)


def _read_mask(path: str, size: int) -> np.ndarray:
    if "|" in path:                                # Montgomery left|right
        lp, rp = path.split("|")
        lm, rm = _read_mask_one(lp), _read_mask_one(rp)
        if rm.size != lm.size:
            rm = rm.resize(lm.size, Image.NEAREST)
        m = Image.fromarray(np.maximum(np.asarray(lm), np.asarray(rm)))
    else:
        m = _read_mask_one(path)
    m = m.resize((size, size), Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.uint8)                   # H x W in {0,1}


@dataclass
class Arrays:
    name: str
    images: np.ndarray      # N x 3 x H x W uint8
    masks: np.ndarray       # N x H x W uint8 {0,1}
    labels: np.ndarray      # N int64, 0 = Normal, 1 = TB
    names: np.ndarray       # N str (image file stems)


def load_arrays(name: str, root: str, size: int, cache_dir: str, verbose: bool = True) -> Arrays:
    assert root, f"Data root for '{name}' is empty. Pass --{name}-root."
    os.makedirs(cache_dir, exist_ok=True)
    cache = Path(cache_dir) / f"{name}_{size}.npz"
    if cache.is_file():
        z = np.load(cache, allow_pickle=False)
        arr = Arrays(name, z["images"], z["masks"], z["labels"], z["names"])
    else:
        items = scan_shenzhen(root) if name == "shenzhen" else scan_montgomery(root)
        if verbose:
            print(f"[{name}] reading and resizing {len(items)} images (first run only) ...", flush=True)
        images = np.stack([_read_image(p, size) for p, _, _ in items])
        masks = np.stack([_read_mask(m, size) for _, m, _ in items])
        labels = np.array([c for _, _, c in items], dtype=np.int64)
        names = np.array([Path(p).stem for p, _, _ in items])
        np.savez_compressed(cache, images=images, masks=masks, labels=labels, names=names)
        arr = Arrays(name, images, masks, labels, names)
    if verbose:
        n, tb = len(arr.labels), int(arr.labels.sum())
        print(f"[{name}] {n} masked images: {n - tb} Normal / {tb} TB ({100.0 * tb / n:.1f}% TB)", flush=True)
    return arr


# ----------------------------------------------------------------------------
# Seed-deterministic stratified split: 80/20 train/test, then one of five folds as validation
# ----------------------------------------------------------------------------
def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def make_split(labels: np.ndarray, seed: int, test_fraction: float = 0.2, n_folds: int = 5, used_fold: int = 1):
    """80/20 stratified train-pool/test split, then one stratified fold of the pool as validation."""
    rng = np.random.RandomState(seed)
    pool, test = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        idx = idx[rng.permutation(len(idx))]
        cut = _round_half_up((1.0 - test_fraction) * len(idx))
        pool.append(idx[:cut])
        test.append(idx[cut:])
    pool, test = np.concatenate(pool), np.concatenate(test)
    pool, test = pool[rng.permutation(len(pool))], test[rng.permutation(len(test))]

    fold = np.zeros(len(pool), dtype=np.int64)
    pool_labels = labels[pool]
    for c in np.unique(pool_labels):
        idx = np.flatnonzero(pool_labels == c)
        idx = idx[rng.permutation(len(idx))]
        fold[idx] = np.arange(len(idx)) % n_folds + 1
    val, train = pool[fold == used_fold], pool[fold != used_fold]
    assert len(set(train) & set(val)) == 0 and len(set(pool) & set(test)) == 0
    return train, val, test


# ----------------------------------------------------------------------------
# GPU tensors and mini-batches
# ----------------------------------------------------------------------------
class GpuData:
    """Holds one dataset on the device and builds (augmented) mini-batches."""

    def __init__(self, arrays: Arrays, device):
        self.arrays = arrays
        self.device = device
        self.images = torch.from_numpy(arrays.images).to(device)                 # uint8 N x 3 x H x W
        self.masks = torch.from_numpy(arrays.masks).to(device).unsqueeze(1)      # uint8 N x 1 x H x W
        self.labels = torch.from_numpy(arrays.labels).to(device)
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def batch(self, idx, augment: bool, generator=None, masked_input: bool = False):
        idx = torch.as_tensor(idx, device=self.device, dtype=torch.long)
        x = self.images[idx].float() / 255.0
        m = self.masks[idx].float()
        y = self.labels[idx]
        if augment:
            x, m = _augment(x, m, generator)
        x = (x - self.mean) / self.std
        m = (m > 0.5).float()
        if masked_input:
            x = x * m                      # outside the lungs -> 0 (= ImageNet mean intensity)
        return x, y, m


def _augment(x, m, generator):
    """Horizontal flip (p = 0.5), translation of up to +/-10 px, scale of +/-5%."""
    b, _, h, w = x.shape
    dev = x.device
    r = torch.rand(b, 4, device=dev, generator=generator)
    flip = r[:, 0] < 0.5
    x = torch.where(flip.view(b, 1, 1, 1), x.flip(-1), x)
    m = torch.where(flip.view(b, 1, 1, 1), m.flip(-1), m)
    tx = (torch.floor(r[:, 1] * 21) - 10) * 2.0 / w        # integer pixels in [-10, 10], normalised
    ty = (torch.floor(r[:, 2] * 21) - 10) * 2.0 / h
    sc = 1.0 + (r[:, 3] - 0.5) * 0.10
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = 1.0 / sc
    theta[:, 1, 1] = 1.0 / sc
    theta[:, 0, 2] = -tx / sc
    theta[:, 1, 2] = -ty / sc
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    m = F.grid_sample(m, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return x, m


def perturb_masks(m: torch.Tensor, mode: str, px: int) -> torch.Tensor:
    """Systematically erode / dilate binary masks by `px` pixels (training-time only)."""
    if mode == "none" or px <= 0:
        return m
    k = 2 * px + 1
    if mode == "dilate":
        return F.max_pool2d(m, k, stride=1, padding=px)
    if mode == "erode":
        return 1.0 - F.max_pool2d(1.0 - m, k, stride=1, padding=px)
    raise ValueError(f"Unknown mask_perturb mode: {mode}")
