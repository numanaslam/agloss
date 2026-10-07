"""Experiment configuration and the registry of method configurations."""
from __future__ import annotations

from dataclasses import dataclass, field, replace, asdict
from typing import Optional, Tuple


@dataclass
class Config:
    # identity
    method: str = "full"
    dataset: str = "shenzhen"            # 'shenzhen' | 'montgomery'
    transfer_dataset: Optional[str] = None
    arch: str = "resnet50"               # 'resnet50' | 'densenet121' | 'convnext_tiny'
    seed: int = 42

    # data
    shenzhen_root: str = ""
    montgomery_root: str = ""
    cache_dir: str = "cache"
    image_size: int = 224
    test_fraction: float = 0.20
    n_folds: int = 5                     # validation = 1 fold of the remaining 80%
    used_fold: int = 1
    augment: bool = True
    masked_input: bool = False           # baseline: multiply the input by the lung mask
    mask_perturb: str = "none"           # 'none' | 'erode' | 'dilate'  (training masks only)
    mask_perturb_px: int = 8

    # model
    pretrained: bool = True

    # loss switches
    use_tversky: bool = True
    use_anat: bool = True

    # focal loss
    focal_gamma: float = 2.0
    focal_alpha: Tuple[float, float] = (0.4, 0.6)   # (Normal, TB) class weights
    label_smoothing: float = 0.10

    # Tversky on Grad-CAM vs lung mask
    tv_alpha: float = 0.7
    tv_beta: float = 0.3
    tv_lambda: float = 0.5
    tv_eps: float = 1e-6

    # anatomical guidance
    rho: float = 0.75
    anat_lambda: float = 0.5
    positivity_clamp: bool = True
    outside_stat: str = "mean"           # 'mean' | 'topk' (mean of the top-k outside-lung values)
    topk_frac: float = 0.10
    cam_norm: str = "minmax"             # 'minmax' | 'mass' (saliency-mass formulation)
    mass_tau: float = 0.30               # 'mass': allowed fraction of saliency outside the lungs

    # curriculum s(e) applied to the anatomical term
    curr_warmup: int = 5
    curr_ramp: int = 4

    # training (harmonised Phase 3 protocol)
    max_epochs: int = 50
    min_epochs: int = 25
    patience: int = 15
    select_from_epoch: int = 25          # best checkpoint is chosen among epochs >= this
    batch_size: int = 16
    lr: float = 3e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    lr_warmup: int = 2

    # output
    out_root: str = "results"
    save_checkpoint: bool = True
    verbose: bool = True

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------------
# Method registry: each entry is a dict of overrides on top of Config().
# ----------------------------------------------------------------------------
METHODS = {
    # --- the six configurations in the paper ---
    "vanilla": dict(use_tversky=False, use_anat=False, focal_gamma=0.0,
                    focal_alpha=(0.5, 0.5), label_smoothing=0.0),
    "focal": dict(use_tversky=False, use_anat=False),
    "focal_tversky": dict(use_tversky=True, use_anat=False),
    "anat_only": dict(use_tversky=False, use_anat=True),
    "full": dict(use_tversky=True, use_anat=True),
    "sym_noclamp": dict(use_tversky=True, use_anat=True, positivity_clamp=False, rho=1.0),
    # --- clamp-only ablation: differs from 'full' in the clamp ONLY ---
    "full_noclamp": dict(use_tversky=True, use_anat=True, positivity_clamp=False),
    # --- rho sensitivity for clamped AG-only (rho = 0.75 is 'anat_only') ---
    "anat_only_rho025": dict(use_tversky=False, use_anat=True, rho=0.25),
    "anat_only_rho050": dict(use_tversky=False, use_anat=True, rho=0.50),
    "anat_only_rho100": dict(use_tversky=False, use_anat=True, rho=1.00),
    # --- loss variants: outside-lung hotspot penalty and saliency-mass normalisation ---
    "anat_only_topk": dict(use_tversky=False, use_anat=True, outside_stat="topk"),
    "anat_only_mass": dict(use_tversky=False, use_anat=True, cam_norm="mass"),
    # --- simple baseline: classifier sees only the lung region ---
    "masked_input": dict(use_tversky=False, use_anat=False, masked_input=True),
    # --- mask-convention test: train with systematically perturbed masks ---
    "focal_tversky_erode": dict(use_tversky=True, use_anat=False, mask_perturb="erode"),
    "focal_tversky_dilate": dict(use_tversky=True, use_anat=False, mask_perturb="dilate"),
    "anat_only_erode": dict(use_tversky=False, use_anat=True, mask_perturb="erode"),
    "anat_only_dilate": dict(use_tversky=False, use_anat=True, mask_perturb="dilate"),
}

GROUPS = {
    "main": ["vanilla", "focal", "focal_tversky", "sym_noclamp", "anat_only", "full"],
    "ablations": ["full_noclamp", "anat_only_rho025", "anat_only_rho050", "anat_only_rho100"],
    "masked": ["masked_input"],
    "maskperturb": ["focal_tversky_erode", "focal_tversky_dilate", "anat_only_erode", "anat_only_dilate"],
}
GROUPS["variants"] = ["anat_only_topk", "anat_only_mass"]
GROUPS["core4"] = ["focal", "focal_tversky", "anat_only", "full"]
GROUPS["all"] = GROUPS["main"] + GROUPS["ablations"] + GROUPS["masked"] + GROUPS["maskperturb"] + GROUPS["variants"]

DISPLAY = {
    "vanilla": "Vanilla", "focal": "Focal", "focal_tversky": "Focal+Tversky",
    "sym_noclamp": "Sym-AG", "anat_only": "AG-only", "full": "Full",
    "full_noclamp": "Full (no clamp)", "masked_input": "Masked input",
    "anat_only_rho025": "AG-only rho=0.25", "anat_only_rho050": "AG-only rho=0.50",
    "anat_only_rho100": "AG-only rho=1.00",
    "anat_only_topk": "AG-only (top-k outside)", "anat_only_mass": "AG-only (saliency mass)",
    "focal_tversky_erode": "Focal+Tversky (eroded masks)",
    "focal_tversky_dilate": "Focal+Tversky (dilated masks)",
    "anat_only_erode": "AG-only (eroded masks)", "anat_only_dilate": "AG-only (dilated masks)",
}


def make_config(method: str, **overrides) -> Config:
    if method not in METHODS:
        raise KeyError(f"Unknown method '{method}'. Known: {sorted(METHODS)}")
    return replace(Config(method=method), **METHODS[method], **overrides)


def expand_methods(names):
    out = []
    for n in names:
        out.extend(GROUPS[n] if n in GROUPS else [n])
    seen, uniq = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq
