"""CPU checks for agloss.robustness (no dataset needed)."""
import numpy as np
import torch

from agloss.model import CamNet
from agloss.robustness import EARLY_SPLIT, control_region, dhash, forward_split, inpaint, unit_vectors


def _lung_mask(b=3, size=64):
    m = torch.zeros(b, 1, size, size)
    m[:, :, 16:48, 8:28] = 1          # left lung
    m[:, :, 16:48, 36:56] = 1         # right lung
    return m


def test_inpaint_keeps_known_pixels_and_stays_in_range():
    torch.manual_seed(0)
    x = torch.randn(3, 3, 64, 64)
    m = _lung_mask()
    y = inpaint(x, m)
    assert torch.isfinite(y).all()
    assert torch.equal(y * (1 - m), x * (1 - m))                       # outside the hole: unchanged
    lo, hi = x.amin(dim=(2, 3), keepdim=True), x.amax(dim=(2, 3), keepdim=True)
    assert ((y >= lo - 1e-5) & (y <= hi + 1e-5)).all()                  # a smooth fill, no new extremes


def test_control_region_is_outside_and_matches_lung_area():
    m = _lung_mask()
    ctrl = control_region(m, torch.Generator().manual_seed(0), block=8)
    assert (ctrl * m).sum() == 0                                          # never touches the lungs
    lung, got = m.flatten(1).sum(1), ctrl.flatten(1).sum(1)
    assert ((got >= 0.9 * lung) & (got <= lung + 8 * 8)).all()           # about the same area


def test_forward_split_matches_forward():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for arch, k in EARLY_SPLIT.items():
        model = CamNet(arch, pretrained=False).eval()
        with torch.no_grad():
            logits, feats = model(x)
            logits2, feats2, early = forward_split(model, x, k)
        assert torch.allclose(logits, logits2, atol=1e-5) and torch.allclose(feats, feats2, atol=1e-5)
        assert early.shape[-1] == 2 * feats.shape[-1]                     # one stage earlier: twice the resolution


def test_dhash_flags_duplicates_only():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    imgs = np.stack([a, a.copy(), a[:, ::-1].copy(), rng.integers(0, 255, size=(64, 64), dtype=np.uint8)])
    h = dhash(imgs)
    assert (h[0] != h[1]).sum() == 0 and (h[0] != h[2]).sum() > 4 and (h[0] != h[3]).sum() > 4


def test_roc_auc_matches_hand_computed_values():
    from agloss.metrics import roc_auc_score
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75          # textbook example
    assert roc_auc_score([0, 1], [0.2, 0.9]) == 1.0 and roc_auc_score([1, 0], [0.2, 0.9]) == 0.0
    assert roc_auc_score([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5            # all tied
    assert roc_auc_score([0, 1, 1], [0.3, 0.3, 0.9]) == 0.75                    # one tie across classes


def test_unit_vectors_correlation_separates_copies_from_other_images():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(128, 128)).astype(np.float32)
    imgs = torch.from_numpy(np.stack([base, base + 3.0, base[::-1].copy(), rng.integers(0, 255, size=(128, 128)).astype(np.float32)]))
    v = unit_vectors(imgs, 64)
    c = v @ v.T
    assert c[0, 1] > 0.999 and c[0, 2] < 0.5 and c[0, 3] < 0.5                 # brightness shift = copy
