"""CPU-only checks. Run with:  pytest -q   (takes about a minute; needs no dataset)."""
import numpy as np
import torch
from PIL import Image

from agloss.config import GROUPS, METHODS, make_config
from agloss.data import _augment, make_split, perturb_masks
from agloss.losses import anatomical_guidance_loss, focal_loss, outside_topk_mean, saliency_mass_loss, tversky_loss
from agloss.metrics import alignment_metrics, classification_metrics
from agloss.model import CamNet, grad_cam
from agloss.train import curriculum, learning_rate, train_single_seed


def _cam_mask():
    mask = torch.zeros(2, 1, 8, 8)
    mask[:, :, 2:6, 2:6] = 1
    return mask


def test_method_registry():
    assert len(GROUPS["main"]) == 6 and all(m in METHODS for m in GROUPS["all"])
    full, sym, noclamp = make_config("full"), make_config("sym_noclamp"), make_config("full_noclamp")
    assert full.use_tversky and full.use_anat and full.positivity_clamp and full.rho == 0.75
    assert sym.use_tversky and not sym.positivity_clamp and sym.rho == 1.0      # Sym-AG uses Tversky
    assert noclamp.rho == 0.75 and not noclamp.positivity_clamp                  # clamp is the only change
    assert make_config("vanilla").focal_gamma == 0.0


def test_ag_loss_values():
    mask = _cam_mask()
    inside = mask.clone()                                 # CAM == mask: R = 1, P = 0
    loss, r, p = anatomical_guidance_loss(inside, mask, 0.75, True)
    assert loss.item() == 0.0 and abs(r.item() - 1) < 1e-6 and p.item() == 0.0
    loss, _, _ = anatomical_guidance_loss(inside, mask, 0.75, False)
    assert abs(loss.item() + 0.75) < 1e-6                 # unclamped: negative, minimised at CAM == mask
    uniform = torch.full_like(mask, 0.4)                  # uniform map: P - rho R = 0.25 c > 0
    loss, _, _ = anatomical_guidance_loss(uniform, mask, 0.75, True)
    assert abs(loss.item() - 0.1) < 1e-6
    outside = 1 - mask                                    # CAM entirely outside: P = 1, R = 0
    loss, _, _ = anatomical_guidance_loss(outside, mask, 0.75, True)
    assert abs(loss.item() - 1.0) < 1e-6


def test_clamp_stops_gradient_when_satisfied():
    mask = _cam_mask()
    cam = (0.9 * mask + 0.1 * (1 - mask)).requires_grad_(True)       # P = 0.1 <= 0.75 * 0.9
    loss, _, _ = anatomical_guidance_loss(cam, mask, 0.75, True)
    loss.backward()
    assert cam.grad.abs().sum().item() == 0.0
    cam2 = cam.detach().clone().requires_grad_(True)
    loss2, _, _ = anatomical_guidance_loss(cam2, mask, 0.75, False)
    loss2.backward()
    assert cam2.grad.abs().sum().item() > 0.0             # unclamped term never switches off


def test_loss_variants():
    mask = _cam_mask()                                    # 16 inside pixels, 48 outside
    cam = 0.8 * mask
    cam[:, :, 0, 0] = 1.0                                 # one intense hotspot outside the lungs
    r, p = 0.8, 1.0 / 48
    plain, _, _ = anatomical_guidance_loss(cam, mask, 0.75, True)
    assert plain.item() == 0.0                            # the outside MEAN hides the hotspot
    assert abs(outside_topk_mean(cam, mask, 0.02)[0].item() - 1.0) < 1e-6
    topk, _, _ = anatomical_guidance_loss(cam, mask, 0.75, True, "topk", 0.02)
    assert abs(topk.item() - (1.0 - 0.75 * r)) < 1e-6     # the top-k variant penalises it
    loss, inside, outside = saliency_mass_loss(mask.clone(), mask, 0.3)
    assert loss.item() == 0.0 and abs(inside.item() - 1.0) < 1e-6
    loss, _, outside = saliency_mass_loss(torch.ones_like(mask), mask, 0.3)
    assert abs(outside.item() - 0.75) < 1e-6 and abs(loss.item() - 0.45) < 1e-6


def test_convnext_grad_cam():
    torch.manual_seed(0)
    model = CamNet("convnext_tiny", pretrained=False)
    logits, feats = model(torch.randn(2, 3, 64, 64))
    cam, raw = grad_cam(logits, feats, (64, 64), create_graph=True, return_raw=True)
    assert cam.shape == (2, 1, 64, 64) and raw.min() >= 0
    cam.sum().backward()


def test_tversky_and_focal():
    mask = _cam_mask()
    assert tversky_loss(mask.clone(), mask).item() < 1e-5
    assert tversky_loss(1 - mask, mask).item() > 0.99
    logits = torch.tensor([[2.0, -1.0], [0.3, 0.8]])
    y = torch.tensor([0, 1])
    ce = torch.nn.functional.cross_entropy(logits, y)
    assert abs(focal_loss(logits, y, 0.0, (0.5, 0.5), 0.0).item() - 0.5 * ce.item()) < 1e-6
    assert focal_loss(logits, y, 2.0, (0.4, 0.6), 0.1).item() > 0


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))["auc"] == 1.0
    assert classification_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))["auc"] == 0.0     # inverted ranking
    mask = _cam_mask()
    a = alignment_metrics(mask.clone(), mask)
    assert a["energyRatio"] == 1.0 and a["pointingGame"] == 1.0 and a["iou"] == 1.0
    b = alignment_metrics(1 - mask, mask)
    assert b["energyRatio"] == 0.0 and b["pointingGame"] == 0.0 and b["iou"] == 0.0


def test_split_is_deterministic_and_disjoint():
    labels = np.array([0] * 80 + [1] * 58)                # Montgomery class counts
    a = make_split(labels, 42)
    b = make_split(labels, 42)
    c = make_split(labels, 43)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[2], c[2])
    tr, va, te = a
    assert len(set(tr) | set(va) | set(te)) == 138 and len(tr) + len(va) + len(te) == 138
    assert len(te) == 28 and abs(labels[te].mean() - labels.mean()) < 0.05


def test_schedules():
    cfg = make_config("full")
    assert [curriculum(e, cfg) for e in (1, 5, 6, 7, 8, 9, 30)] == [0, 0, 0.25, 0.5, 0.75, 1, 1]
    assert abs(learning_rate(1, cfg) - cfg.lr / 2) < 1e-12 and abs(learning_rate(2, cfg) - cfg.lr) < 1e-12
    assert learning_rate(cfg.max_epochs, cfg) < 1e-9


def test_augment_and_mask_perturbation():
    x, m = torch.rand(4, 3, 32, 32), (torch.rand(4, 1, 32, 32) > 0.5).float()
    xa, ma = _augment(x, m, torch.Generator().manual_seed(0))
    assert xa.shape == x.shape and set(ma.unique().tolist()) <= {0.0, 1.0}
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 8:24, 8:24] = 1
    assert perturb_masks(mask, "erode", 2).sum() == 12 * 12
    assert perturb_masks(mask, "dilate", 2).sum() == 20 * 20
    assert torch.equal(perturb_masks(mask, "none", 2), mask)


def test_grad_cam_is_trainable():
    torch.manual_seed(0)
    model = CamNet("resnet50", pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    logits, feats = model(x)
    cam = grad_cam(logits, feats, (64, 64), create_graph=True)
    assert cam.shape == (2, 1, 64, 64) and cam.min() >= 0 and cam.max() <= 1
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 16:48, 16:48] = 1
    tversky_loss(cam, mask).backward()
    assert sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None) > 0


def _fake_dataset(root, prefix, n):
    (root / "CXR_png").mkdir(parents=True)
    (root / "mask").mkdir()
    rng = np.random.RandomState(0)
    for i in range(n):
        c = i % 2
        img = (rng.rand(48, 48) * 255).astype(np.uint8)
        msk = np.zeros((48, 48), np.uint8)
        msk[10:38, 8:40] = 255
        Image.fromarray(img).save(root / "CXR_png" / f"{prefix}_{i:04d}_{c}.png")
        Image.fromarray(msk).save(root / "mask" / f"{prefix}_{i:04d}_{c}_mask.png")


def test_end_to_end_smoke(tmp_path):
    _fake_dataset(tmp_path / "sz", "CHNCXR", 40)
    _fake_dataset(tmp_path / "mo", "MCUCXR", 30)
    for method in ("full", "sym_noclamp", "masked_input", "anat_only_erode", "anat_only_topk", "anat_only_mass"):
        cfg = make_config(method, dataset="shenzhen", transfer_dataset="montgomery", seed=1,
                          shenzhen_root=str(tmp_path / "sz"), montgomery_root=str(tmp_path / "mo"),
                          cache_dir=str(tmp_path / "cache"), out_root=str(tmp_path / "res"),
                          image_size=64, pretrained=False, max_epochs=2, min_epochs=2,
                          select_from_epoch=1, batch_size=8, save_checkpoint=False, verbose=False,
                          mask_perturb_px=2)
        row = train_single_seed(cfg, torch.device("cpu"))
        assert row["nTransfer"] == 30 and 0.0 <= row["energyRatio"] <= 1.0
        assert (tmp_path / "res" / method / "shenzhen" / "resnet50" / "per_seed.csv").is_file()
