"""Focal, Tversky-on-CAM, and Anatomical Guidance losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(logits, y, gamma: float = 2.0, alpha=(0.5, 0.5), label_smoothing: float = 0.0):
    """Focal loss with per-class weights and label smoothing.

    gamma = 0, alpha = (0.5, 0.5), label_smoothing = 0 gives plain cross-entropy
    (scaled by 0.5), which is the 'Vanilla' configuration.
    """
    c = logits.shape[1]
    t = F.one_hot(y, c).float()
    if label_smoothing > 0:
        t = (1.0 - label_smoothing) * t + label_smoothing / c
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    a = torch.as_tensor(alpha, dtype=logits.dtype, device=logits.device).view(1, c)
    per_sample = -(a * (1.0 - p).pow(gamma) * t * logp).sum(dim=1)
    return per_sample.mean()


def tversky_loss(cam, mask, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-6):
    """1 - Tversky index between a [0,1] CAM and a binary mask (confinement)."""
    tp = (cam * mask).sum(dim=(1, 2, 3))
    fp = (cam * (1.0 - mask)).sum(dim=(1, 2, 3))
    fn = ((1.0 - cam) * mask).sum(dim=(1, 2, 3))
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1.0 - ti).mean()


def inside_outside_means(cam, mask):
    """R_i: mean CAM inside the mask.  P_i: mean CAM outside the mask."""
    r = (cam * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-8)
    p = (cam * (1.0 - mask)).sum(dim=(1, 2, 3)) / ((1.0 - mask).sum(dim=(1, 2, 3)) + 1e-8)
    return r, p


def outside_topk_mean(cam, mask, frac: float = 0.10):
    """Mean of the top `frac` of CAM values OUTSIDE the mask, per image.

    Unlike the plain outside mean, a small intense hotspot cannot be diluted by
    the many near-zero pixels around it.
    """
    out = []
    for c, m in zip(cam.flatten(1), mask.flatten(1)):
        v = c[m < 0.5]
        if v.numel() == 0:
            out.append(c.sum() * 0.0)
            continue
        k = max(1, int(frac * v.numel()))
        out.append(v.topk(k).values.mean())
    return torch.stack(out)


def anatomical_guidance_loss(cam, mask, rho: float = 0.75, positivity_clamp: bool = True,
                             outside_stat: str = "mean", topk_frac: float = 0.10):
    """L_AG = mean_i max(0, P_i - rho * R_i)   (guidance).

    With the clamp, the term and its gradient are zero for every image with
    P_i <= rho * R_i. Without it, the term stays active and can become negative.
    outside_stat='topk' replaces the outside mean P_i by the mean of the top-k
    outside-lung values. Returns (loss, mean R, mean P); R and P are detached.
    """
    r, p = inside_outside_means(cam, mask)
    if outside_stat == "topk":
        p = outside_topk_mean(cam, mask, topk_frac)
    raw = p - rho * r
    if positivity_clamp:
        raw = F.relu(raw)
    return raw.mean(), r.detach().mean(), p.detach().mean()


def saliency_mass_loss(cam_raw, mask, tau: float = 0.30):
    """Guidance on saliency mass: at most a fraction `tau` of the (non-negative,
    un-normalised) CAM mass may lie outside the lungs.  L = mean_i max(0, outside_i - tau).
    Returns (loss, mean inside fraction, mean outside fraction)."""
    total = cam_raw.sum(dim=(1, 2, 3))
    outside = (cam_raw * (1.0 - mask)).sum(dim=(1, 2, 3)) / (total + 1e-8)
    loss = F.relu(outside - tau).mean()
    return loss, (1.0 - outside).detach().mean(), outside.detach().mean()
