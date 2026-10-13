"""Classification metrics and CAM-vs-mask anatomical alignment metrics."""
from __future__ import annotations

import numpy as np
import torch


def roc_auc_score(y_true, scores) -> float:
    """ROC AUC as the Mann-Whitney statistic with average ranks for ties.
    Identical to sklearn.metrics.roc_auc_score for binary labels; numpy only, so it also works
    where compiled scikit-learn extensions cannot be loaded."""
    y = np.asarray(y_true).ravel().astype(bool)
    s = np.asarray(scores, dtype=np.float64).ravel()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC AUC needs both classes")
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):                                   # average rank within each block of ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def classification_metrics(y_true: np.ndarray, p_tb: np.ndarray) -> dict:
    y_pred = (p_tb >= 0.5).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    auc = float(roc_auc_score(y_true, p_tb)) if len(np.unique(y_true)) == 2 else float("nan")
    return dict(auc=auc, acc=float((y_pred == y_true).mean()), f1=f1, sens=sens, spec=spec,
                ece=expected_calibration_error(p_tb, y_true))


def expected_calibration_error(p_tb, y_true, n_bins: int = 15) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(p_tb)
    for b in range(n_bins):
        in_bin = (p_tb > edges[b]) & (p_tb <= edges[b + 1])
        if not in_bin.any():
            continue
        conf = p_tb[in_bin].mean()
        acc = (y_true[in_bin] == (p_tb[in_bin] >= 0.5)).mean()
        ece += in_bin.sum() / n * abs(conf - acc)
    return float(ece)


@torch.no_grad()
def alignment_per_image(cams: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5):
    """Per-image Energy Ratio, Pointing Game hit, CAM-IoU.  cams, masks: N x 1 x H x W."""
    n = cams.shape[0]
    c = cams.reshape(n, -1).float()
    m = masks.reshape(n, -1) > 0.5
    total = c.sum(dim=1)
    er = (c * m).sum(dim=1) / total.clamp_min(1e-12)
    er[total < 1e-12] = float("nan")
    hit = m.gather(1, c.argmax(dim=1, keepdim=True)).squeeze(1).float()
    cb = c > threshold
    inter = (cb & m).sum(dim=1).float()
    union = (cb | m).sum(dim=1).float()
    iou = inter / union.clamp_min(1.0)
    iou[union == 0] = float("nan")
    return er, hit, iou


def alignment_metrics(cams, masks, threshold: float = 0.5) -> dict:
    er, hit, iou = alignment_per_image(cams, masks, threshold)
    return dict(energyRatio=float(torch.nanmean(er)), pointingGame=float(hit.mean()),
                iou=float(torch.nanmean(iou)))
