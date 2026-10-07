"""Train one (method, dataset, seed) run and evaluate in-domain and transfer."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import GpuData, load_arrays, make_split, perturb_masks
from .losses import anatomical_guidance_loss, focal_loss, saliency_mass_loss, tversky_loss
from .metrics import alignment_metrics, classification_metrics
from .model import CamNet, grad_cam

CSV_FIELDS = ["method", "dataset", "arch", "seed", "bestEpoch", "bestValAUC", "epochsRun",
              "testAUC", "testAcc", "testF1", "testSens", "testSpec", "testECE",
              "energyRatio", "pointingGame", "camIoU",
              "transferDataset", "transferAUC", "transferAcc", "transferF1", "transferSens",
              "transferSpec", "transferER", "transferPG", "transferIoU", "nTrain", "nVal",
              "nTest", "nTransfer", "minutes"]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def curriculum(epoch: int, cfg: Config) -> float:
    """s(e): 0 up to the warm-up epoch, then a linear ramp to 1 over `curr_ramp` epochs."""
    if epoch < cfg.curr_warmup:
        return 0.0
    if epoch >= cfg.curr_warmup + cfg.curr_ramp:
        return 1.0
    return (epoch - cfg.curr_warmup) / cfg.curr_ramp


def learning_rate(epoch: int, cfg: Config) -> float:
    """Linear warm-up, then cosine annealing (per-epoch)."""
    if epoch <= cfg.lr_warmup:
        return cfg.lr * epoch / max(cfg.lr_warmup, 1)
    t = (epoch - cfg.lr_warmup) / max(cfg.max_epochs - cfg.lr_warmup, 1)
    return 0.5 * cfg.lr * (1.0 + math.cos(math.pi * t))


def data_root(cfg: Config, name: str) -> str:
    return cfg.shenzhen_root if name == "shenzhen" else cfg.montgomery_root


def out_dir(cfg: Config) -> Path:
    return Path(cfg.out_root) / cfg.method / cfg.dataset / cfg.arch


def evaluate(model, data: GpuData, idx, cfg: Config, return_arrays: bool = False):
    """Classification + alignment metrics on the given indices (model in eval mode)."""
    model.eval()
    probs, cams, masks = [], [], []
    bs = max(32, cfg.batch_size)
    for b0 in range(0, len(idx), bs):
        x, _, m = data.batch(idx[b0:b0 + bs], augment=False, masked_input=cfg.masked_input)
        with torch.enable_grad():
            logits, feats = model(x)
            cam = grad_cam(logits, feats, x.shape[-2:], target_class=1, create_graph=False)
        probs.append(torch.softmax(logits.detach(), dim=1)[:, 1])
        cams.append(cam.detach())
        masks.append(m)
    probs, cams, masks = torch.cat(probs), torch.cat(cams), torch.cat(masks)
    y_true = data.arrays.labels[np.asarray(idx)]
    out = classification_metrics(y_true, probs.cpu().numpy().astype(np.float64))
    out.update(alignment_metrics(cams, masks))
    if return_arrays:
        out["p_tb"], out["cams"], out["masks"] = probs.cpu().numpy(), cams.cpu(), masks.cpu()
    return out


def already_done(cfg: Config) -> bool:
    p = out_dir(cfg) / "per_seed.csv"
    if not p.is_file():
        return False
    with open(p, newline="") as f:
        return any(int(float(r["seed"])) == cfg.seed for r in csv.DictReader(f))


def train_single_seed(cfg: Config, device=None, data_cache: dict | None = None) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    seed_everything(cfg.seed)
    if cfg.verbose:
        print(f"\n=== seed {cfg.seed} | {cfg.method} | {cfg.dataset} | {cfg.arch} ===", flush=True)

    data_cache = data_cache if data_cache is not None else {}

    def get_data(name):
        if name not in data_cache:
            arrays = load_arrays(name, data_root(cfg, name), cfg.image_size, cfg.cache_dir, cfg.verbose)
            data_cache[name] = GpuData(arrays, device)
        return data_cache[name]

    data = get_data(cfg.dataset)
    train_idx, val_idx, test_idx = make_split(data.arrays.labels, cfg.seed, cfg.test_fraction,
                                              cfg.n_folds, cfg.used_fold)

    model = CamNet(cfg.arch, cfg.pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    order_rng = np.random.RandomState(cfg.seed + 10_000)
    need_cam = cfg.use_tversky or cfg.use_anat

    best_auc, best_epoch, since_best, best_state = -math.inf, 0, 0, None
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        s = curriculum(epoch, cfg)
        for g in opt.param_groups:
            g["lr"] = learning_rate(epoch, cfg)

        model.train()
        order = train_idx[order_rng.permutation(len(train_idx))]
        sums = dict(total=0.0, focal=0.0, tversky=0.0, anat=0.0, R=0.0, P=0.0)
        n_batches = 0
        for b0 in range(0, len(order), cfg.batch_size):
            idx_b = order[b0:b0 + cfg.batch_size]
            if len(idx_b) < 2:                    # batch-norm needs more than one sample
                continue
            x, y, m = data.batch(idx_b, augment=cfg.augment, generator=gen, masked_input=cfg.masked_input)
            m_loss = perturb_masks(m, cfg.mask_perturb, cfg.mask_perturb_px)

            logits, feats = model(x)
            l_focal = focal_loss(logits, y, cfg.focal_gamma, cfg.focal_alpha, cfg.label_smoothing)
            total = l_focal
            l_tv = l_ag = r_mean = p_mean = torch.zeros((), device=device)
            if need_cam:
                cam, cam_raw = grad_cam(logits, feats, x.shape[-2:], target_class=1, create_graph=True,
                                        return_raw=True)
                if cfg.use_tversky:
                    l_tv = tversky_loss(cam, m_loss, cfg.tv_alpha, cfg.tv_beta, cfg.tv_eps)
                    total = total + cfg.tv_lambda * l_tv
                if cfg.use_anat:
                    if cfg.cam_norm == "mass":
                        l_ag, r_mean, p_mean = saliency_mass_loss(cam_raw, m_loss, cfg.mass_tau)
                    else:
                        l_ag, r_mean, p_mean = anatomical_guidance_loss(
                            cam, m_loss, cfg.rho, cfg.positivity_clamp, cfg.outside_stat, cfg.topk_frac)
                    total = total + cfg.anat_lambda * s * l_ag

            opt.zero_grad(set_to_none=True)
            total.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            n_batches += 1
            for k, v in (("total", total), ("focal", l_focal), ("tversky", l_tv), ("anat", l_ag),
                         ("R", r_mean), ("P", p_mean)):
                sums[k] += float(v.detach())

        avg = {k: v / max(n_batches, 1) for k, v in sums.items()}
        val = evaluate(model, data, val_idx, cfg)
        history.append(dict(epoch=epoch, s=s, lr=opt.param_groups[0]["lr"], valAUC=val["auc"],
                            valER=val["energyRatio"], valIoU=val["iou"], **avg))
        if cfg.verbose:
            print(f"epoch {epoch:2d}/{cfg.max_epochs} | s={s:.2f} | loss={avg['total']:.4f} "
                  f"(focal {avg['focal']:.3f} tv {avg['tversky']:.3f} ag {avg['anat']:+.3f} "
                  f"R {avg['R']:.2f} P {avg['P']:.2f}) | val AUC={val['auc']:.4f}", flush=True)

        # Best checkpoint is selected among epochs >= select_from_epoch (after the curriculum).
        if epoch >= cfg.select_from_epoch:
            if val["auc"] > best_auc:
                best_auc, best_epoch, since_best = val["auc"], epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since_best += 1
        if epoch >= cfg.min_epochs and since_best >= cfg.patience:
            if cfg.verbose:
                print(f"early stop at epoch {epoch} (best epoch {best_epoch}, val AUC {best_auc:.4f})")
            break

    if best_state is None:                        # max_epochs < select_from_epoch (smoke tests)
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_auc, best_epoch = val["auc"], epoch
    model.load_state_dict(best_state)

    test = evaluate(model, data, test_idx, cfg)
    row = dict(method=cfg.method, dataset=cfg.dataset, arch=cfg.arch, seed=cfg.seed,
               bestEpoch=best_epoch, bestValAUC=best_auc, epochsRun=epoch,
               testAUC=test["auc"], testAcc=test["acc"], testF1=test["f1"], testSens=test["sens"],
               testSpec=test["spec"], testECE=test["ece"], energyRatio=test["energyRatio"],
               pointingGame=test["pointingGame"], camIoU=test["iou"],
               transferDataset="", transferAUC=float("nan"), transferAcc=float("nan"),
               transferF1=float("nan"), transferSens=float("nan"), transferSpec=float("nan"),
               transferER=float("nan"), transferPG=float("nan"), transferIoU=float("nan"),
               nTrain=len(train_idx), nVal=len(val_idx), nTest=len(test_idx), nTransfer=0)

    if cfg.transfer_dataset:
        tdata = get_data(cfg.transfer_dataset)
        all_idx = np.arange(len(tdata.arrays.labels))       # the ENTIRE target dataset
        tr = evaluate(model, tdata, all_idx, cfg)
        row.update(transferDataset=cfg.transfer_dataset, transferAUC=tr["auc"], transferAcc=tr["acc"],
                   transferF1=tr["f1"], transferSens=tr["sens"], transferSpec=tr["spec"],
                   transferER=tr["energyRatio"], transferPG=tr["pointingGame"], transferIoU=tr["iou"],
                   nTransfer=len(all_idx))
    row["minutes"] = (time.time() - t0) / 60.0

    d = out_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    csv_path = d / "per_seed.csv"
    new = not csv_path.is_file()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    with open(d / f"seed_{cfg.seed}_history.json", "w") as f:
        json.dump(dict(config=cfg.to_dict(), history=history), f, indent=1)
    if cfg.save_checkpoint:
        torch.save(dict(state_dict=best_state, config=cfg.to_dict()), d / f"seed_{cfg.seed}_best.pt")

    if cfg.verbose:
        msg = (f"test: AUC={test['auc']:.4f} ER={test['energyRatio']:.4f} "
               f"PG={test['pointingGame']:.4f} IoU={test['iou']:.4f}")
        if cfg.transfer_dataset:
            msg += (f" | transfer->{cfg.transfer_dataset}: AUC={row['transferAUC']:.4f} "
                    f"ER={row['transferER']:.4f} IoU={row['transferIoU']:.4f}")
        print(msg + f" | {row['minutes']:.1f} min", flush=True)
    return row


def load_checkpoint(path, device):
    """Rebuild a trained model and its Config from a saved checkpoint."""
    ck = torch.load(path, map_location=device)
    c = ck["config"]
    c["focal_alpha"] = tuple(c["focal_alpha"])
    cfg = Config(**c)
    model = CamNet(cfg.arch, pretrained=False).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg
