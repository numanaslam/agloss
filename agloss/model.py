"""Backbone with access to the last convolutional feature map, and Grad-CAM."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CamNet(nn.Module):
    """ImageNet-pretrained classifier returning (logits, last-conv feature map)."""

    def __init__(self, arch: str = "resnet50", pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        self.arch = arch
        if arch == "resnet50":
            w = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            m = torchvision.models.resnet50(weights=w)
            self.features = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                                          m.layer1, m.layer2, m.layer3, m.layer4)
            self.fc = nn.Linear(m.fc.in_features, num_classes)
        elif arch == "densenet121":
            w = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            m = torchvision.models.densenet121(weights=w)
            self.features = nn.Sequential(m.features, nn.ReLU(inplace=False))
            self.fc = nn.Linear(m.classifier.in_features, num_classes)
        elif arch == "convnext_tiny":
            w = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            m = torchvision.models.convnext_tiny(weights=w)
            self.features = m.features
            self.head_norm = m.classifier[0]              # LayerNorm2d applied after pooling
            self.fc = nn.Linear(m.classifier[2].in_features, num_classes)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        # No dropout layers are used, so Grad-CAM is deterministic for a given input.

    def forward(self, x):
        feats = self.features(x)                              # B x C x h x w
        pooled = F.adaptive_avg_pool2d(feats, 1)
        if self.arch == "convnext_tiny":                  # here the Grad-CAM weights depend on the input
            pooled = self.head_norm(pooled)
        logits = self.fc(torch.flatten(pooled, 1))
        return logits, feats


def grad_cam(logits, feats, out_size, target_class: int = 1, create_graph: bool = False, return_raw: bool = False):
    """Grad-CAM for `target_class` (1 = TB), upsampled and min-max normalised to [0, 1].

    With create_graph=True the map stays differentiable w.r.t. the network
    parameters (second-order path), which is what the spatial losses train through.
    """
    score = logits[:, target_class].sum()
    grads = torch.autograd.grad(score, feats, create_graph=create_graph, retain_graph=True)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)            # B x C x 1 x 1
    cam = F.relu((weights * feats).sum(dim=1, keepdim=True))  # B x 1 x h x w
    if cam.shape[-2:] != tuple(out_size):
        cam = F.interpolate(cam, size=tuple(out_size), mode="bilinear", align_corners=False)
    cmin = cam.amin(dim=(2, 3), keepdim=True)
    cmax = cam.amax(dim=(2, 3), keepdim=True)
    norm = (cam - cmin) / (cmax - cmin + 1e-8)
    return (norm, cam) if return_raw else norm        # cam: non-negative, un-normalised
