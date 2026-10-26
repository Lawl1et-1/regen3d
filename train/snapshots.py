"""Small CPU/GPU volume renderer for training logs: rendered 3D snapshots, not loss curves."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

TYPE_COLORS = np.array([
    [0.95, 0.86, 0.82],  # capsule
    [0.92, 0.48, 0.44],  # parenchyma
    [0.80, 0.06, 0.14],  # vessel
    [0.36, 0.42, 0.88],  # drain
], np.float32)


def _rotation(yaw: float, pitch: float) -> torch.Tensor:
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return (rx @ ry).float()


@torch.no_grad()
def render(state: torch.Tensor, yaw=0.6, pitch=-0.35, size=160) -> np.ndarray:
    """state: (C, n, n, n) tensor -> (size, size, 3) float image in [0, 1]."""
    x = state[None, :5].float().cpu()
    rot = _rotation(yaw, pitch)
    theta = torch.cat([rot, torch.zeros(3, 1)], 1)[None]
    d = size
    grid = F.affine_grid(theta, (1, 5, d, size, size), align_corners=False)
    v = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    alpha = v[0].clamp(0, 1)  # (d, h, w), depth along d (front = 0)
    types = v[1:5].clamp(0, None)
    wsum = types.sum(0, keepdim=True) + 1e-4
    col = torch.einsum("tdhw,tc->dhwc", types / wsum, torch.from_numpy(TYPE_COLORS))
    # cheap lighting: darken with depth into the tissue
    depth_shade = torch.linspace(1.15, 0.45, d)[:, None, None, None]
    col = col * depth_shade
    a = (alpha * 0.22)[..., None]
    trans = torch.cumprod(torch.cat([torch.ones_like(a[:1]), 1 - a[:-1]], 0), 0)
    img = (trans * a * col).sum(0)
    acc = (trans * a).sum(0)
    bg = torch.tensor([0.04, 0.045, 0.06])
    img = img + (1 - acc) * bg
    return img.flip(0).clamp(0, 1).numpy()


def strip(images, labels=None) -> np.ndarray:
    pad = 4
    h, w, _ = images[0].shape
    out = np.zeros((h, (w + pad) * len(images) - pad, 3), np.float32)
    for i, im in enumerate(images):
        out[:, i * (w + pad): i * (w + pad) + w] = im
    return out


def save_png(img: np.ndarray, path, rows_labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(str(path), np.clip(img, 0, 1))
