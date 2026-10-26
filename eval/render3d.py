"""Offline lit renderer for NCA states (PyTorch, GPU if available).

Two looks:
  tissue  first-surface shading: smoothed density isosurface, normals from the
          depth map, wrap diffuse + soft specular + rim + screen-space occlusion.
          Tissue the AI regrew can be tinted cyan; fresh cut walls show a pale section.
  xray    vessels and drains glowing inside a faint tissue silhouette.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F

BG = np.array([5, 7, 11]) / 255.0
COL = {
    "capsule": torch.tensor([0.42, 0.13, 0.10]),
    "parenchyma": torch.tensor([0.36, 0.085, 0.065]),
    "vessel": torch.tensor([0.30, 0.02, 0.03]),
    "drain": torch.tensor([0.28, 0.24, 0.42]),
}
REGROWN = torch.tensor([0.25, 0.85, 1.0])
SECTION = torch.tensor([0.62, 0.36, 0.33])


def _dev():
    want = os.environ.get("REGEN_DEVICE")
    if want:
        return torch.device(want)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _rot(yaw: float, pitch: float) -> torch.Tensor:
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return (rx @ ry).float()


def _blur(v: torch.Tensor, sigma: float) -> torch.Tensor:
    r = max(1, int(math.ceil(sigma * 2.5)))
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=v.device)
    k = torch.exp(-x * x / (2 * sigma * sigma))
    k = k / k.sum()
    c = v.shape[1]
    v = F.conv3d(F.pad(v, (r, r, 0, 0, 0, 0), mode="replicate"), k.view(1, 1, 1, 1, -1).repeat(c, 1, 1, 1, 1), groups=c)
    v = F.conv3d(F.pad(v, (0, 0, r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1, 1).repeat(c, 1, 1, 1, 1), groups=c)
    v = F.conv3d(F.pad(v, (0, 0, 0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1, 1).repeat(c, 1, 1, 1, 1), groups=c)
    return v


@torch.no_grad()
def render(state: torch.Tensor, yaw=0.0, pitch=-0.12, size=320, zoom=0.82, mode="tissue",
           regrown: torch.Tensor | None = None, wound: torch.Tensor | None = None,
           ghost: torch.Tensor | None = None) -> np.ndarray:
    """state: (C, n, n, n) -> (size, size, 3) float image in [0, 1].

    regrown: bool (n,n,n) voxels the AI rebuilt (tinted cyan)
    wound:   bool (n,n,n) fresh cut walls (pale section colour)
    ghost:   bool (n,n,n) shape to draw as a faint outline (e.g. the missing tissue)
    """
    dev = _dev()
    n = state.shape[-1]
    vol = state[:5].float().to(dev).clamp(0, 1)
    extra = []
    for m in (regrown, wound, ghost):
        extra.append((m.float() if m is not None else torch.zeros(n, n, n)).to(dev))
    vol = torch.cat([vol, torch.stack(extra)], 0)[None]  # (1, 8, n, n, n)
    vol = _blur(vol, 0.75 if mode == "tissue" else 0.5)
    rot = _rot(yaw, pitch)
    theta = torch.cat([rot / zoom, torch.zeros(3, 1)], 1)[None].to(dev)
    depth = size
    grid = F.affine_grid(theta, (1, 8, depth, size, size), align_corners=False)
    v = F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    v = v.flip(1).flip(2)  # depth: front (+z, facing the viewer) first; rows: top (+y) first
    a = v[0]  # (D, H, W), D = depth from the viewer (index 0 = front)
    bg = torch.tensor(BG, dtype=torch.float32, device=dev)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size, device=dev), torch.linspace(-1, 1, size, device=dev), indexing="ij")
    vign = (1 - 0.35 * (xx ** 2 + yy ** 2)).clamp(0.5, 1)[..., None]

    if mode == "xray":
        w = a.clamp(0, 1)
        ves = ((v[3] - 0.15).clamp(min=0) * w).sum(0) / depth * 9.0
        drn = (v[4] * w).sum(0) / depth * 6.0
        tissue = w.sum(0) / depth * 2.0
        edge = (tissue - _blur2(tissue, 2.0)).abs() * 6
        img = (bg + torch.tensor([0.08, 0.16, 0.26], device=dev) * (tissue * 0.5 + edge)[..., None]
               + torch.tensor([1.3, 0.12, 0.18], device=dev) * ves[..., None]
               + torch.tensor([0.30, 0.45, 1.3], device=dev) * drn[..., None])
        if ghost is not None:
            g = (v[7] * 1.0).sum(0) / depth * 3.0
            img = img + torch.tensor([0.2, 0.8, 1.0], device=dev) * g[..., None] * 0.6
        img = _tonemap(img) * vign
        return img.clamp(0, 1).cpu().numpy()

    hit = a > 0.45
    any_hit = hit.any(0)
    idx = torch.where(any_hit, hit.float().argmax(0), torch.full_like(any_hit, depth - 1, dtype=torch.long))
    # refine the crossing between the two samples around the threshold (removes contour bands)
    a_hit = a.gather(0, idx[None]).squeeze(0)
    a_prev = a.gather(0, (idx - 1).clamp(min=0)[None]).squeeze(0)
    frac = ((0.45 - a_prev) / (a_hit - a_prev).clamp(min=1e-4)).clamp(0, 1)
    zmap = (idx.float() - 1 + frac) / depth * 2.0  # depth in view units
    # normals from the depth map
    dzdx = torch.zeros_like(zmap); dzdy = torch.zeros_like(zmap)
    zs = _blur2(torch.where(any_hit, zmap, zmap.max()), 1.2)
    dzdx[:, 1:-1] = (zs[:, 2:] - zs[:, :-2]) * size / 4
    dzdy[1:-1] = (zs[2:] - zs[:-2]) * size / 4
    nrm = torch.stack([dzdx, -dzdy, torch.ones_like(zmap)], -1)
    nrm = nrm / nrm.norm(dim=-1, keepdim=True)
    take = lambda c: v[c].gather(0, idx[None]).squeeze(0)  # noqa: E731
    types = torch.stack([take(1), take(2), take(3), take(4)], -1).clamp(min=0)
    tw = types / types.sum(-1, keepdim=True).clamp(min=1e-3)
    pal = torch.stack([COL["capsule"], COL["parenchyma"], COL["vessel"], COL["drain"]]).to(dev)
    alb = tw @ pal
    rg = take(5).clamp(0, 1)[..., None]
    wd = take(6).clamp(0, 1)[..., None]
    alb = alb * (1 - wd * 0.9) + SECTION.to(dev) * wd * 0.9
    alb = torch.lerp(alb, alb * 0.4 + REGROWN.to(dev) * 0.35, rg * 0.55)
    L = torch.tensor([-0.45, 0.6, 0.66], device=dev); L = L / L.norm()
    V = torch.tensor([0.0, 0.0, 1.0], device=dev)
    H = (L + V); H = H / H.norm()
    ndl = (nrm @ L)
    wrap = ((ndl + 0.4) / 1.4).clamp(min=0)[..., None]
    spec = (nrm @ H).clamp(min=0) ** 60 * 0.55 + (nrm @ H).clamp(min=0) ** 10 * 0.06
    rim = (1 - nrm[..., 2]).clamp(0, 1) ** 2.5
    ao = (1 - (zs - _blur2(zs, 6.0)).clamp(min=0) * 3.0).clamp(0.45, 1)[..., None]
    sss = torch.tensor([1.0, 0.35, 0.28], device=dev) * (1 - ndl.clamp(min=0))[..., None] * 0.25
    col = alb * (wrap * 1.9 * torch.tensor([1.0, 0.95, 0.9], device=dev) + 0.35) * ao + alb * sss
    col = col + spec[..., None] * (1 - wd * 0.8) + torch.tensor([0.35, 0.5, 0.75], device=dev) * rim[..., None] * 0.25
    col = col + REGROWN.to(dev) * rg * 0.18
    img = torch.where(any_hit[..., None], col, bg.expand_as(col))
    if ghost is not None:
        g = (v[7] * (1 - a.clamp(0, 1))).sum(0) / depth * 4.0
        img = img + torch.tensor([0.2, 0.85, 1.0], device=dev) * g.clamp(0, 1)[..., None] * 0.5
    img = _tonemap(img) * vign
    return img.clamp(0, 1).cpu().numpy()


def _blur2(img: torch.Tensor, sigma: float) -> torch.Tensor:
    return _blur(img[None, None, None], sigma)[0, 0, 0]


def _tonemap(x: torch.Tensor) -> torch.Tensor:
    a = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return a.clamp(0, 1) ** (1 / 2.2)
