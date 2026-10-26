"""3D damage used during training (and reproduced by the viewer's tools).

All functions zero the state inside the damaged region, exactly like the
viewer's scalpel, slice and toxin tools.
"""
from __future__ import annotations

import torch


def _coords(n: int, device) -> torch.Tensor:
    r = torch.arange(n, device=device, dtype=torch.float32)
    z, y, x = torch.meshgrid(r, r, r, indexing="ij")
    return torch.stack([x, y, z], -1)  # (n, n, n, 3)


def _random_alive_point(x: torch.Tensor, gen: torch.Generator | None) -> torch.Tensor:
    idx = torch.nonzero(x[0] > 0.1)
    if len(idx) == 0:
        n = x.shape[-1]
        return torch.full((3,), n / 2, device=x.device)
    i = torch.randint(len(idx), (1,), generator=gen, device="cpu").item()
    z, y, xx = idx[i].tolist()
    return torch.tensor([xx, y, z], device=x.device, dtype=torch.float32)


def _random_dir(device) -> torch.Tensor:
    v = torch.randn(3, device=device)
    return v / v.norm()


def sphere(x: torch.Tensor, frac_range=(0.10, 0.32)) -> torch.Tensor:
    """x: (C, n, n, n). Spherical hole centred on a random alive voxel."""
    n = x.shape[-1]
    c = _random_alive_point(x, None)
    r = n * (frac_range[0] + torch.rand(1).item() * (frac_range[1] - frac_range[0]))
    p = _coords(n, x.device)
    keep = ((p - c).norm(dim=-1) > r).float()
    return x * keep


def plane(x: torch.Tensor, remove_range=(0.08, 0.5)) -> torch.Tensor:
    """Remove a half-space: random normal, offset chosen so a fraction of tissue is cut."""
    n = x.shape[-1]
    p = _coords(n, x.device)
    nrm = _random_dir(x.device)
    proj = p @ nrm
    alive = x[0] > 0.1
    if alive.sum() < 10:
        return x
    f = remove_range[0] + torch.rand(1).item() * (remove_range[1] - remove_range[0])
    q = torch.quantile(proj[alive].float(), 1.0 - f)
    return x * (proj <= q).float()


def tunnel(x: torch.Tensor, radius_range=(1.5, 5.0)) -> torch.Tensor:
    """A straight cylindrical tunnel through a random alive voxel."""
    n = x.shape[-1]
    p = _coords(n, x.device)
    c = _random_alive_point(x, None)
    d = _random_dir(x.device)
    r = radius_range[0] + torch.rand(1).item() * (radius_range[1] - radius_range[0])
    v = p - c
    perp = v - (v @ d)[..., None] * d
    return x * (perp.norm(dim=-1) > r).float()


def wedge_region(n: int, e, nrm, v, u, s0, s1, depth, w0, device="cpu") -> torch.Tensor:
    """Bool mask of the swipe incision (same as wedgeDist in web/src/shaders/fields.wgsl).

    e: entry point on the surface (voxels), nrm: blade-plane normal, v: direction into
    the tissue, u: along the stroke; [s0, s1] stroke extent, depth and half-width in voxels.
    """
    p = _coords(n, device)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    d = p - t(e)
    tt = d @ t(v)
    ss = d @ t(u)
    aa = (d @ t(nrm)).abs()
    w = w0 * (1.0 - 0.7 * (tt / depth).clamp(0.0, 1.0))
    sd = torch.maximum(torch.maximum(aa - w, tt - depth), torch.maximum(s0 - ss, ss - s1))
    return sd < 0


def _orthonormal(v: torch.Tensor):
    a = torch.tensor([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else torch.tensor([0.0, 1.0, 0.0])
    u = torch.linalg.cross(v, a)
    u = u / u.norm()
    return u, torch.linalg.cross(v, u)


def wedge(x: torch.Tensor, depth_range=(0.15, 0.9), width_range=(1.5, 3.5)) -> torch.Tensor:
    """Random swipe incision, as a user would make it: pick a view direction, a blade
    plane containing it, the entry point where that plane meets the visible surface,
    a stroke extent and a depth (fraction of the tissue thickness along the view)."""
    n = x.shape[-1]
    dev = x.device
    alive = x[0] > 0.1
    if alive.sum() < 10:
        return x
    v = _random_dir("cpu")                       # camera looks along v
    u_, w_ = _orthonormal(v)
    ang = torch.rand(1).item() * 3.14159
    u = u_ * torch.cos(torch.tensor(ang)) + w_ * torch.sin(torch.tensor(ang))   # stroke direction
    nrm = torch.linalg.cross(u, v)               # blade plane contains u and v
    pts = _coords(n, dev)[alive]                 # (A, 3)
    c = pts[torch.randint(len(pts), (1,)).item()]
    # entry: the frontmost alive voxel near the blade line through c
    rel = pts - c
    near = ((rel @ nrm.to(dev)).abs() < 1.5) & ((rel @ u.to(dev)).abs() < 1.5)
    cand = pts[near] if near.any() else pts
    tv = cand @ v.to(dev)
    e = cand[tv.argmin()]
    thick = (tv.max() - tv.min()).item() + 1.0
    depth = max(2.0, thick * (depth_range[0] + torch.rand(1).item() * (depth_range[1] - depth_range[0])))
    su = pts @ u.to(dev)
    lo, hi = su.min().item(), su.max().item()
    se = (e @ u.to(dev)).item()
    s0 = se - (se - lo) * (0.3 + 0.7 * torch.rand(1).item()) - 1.0
    s1 = se + (hi - se) * (0.3 + 0.7 * torch.rand(1).item()) + 1.0
    w0 = width_range[0] + torch.rand(1).item() * (width_range[1] - width_range[0])
    m = wedge_region(n, e.tolist(), nrm.tolist(), v.tolist(), u.tolist(), s0 - se, s1 - se, depth, w0, dev)
    return x * (~m).float()


DAMAGES = {"sphere": sphere, "plane": plane, "tunnel": tunnel, "wedge": wedge}


def random_damage(x: torch.Tensor, kinds=("sphere", "plane", "tunnel"),
                  weights=(0.35, 0.40, 0.25)) -> tuple[torch.Tensor, str]:
    k = torch.multinomial(torch.tensor(weights), 1).item()
    name = kinds[k]
    return DAMAGES[name](x), name

