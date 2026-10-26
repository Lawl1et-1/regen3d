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


DAMAGES = {"sphere": sphere, "plane": plane, "tunnel": tunnel}


def random_damage(x: torch.Tensor, kinds=("sphere", "plane", "tunnel"),
                  weights=(0.35, 0.40, 0.25)) -> tuple[torch.Tensor, str]:
    k = torch.multinomial(torch.tensor(weights), 1).item()
    name = kinds[k]
    return DAMAGES[name](x), name

