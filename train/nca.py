"""3D Neural Cellular Automaton.

State: (B, C, D, H, W) with C = 16 channels
    0      alpha (alive)
    1..4   cell types: capsule, parenchyma, vessel, drain
    5..15  hidden channels (signals / memory)

Axis convention: W is x, H is y, D is z. The web viewer stores voxels as
index = x + y*N + z*N*N and must use the same perception ordering:
    [identity(C), sobel_x(C), sobel_y(C), sobel_z(C)]

Speed: the Sobel filters run as one depthwise conv3d, and the per-cell MLP
only runs on cells inside the alive neighbourhood (everything else is masked
to zero anyway), which is ~5x less work for an organ filling ~15% of the grid.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

CHANNELS = 16
N_TYPES = 4
ALIVE_THRESHOLD = 0.1


def sobel_kernels() -> torch.Tensor:
    s = torch.tensor([1.0, 2.0, 1.0])
    d = torch.tensor([-1.0, 0.0, 1.0])
    kx = torch.einsum("z,y,x->zyx", s, s, d)
    ky = torch.einsum("z,y,x->zyx", s, d, s)
    kz = torch.einsum("z,y,x->zyx", d, s, s)
    return (torch.stack([kx, ky, kz]) / 32.0)[:, None]  # (3, 1, 3, 3, 3)


def alive_mask(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(x[:, :1], 3, 1, 1) > ALIVE_THRESHOLD


class NCA3D(nn.Module):
    def __init__(self, channels: int = CHANNELS, hidden: int = 96, fire_rate: float = 0.5):
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.fire_rate = fire_rate
        # depthwise weights: output channel 3c+k is kernel k applied to channel c
        self.register_buffer("kdw", sobel_kernels().repeat(channels, 1, 1, 1, 1))
        self.w1 = nn.Linear(channels * 4, hidden)   # input order [id, sx, sy, sz] x C
        self.w2 = nn.Linear(hidden, channels, bias=False)
        nn.init.zeros_(self.w2.weight)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        pre = alive_mask(x)
        g = F.conv3d(x, self.kdw.to(x.dtype), padding=1, groups=c)  # (B, 3C, ...)
        bi, vi = pre.view(b, -1).nonzero(as_tuple=True)
        xr = x.view(b, c, -1)[bi, :, vi]                                   # (A, C)
        gr = g.view(b, 3 * c, -1)[bi, :, vi].view(-1, c, 3).transpose(1, 2).reshape(-1, 3 * c)
        p = torch.cat([xr, gr], 1)
        dr = self.w2(F.relu(self.w1(p))).float()
        fire = (torch.rand(dr.shape[0], 1, device=x.device) < self.fire_rate).to(dr.dtype)
        dx = x.new_zeros(b, d * h * w, c).index_put((bi, vi), dr * fire)
        x = x + dx.permute(0, 2, 1).reshape(b, c, d, h, w)
        post = alive_mask(x)
        return x * (pre & post).to(x.dtype)

    def forward(self, x: torch.Tensor, steps: int, use_checkpoint: bool = False,
                chunk: int = 4, post=None) -> torch.Tensor:
        """Run `steps` updates. `post(x, step)` (optional) is applied after every step, e.g. a
        lesion that keeps spreading while the cells react."""
        def run(x_, start, k):
            for i in range(k):
                x_ = self.step(x_)
                if post is not None:
                    x_ = post(x_, start + i)
            return x_

        if not use_checkpoint:
            return run(x, 0, steps)
        done = 0
        while done < steps:
            k = min(chunk, steps - done)
            x = checkpoint(run, x, done, k, use_reentrant=False)
            done += k
        return x


def make_seed(n: int, seed_xyz, batch: int = 1, channels: int = CHANNELS, device="cpu"):
    x = torch.zeros(batch, channels, n, n, n, device=device)
    sx, sy, sz = seed_xyz
    x[:, 0, sz, sy, sx] = 1.0
    x[:, 1 + N_TYPES:, sz, sy, sx] = 1.0
    return x
