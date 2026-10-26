"""Render every target organ (two views each) to targets/preview.png."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
from snapshots import render, save_png, strip  # noqa: E402

rows = []
for p in sorted((ROOT / "targets" / "data").glob("*.npy")):
    lab = torch.from_numpy(np.load(p).astype(np.int64))
    x = torch.zeros(16, *lab.shape)
    x[0] = (lab > 0).float()
    for t in range(1, 5):
        x[t] = (lab == t).float()
    xr = x.clone(); xr[1] *= 0.15  # see-through capsule
    rows.append(strip([render(x), render(x, yaw=2.2, pitch=-0.2), render(xr, yaw=0.6, pitch=-1.2)]))
w = max(r.shape[1] for r in rows)
img = np.concatenate([np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0))) for r in rows], 0)
save_png(img, ROOT / "targets" / "preview.png")
print("ok", [p.stem for p in sorted((ROOT / "targets" / "data").glob("*.npy"))])
