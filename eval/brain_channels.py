"""Which hidden channels carry the wound signal? (used by the viewer's AI Brain view)

Healthy organ -> settle -> swipe incision -> 25 steps. For each hidden channel, the
mean change near the wound vs far from it. Prints the top channels as state indices.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "eval"))
from liver_results import load_model, swipe_lesion  # noqa: E402
from nca import CHANNELS  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target = np.load(ROOT / "targets" / "data" / "liver.npy")
model, _ = load_model(ROOT / "train" / "runs" / "liver" / "model.pt", dev)
n = target.shape[0]
x = torch.zeros(1, CHANNELS, n, n, n, device=dev)
x[0, 0] = torch.from_numpy(target > 0).float()
for t in range(1, 5):
    x[0, t] = torch.from_numpy(target == t).float()
torch.manual_seed(0)
with torch.no_grad():
    for _ in range(60):
        x = model.step(x)
    healthy = x.clone()
    mask = swipe_lesion(target)
    y = x * (~torch.from_numpy(mask).to(dev)).float()
    for _ in range(25):
        y = model.step(y)
near = torch.from_numpy(ndimage.binary_dilation(mask, iterations=4) & (target > 0)).to(dev)
far = torch.from_numpy((target > 0) & ~ndimage.binary_dilation(mask, iterations=10)).to(dev)
scores = []
for c in range(5, CHANNELS):
    d = (y[0, c] - healthy[0, c]).abs()
    scores.append((float(d[near].mean() / (d[far].mean() + 1e-4)), float(d[near].mean()), c))
scores.sort(reverse=True)
for s in scores:
    print(f"channel {s[2]:2d}  near/far {s[0]:7.2f}  near change {s[1]:.3f}")
# focused on the wound (near/far) and strong (near change)
top = [s[2] for s in sorted(scores, key=lambda s: -s[0] * s[1])[:3]]
print("brain channels:", top)
(ROOT / "targets" / "data" / "liver_brain.json").write_text(json.dumps({"brain_channels": top}))
