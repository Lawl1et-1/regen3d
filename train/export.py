"""Export trained weights + target labels for the web viewer.

    python train/export.py liver

Writes web/public/organs/<organ>.json (meta + weights) and <organ>.labels.bin
(uint8, index = x + y*N + z*N*N, same layout as the WebGPU state buffer).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def export(organ: str):
    ck = torch.load(ROOT / "train" / "runs" / organ / "model.pt", map_location="cpu")
    sd = ck["model"]
    w1 = sd["w1.weight"].reshape(sd["w1.weight"].shape[0], -1).numpy()  # (hidden, 4C)
    b1 = sd["w1.bias"].numpy()
    w2 = sd["w2.weight"].reshape(sd["w2.weight"].shape[0], -1).numpy()  # (C, hidden)
    meta = json.loads((ROOT / "targets" / "data" / f"{organ}.json").read_text())
    labels = np.load(ROOT / "targets" / "data" / f"{organ}.npy")  # (z, y, x)
    out = ROOT / "web" / "public" / "organs"
    out.mkdir(parents=True, exist_ok=True)
    meta.update(
        hidden=int(w1.shape[0]), fire_rate=0.5, trained_iters=int(ck["iter"]) + 1,
        w1=[round(float(v), 6) for v in w1.ravel()],
        b1=[round(float(v), 6) for v in b1.ravel()],
        w2=[round(float(v), 6) for v in w2.ravel()],
    )
    (out / f"{organ}.json").write_text(json.dumps(meta))
    labels.astype(np.uint8).tofile(out / f"{organ}.labels.bin")
    print("exported", organ, "hidden", w1.shape[0], "iters", meta["trained_iters"])


LABELS = {"blob": "Blob", "liver": "Liver", "kidney": "Kidney", "heart": "Heart", "lung": "Lung"}


def write_index():
    out = ROOT / "web" / "public" / "organs"
    items = []
    for p in sorted(out.glob("*.json")):
        if p.stem == "index":
            continue
        m = json.loads(p.read_text())
        items.append(dict(name=p.stem, label=LABELS.get(p.stem, p.stem.title()), n=m["n"]))
    (out / "index.json").write_text(json.dumps(items, indent=1))


if __name__ == "__main__":
    for o in sys.argv[1:] or ["blob"]:
        export(o)
    write_index()
