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
    meta = json.loads((ROOT / "targets" / "data" / f"{organ}.json").read_text())
    ckpt = ROOT / "train" / "runs" / organ / "model.pt"
    if not ckpt.exists():  # target only (no trained model yet)
        return export_target(organ, meta)
    ck = torch.load(ckpt, map_location="cpu")
    sd = ck["model"]
    w1 = sd["w1.weight"].reshape(sd["w1.weight"].shape[0], -1).numpy()  # (hidden, 4C)
    b1 = sd["w1.bias"].numpy()
    w2 = sd["w2.weight"].reshape(sd["w2.weight"].shape[0], -1).numpy()  # (C, hidden)
    labels = np.load(ROOT / "targets" / "data" / f"{organ}.npy")  # (z, y, x)
    out = ROOT / "web" / "public" / "organs"
    out.mkdir(parents=True, exist_ok=True)
    meta.update(
        hidden=int(w1.shape[0]), fire_rate=0.5, trained_iters=int(ck["iter"]) + 1,
        w1=[round(float(v), 6) for v in w1.ravel()],
        b1=[round(float(v), 6) for v in b1.ravel()],
        w2=[round(float(v), 6) for v in w2.ravel()],
    )
    # training checkpoints for "Watch the AI learn": weights only, one small file each
    ckpts = []
    for cp in sorted((ROOT / "train" / "runs" / organ).glob("ckpt_*.pt")):
        c = torch.load(cp, map_location="cpu")["model"]
        it = int(cp.stem.split("_")[1])
        ckpts.append(it)
        (out / f"{organ}.ckpt_{it:05d}.json").write_text(json.dumps(dict(
            iteration=it,
            w1=[round(float(v), 6) for v in c["w1.weight"].reshape(c["w1.weight"].shape[0], -1).numpy().ravel()],
            b1=[round(float(v), 6) for v in c["w1.bias"].numpy().ravel()],
            w2=[round(float(v), 6) for v in c["w2.weight"].reshape(c["w2.weight"].shape[0], -1).numpy().ravel()],
        )))
    meta["checkpoints"] = ckpts
    brain = ROOT / "targets" / "data" / f"{organ}_brain.json"
    if brain.exists():
        meta.update(json.loads(brain.read_text()))
    (out / f"{organ}.json").write_text(json.dumps(meta))
    labels.astype(np.uint8).tofile(out / f"{organ}.labels.bin")
    print("exported", organ, "hidden", w1.shape[0], "iters", meta["trained_iters"])


def export_target(organ: str, meta: dict):
    out = ROOT / "web" / "public" / "organs"
    out.mkdir(parents=True, exist_ok=True)
    labels = np.load(ROOT / "targets" / "data" / f"{organ}.npy")
    meta.update(hidden=0, fire_rate=0.5, trained_iters=0, w1=[], b1=[], w2=[])
    (out / f"{organ}.json").write_text(json.dumps(meta))
    labels.astype(np.uint8).tofile(out / f"{organ}.labels.bin")
    print("exported target only", organ)


LABELS = {"blob": "Blob", "liver": "Liver", "kidney": "Kidney", "heart": "Heart", "lung": "Lung"}


def write_index():
    out = ROOT / "web" / "public" / "organs"
    items = []
    for p in sorted(out.glob("*.json")):
        if p.stem == "index" or ".ckpt_" in p.name:
            continue
        m = json.loads(p.read_text())
        items.append(dict(name=p.stem, label=LABELS.get(p.stem, p.stem.title()), n=m["n"]))
    (out / "index.json").write_text(json.dumps(items, indent=1))


if __name__ == "__main__":
    for o in sys.argv[1:] or ["blob"]:
        export(o)
    write_index()
