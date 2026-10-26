"""Voxel target for the liver, built from the real BodyParts3D liver surface.

The liver mesh in web/public/models/body.glb is voxelised in a cube that fits
it, and a vascular tree (portal side, C) is grown inside from the porta hepatis
on the inferior surface, plus hepatic veins (D) draining up and back towards
the inferior vena cava. The cube's placement is stored in the JSON ("world")
so the viewer can put the volume exactly where the mesh was.

    python targets/liver_target.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage

from organs import (CAPSULE, DRAIN, EMPTY, PARENCHYMA, VESSEL, finalize, rasterize_polyline,
                    rasterize_tree, space_colonization, tree_radii)

ROOT = Path(__file__).resolve().parents[1]
N = 48


def load_liver() -> trimesh.Trimesh:
    scene = trimesh.load(ROOT / "web" / "public" / "models" / "body.glb", force="scene")
    for name, geom in scene.geometry.items():
        if name.startswith("liver"):
            tf = scene.graph.get(name)[0] if name in scene.graph.nodes else np.eye(4)
            g = geom.copy()
            g.apply_transform(tf)
            return g
    raise RuntimeError("liver not found in body.glb")


def voxelise(mesh: trimesh.Trimesh, center, half, n: int) -> np.ndarray:
    """Solid occupancy at 2x resolution, then averaged down to n^3."""
    hi = n * 2
    pts = mesh.sample(1_500_000)
    idx = np.floor(((pts - center) / half + 1) * 0.5 * hi).astype(int)
    idx = np.clip(idx, 0, hi - 1)
    g = np.zeros((hi, hi, hi), bool)
    g[idx[:, 2], idx[:, 1], idx[:, 0]] = True  # (z, y, x)
    g = ndimage.binary_closing(g, iterations=2)
    g = ndimage.binary_fill_holes(g)
    return g.reshape(n, 2, n, 2, n, 2).mean(axis=(1, 3, 5)) > 0.5


def to_local(w, center, half):
    return (np.asarray(w, float) - center) / half


def build(seed: int = 11):
    rng = np.random.default_rng(seed)
    mesh = load_liver()
    lo, hi_b = mesh.bounds
    center = (lo + hi_b) / 2
    half = float((hi_b - lo).max() / 2 * 1.10)
    body = voxelise(mesh, center, half, N)
    lbl, k = ndimage.label(body)
    if k > 1:
        body = lbl == (np.argmax(ndimage.sum(body, lbl, range(1, k + 1))) + 1)

    zz, yy, xx = np.nonzero(body)
    vox = np.stack([xx, yy, zz], 1)
    loc = (vox + 0.5) / N * 2 - 1
    ext_lo, ext_hi = loc.min(0), loc.max(0)
    mid = (ext_lo + ext_hi) / 2
    # porta hepatis: inferior surface, near the middle, slightly posterior
    target = np.array([mid[0] + 0.05, ext_lo[1], mid[2] - 0.15])
    surf = body & ~ndimage.binary_erosion(body)
    s_idx = np.argwhere(surf)[:, ::-1]
    s_loc = (s_idx + 0.5) / N * 2 - 1
    below = np.zeros_like(body)
    below[:, 1:] = body[:, :-1]          # voxel underneath (y - 1) is tissue
    inferior = surf & ~below             # surface facing down
    i_idx = np.argwhere(inferior)[:, ::-1]
    i_loc = (i_idx + 0.5) / N * 2 - 1
    i_loc = i_loc[np.abs(i_loc[:, 0] - mid[0]) < 0.22] if (np.abs(i_loc[:, 0] - mid[0]) < 0.22).any() else i_loc
    hilum = i_loc[np.argmin(np.linalg.norm((i_loc - target) * [1, 0.5, 1], axis=1))]
    # hepatic veins: converge at the superior-posterior border (towards the vena cava)
    t2 = np.array([mid[0] - 0.05, ext_hi[1], ext_lo[2] + 0.1])
    drain_root = s_loc[np.argmin(np.linalg.norm((s_loc - t2) * [1, 2.0, 2.0], axis=1))]

    inner = ndimage.binary_erosion(body, iterations=2)

    def inside(pts):
        v = np.clip(((pts + 1) * 0.5 * N).astype(int), 0, N - 1)
        return inner[v[:, 2], v[:, 1], v[:, 0]]

    hdir = np.array([0.0, 1.0, 0.25])
    hdir /= np.linalg.norm(hdir)
    root = hilum - hdir * 0.10
    tree = space_colonization(inside, root, hdir, 340, rng, step=0.04, kill=0.09, trunk_len=0.14)
    vessel = rasterize_tree(tree, tree_radii(tree, r_leaf=0.8, r_max=2.1), N)

    ddir = np.array([0.0, -1.0, 0.6])
    ddir /= np.linalg.norm(ddir)
    droot = drain_root - ddir * 0.08
    dtree = space_colonization(inside, droot, ddir, 60, rng, step=0.045, influence=0.5, kill=0.15,
                               max_iter=70, trunk_len=0.18)
    drain = rasterize_tree(dtree, tree_radii(dtree, r_leaf=0.9, r_max=2.2, exponent=2.0), N) & ~vessel
    # short stubs outside the organ: portal vein below, vena cava behind
    stub = rasterize_polyline([root, root - hdir * 0.12], 2.0, N)
    vessel |= stub
    dstub = rasterize_polyline([droot, droot - ddir * 0.10], 2.2, N)
    drain |= dstub & ~vessel

    alive = body | stub | dstub
    alive |= (vessel | drain) & body
    vessel &= alive
    drain &= alive
    hv = np.round((hilum + 1) * 0.5 * N - 0.5).astype(int)
    zz, yy, xx = np.indices((N, N, N))
    near_hilum = (xx - hv[0]) ** 2 + (yy - hv[1]) ** 2 + (zz - hv[2]) ** 2 <= (0.12 * N) ** 2
    dv = np.round((drain_root + 1) * 0.5 * N - 0.5).astype(int)
    near_drain = (xx - dv[0]) ** 2 + (yy - dv[1]) ** 2 + (zz - dv[2]) ** 2 <= (0.10 * N) ** 2

    shell = alive & ~ndimage.binary_erosion(alive, iterations=1)
    labels = np.full((N, N, N), EMPTY, np.uint8)
    labels[alive] = PARENCHYMA
    labels[shell] = CAPSULE
    open_shell = near_hilum | near_drain | ~body
    labels[drain & (~shell | open_shell)] = DRAIN
    labels[vessel & (~shell | open_shell)] = VESSEL

    labels, meta = finalize("liver", labels, hv, {})
    meta["world"] = dict(center=[float(c) for c in center], half=half)
    meta["source"] = "BodyParts3D liver surface (CC BY 4.0) + procedural vessels"
    return labels, meta


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "eval"))
    from metrics import function_score

    labels, meta = build()
    out = ROOT / "targets" / "data"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "liver.npy", labels)
    (out / "liver.json").write_text(json.dumps(meta, indent=2))
    print(meta["counts"], "seed", meta["seed"], "hilum", meta["hilum"], "world", meta["world"])
    print({k: round(v, 3) for k, v in function_score(labels, labels, meta["hilum"]).items()})
