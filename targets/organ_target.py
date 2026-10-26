"""Voxel targets for the heart, a kidney and a lung, built from the BodyParts3D surfaces
in web/public/models/body.glb (same method as liver_target.py).

Each organ: the mesh is voxelised in a cube that fits it, a vascular tree (C) is grown
from the hilum with space colonisation, and a drainage tree (D) from a second root
(coronary veins / ureter / main bronchus). Kidneys and lungs come in pairs: one side is
built and trained, the viewer mirrors it for the other side.

    python targets/organ_target.py heart kidney lung
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

from liver_target import N, ROOT, voxelise
from organs import (CAPSULE, DRAIN, EMPTY, PARENCHYMA, VESSEL, finalize, rasterize_polyline,
                    rasterize_tree, space_colonization, tree_radii)

ORGANS = {
    # side: direction (body axes, x = patient left/right, y up, z front) the hilum faces
    "heart": dict(mesh="heart", side=(0.0, 1.0, 0.1), drain_offset=(0.22, -0.02, -0.05),
                  drain_out=(0.0, 1.0, -0.2), points=420, inner_erode=1,
                  label="BodyParts3D heart walls (CC BY 4.0) + procedural coronary vessels"),
    "kidney": dict(mesh="kidney_left", side=(1.0, 0.0, 0.35), drain_offset=(0.0, -0.16, 0.0),
                   drain_out=(0.3, -1.0, 0.0), points=240,
                   label="BodyParts3D left kidney (CC BY 4.0) + procedural renal vessels and pelvis"),
    "lung": dict(mesh="lung_right", side=(1.0, 0.0, 0.0), drain_offset=(0.0, 0.12, -0.04),
                 drain_out=(0.5, 1.0, 0.0), points=380,
                 label="BodyParts3D bronchial tree envelope (CC BY 4.0) + procedural vessels and airways"),
}


def load_mesh(name: str):
    import trimesh
    scene = trimesh.load(ROOT / "web" / "public" / "models" / "body.glb", force="scene")
    for gname, geom in scene.geometry.items():
        if gname == name or gname.startswith(name):
            g = geom.copy()
            if gname in scene.graph.nodes:
                g.apply_transform(scene.graph.get(gname)[0])
            return g
    raise RuntimeError(f"{name} not found in body.glb")


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def build(organ: str, seed: int = 11):
    cfg = ORGANS[organ]
    rng = np.random.default_rng(seed)
    mesh = load_mesh(cfg["mesh"])
    lo, hi_b = mesh.bounds
    center = (lo + hi_b) / 2
    half = float((hi_b - lo).max() / 2 * 1.10)
    body = voxelise(mesh, center, half, N)
    lbl, k = ndimage.label(body)
    if k > 1:
        body = lbl == (np.argmax(ndimage.sum(body, lbl, range(1, k + 1))) + 1)

    surf = body & ~ndimage.binary_erosion(body)
    s_loc = (np.argwhere(surf)[:, ::-1] + 0.5) / N * 2 - 1
    alive_loc = (np.argwhere(body)[:, ::-1] + 0.5) / N * 2 - 1
    mid = alive_loc.mean(0)
    side = unit(cfg["side"])
    rel = s_loc - mid
    along = rel @ side
    perp = np.linalg.norm(rel - along[:, None] * side, axis=1)
    hilum = s_loc[np.argmax(along - 0.8 * perp)]          # furthest out on that side, near its middle
    t2 = hilum + np.asarray(cfg["drain_offset"])
    drain_root = s_loc[np.argmin(np.linalg.norm(s_loc - t2, axis=1))]

    inner = ndimage.binary_erosion(body, iterations=cfg.get("inner_erode", 2))
    if cfg.get("surface_tree"):
        inner &= ~ndimage.binary_erosion(body, iterations=6)

    def inside(pts):
        v = np.clip(((pts + 1) * 0.5 * N).astype(int), 0, N - 1)
        return inner[v[:, 2], v[:, 1], v[:, 0]]

    hdir = -side
    root = hilum - hdir * 0.10
    tree = space_colonization(inside, root, hdir, cfg["points"], rng, step=0.04, kill=0.09, trunk_len=0.14)
    vessel = rasterize_tree(tree, tree_radii(tree, r_leaf=0.8, r_max=2.1), N)

    ddir = unit(mid - drain_root)
    droot = drain_root - ddir * 0.06
    dtree = space_colonization(inside, droot, ddir, 60, rng, step=0.045, influence=0.5, kill=0.15,
                               max_iter=70, trunk_len=0.16)
    drain = rasterize_tree(dtree, tree_radii(dtree, r_leaf=0.9, r_max=2.2, exponent=2.0), N) & ~vessel
    # short stubs outside the organ where the vessels / drain leave it
    stub = rasterize_polyline([root, root - hdir * 0.12], 2.0, N)
    vessel |= stub
    dstub = rasterize_polyline([droot, droot + unit(cfg["drain_out"]) * 0.12], 2.2, N)
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

    # capsule only on the outer surface (not on the walls of inner cavities such as heart chambers)
    empty_lbl, _ = ndimage.label(~alive)
    border = np.unique(np.concatenate([empty_lbl[[0, -1]].ravel(), empty_lbl[:, [0, -1]].ravel(), empty_lbl[:, :, [0, -1]].ravel()]))
    exterior = np.isin(empty_lbl, border[border > 0])
    shell = alive & ndimage.binary_dilation(exterior, iterations=1)
    labels = np.full((N, N, N), EMPTY, np.uint8)
    labels[alive] = PARENCHYMA
    labels[shell] = CAPSULE
    open_shell = near_hilum | near_drain | ~body | (shell if cfg.get("surface_tree") else np.zeros_like(shell))
    labels[drain & (~shell | open_shell)] = DRAIN
    labels[vessel & (~shell | open_shell)] = VESSEL

    labels, meta = finalize(organ, labels, hv, {})
    meta["world"] = dict(center=[float(c) for c in center], half=half)
    meta["mesh"] = cfg["mesh"]
    meta["source"] = cfg["label"]
    return labels, meta


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "eval"))
    from metrics import function_score

    out = ROOT / "targets" / "data"
    for organ in sys.argv[1:] or list(ORGANS):
        labels, meta = build(organ)
        np.save(out / f"{organ}.npy", labels)
        (out / f"{organ}.json").write_text(json.dumps(meta, indent=2))
        print(organ, meta["counts"], "seed", meta["seed"], "hilum", meta["hilum"])
        print("  ", {k: round(v, 3) for k, v in function_score(labels, labels, meta["hilum"]).items()})
