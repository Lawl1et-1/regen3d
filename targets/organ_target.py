"""Voxel targets for the heart, a kidney and a lung, built from the BodyParts3D surfaces
in web/public/models/body.glb (same method as liver_target.py).

Each organ: the mesh is voxelised in a cube that fits it, a vascular tree (C) is grown
from the hilum with space colonisation, and a drainage tree (D) from a second root
(coronary veins / ureter / main bronchus). Kidneys and lungs come in pairs: one side is
built and trained, the viewer mirrors it for the other side.

    python targets/organ_target.py heart kidney lung stomach spleen pancreas brain
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

# Cell types (label 1..4) of each organ: name and X-ray colour. Label 3 is always the vascular
# tree (blood comes in through it), the others depend on the organ.
TYPES = {
    "liver": [("capsule", "#f2d27a"), ("tissue", "#d9705f"), ("vessel tree", "#ff2d48"), ("central vein", "#5b7cff")],
    "heart": [("epicardium", "#f2d27a"), ("muscle wall", "#e0485a"), ("coronary arteries", "#ff9a3c"), ("coronary veins", "#5b7cff")],
    "kidney": [("cortex", "#f0a35e"), ("medulla", "#a8466b"), ("vessels", "#ff2d48"), ("ureter exit", "#f5e06b")],
    "lung": [("pleura", "#9fd8ff"), ("tissue", "#f2a0b0"), ("vessels", "#ff2d48"), ("bronchial tree", "#7cf0c8")],
    "bone": [("cortical shell", "#f4f1e6"), ("spongy bone", "#d9b98a"), ("vessels", "#ff2d48"), ("marrow", "#c4364f")],
    "skin": [("epidermis", "#f7d9c4"), ("dermis", "#e08c84"), ("small vessels", "#ff2d48"), ("(unused)", "#888888")],
    "stomach": [("serosa", "#f2d27a"), ("wall", "#e07a6a"), ("vessels", "#ff2d48"), ("veins", "#5b7cff")],
    "spleen": [("capsule", "#f2d27a"), ("pulp", "#b8405e"), ("vessels", "#ff2d48"), ("veins", "#5b7cff")],
    "pancreas": [("capsule", "#f2d27a"), ("tissue", "#e8b07a"), ("vessels", "#ff2d48"), ("pancreatic duct", "#7cf0c8")],
    "brain": [("cortex", "#d8a0c0"), ("white matter", "#f1e4ea"), ("vessels", "#ff2d48"), ("venous sinuses", "#5b7cff")],
}

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
    "stomach": dict(mesh="stomach", side=(-1.0, 0.5, 0.2), drain_offset=(0.05, 0.22, 0.0),
                    drain_out=(0.0, 1.0, -0.2), points=320,
                    label="BodyParts3D stomach (CC BY 4.0) + procedural gastric vessels"),
    "spleen": dict(mesh="spleen", side=(-1.0, 0.0, 0.4), drain_offset=(0.0, -0.12, 0.0),
                   drain_out=(-1.0, 0.0, 0.0), points=240,
                   label="BodyParts3D spleen (CC BY 4.0) + procedural splenic vessels"),
    "pancreas": dict(mesh="pancreas", side=(0.0, 0.0, -1.0), drain_offset=(-0.35, -0.05, 0.0),
                     drain_out=(-1.0, -0.3, 0.0), points=280,
                     label="BodyParts3D pancreas (CC BY 4.0) + procedural vessels and pancreatic duct"),
    "bone": dict(mesh="femur_right", side=(0.0, 0.0, -1.0), drain_offset=(0.0, 0.0, 0.0),
                 drain_out=(0.0, 0.0, -1.0), points=200, no_drain=True, crop_top=0.18,
                 label="BodyParts3D right femur, hip end (CC BY 4.0) + procedural cortical/spongy/marrow layers and nutrient artery"),
    "brain": dict(mesh="brain", side=(0.0, -1.0, 0.25), drain_offset=(0.0, 0.55, -0.45),
                  drain_out=(0.0, 0.2, -1.0), points=480,
                  label="BodyParts3D cortex, white matter, cerebellum and brainstem (CC BY 4.0) + procedural vessels"),
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


def voxelise_slab(mesh, center, half, n: int) -> np.ndarray:
    """Like liver_target.voxelise for a mesh that continues outside the box (open at the
    bottom): surface samples inside the box, then each horizontal slice is filled in 2D."""
    hi = n * 2
    pts = mesh.sample(1_500_000)
    loc = (pts - center) / half
    pts = pts[np.all(np.abs(loc) < 1, axis=1)]
    idx = np.clip(np.floor(((pts - center) / half + 1) * 0.5 * hi).astype(int), 0, hi - 1)
    g = np.zeros((hi, hi, hi), bool)
    g[idx[:, 2], idx[:, 1], idx[:, 0]] = True
    g = ndimage.binary_closing(g, iterations=2)
    for y in range(hi):
        g[:, y, :] = ndimage.binary_fill_holes(g[:, y, :])
    g = ndimage.binary_fill_holes(g)
    return g.reshape(n, 2, n, 2, n, 2).mean(axis=(1, 3, 5)) > 0.5


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def build(organ: str, seed: int = 11):
    cfg = ORGANS[organ]
    rng = np.random.default_rng(seed)
    mesh = load_mesh(cfg["mesh"])
    lo, hi_b = mesh.bounds
    if cfg.get("crop_top"):
        # keep the upper part only (e.g. the hip end of the femur): fine enough on a 48^3 grid
        v = mesh.vertices[mesh.vertices[:, 1] > hi_b[1] - cfg["crop_top"]]
        lo, hi_b = v.min(0), v.max(0)
    center = (lo + hi_b) / 2
    half = float((hi_b - lo).max() / 2 * 1.10)
    body = voxelise_slab(mesh, center, half, N) if cfg.get("crop_top") else voxelise(mesh, center, half, N)
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
    if cfg.get("no_drain"):
        drain = np.zeros_like(vessel)
    else:
        dtree = space_colonization(inside, droot, ddir, 60, rng, step=0.045, influence=0.5, kill=0.15,
                                   max_iter=70, trunk_len=0.16)
        drain = rasterize_tree(dtree, tree_radii(dtree, r_leaf=0.9, r_max=2.2, exponent=2.0), N) & ~vessel
    # short stubs outside the organ where the vessels / drain leave it
    stub = rasterize_polyline([root, root - hdir * 0.12], 2.0, N)
    vessel |= stub
    dstub = np.zeros_like(stub) if cfg.get("no_drain") else rasterize_polyline([droot, droot + unit(cfg["drain_out"]) * 0.12], 2.2, N)
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

    labels = layer(organ, labels, body)
    labels, meta = finalize(organ, labels, hv, {})
    meta["world"] = dict(center=[float(c) for c in center], half=half)
    meta["mesh"] = cfg["mesh"]
    meta["source"] = cfg["label"]
    meta["types"] = [dict(name=a, color=b) for a, b in TYPES[organ]]
    return labels, meta


def layer(organ: str, labels: np.ndarray, body: np.ndarray) -> np.ndarray:
    """Organ-specific internal structure on top of the generic labels."""
    tissue = (labels == CAPSULE) | (labels == PARENCHYMA)
    depth = ndimage.distance_transform_edt(labels > 0)          # voxels from the outside
    if organ == "kidney":
        # outer cortex (~3 voxels) around the inner medulla
        labels[tissue & (depth <= 2.2)] = CAPSULE
        labels[tissue & (depth > 2.2)] = PARENCHYMA
    elif organ == "brain":
        # grey-matter cortex (~2 voxels) around the white matter
        labels[tissue & (depth <= 2.2)] = CAPSULE
    elif organ == "bone":
        # thin cortical shell, marrow in the middle of the shaft, spongy bone elsewhere inside
        ys = np.nonzero(body.any(axis=(0, 2)))[0]
        y0, y1 = ys.min(), ys.max()
        yy = np.arange(labels.shape[1])[None, :, None]
        shaft = (yy > y0 + 0.2 * (y1 - y0)) & (yy < y0 + 0.8 * (y1 - y0))
        inner = tissue & (depth > 1.6)
        labels[tissue & (depth <= 1.6)] = CAPSULE
        labels[inner] = PARENCHYMA
        labels[inner & shaft & (depth > 2.4)] = DRAIN
    return labels


def build_skin(seed: int = 11):
    """A small patch of skin: epidermis on top, dermis below with a network of small vessels
    fed by a perforating vessel coming up from underneath."""
    rng = np.random.default_rng(seed)
    n = N
    zz, yy, xx = np.indices((n, n, n))
    c = (np.stack([xx, yy, zz], -1) + 0.5) / n * 2 - 1
    bulge = 0.10 * (1 - c[..., 0] ** 2)                         # gently curved, like a limb
    top = 0.18 + bulge
    inside_xz = (np.abs(c[..., 0]) < 0.82) & (np.abs(c[..., 2]) < 0.82)
    body = inside_xz & (c[..., 1] < top) & (c[..., 1] > top - 0.42)
    epid = body & (c[..., 1] > top - 0.07)
    dermis = body & ~epid

    def inside(pts):
        v = np.clip(((pts + 1) * 0.5 * n).astype(int), 0, n - 1)
        return dermis[v[:, 2], v[:, 1], v[:, 0]] & (pts[:, 1] < (0.18 + 0.10 * (1 - pts[:, 0] ** 2)) - 0.10)

    root = np.array([0.0, -0.40, 0.0])
    tree = space_colonization(inside, root, (0, 1, 0), 260, rng, step=0.04, kill=0.08, trunk_len=0.22)
    vessel = rasterize_tree(tree, tree_radii(tree, r_leaf=0.7, r_max=1.8), n)
    stub = rasterize_polyline([root, root + np.array([0, -0.15, 0])], 1.8, n)
    labels = np.full((n, n, n), EMPTY, np.uint8)
    # the perforating vessel runs from below the patch up into the dermal network
    trunk = vessel & ~body & (c[..., 1] < top - 0.42 + 0.02) & (np.abs(c[..., 0]) < 0.25) & (np.abs(c[..., 2]) < 0.25)
    alive = body | stub | trunk
    labels[dermis] = PARENCHYMA
    labels[epid] = CAPSULE
    labels[(vessel & dermis) | stub | trunk] = VESSEL
    hv = np.round((root + 1) * 0.5 * n - 0.5).astype(int)
    labels, meta = finalize("skin", labels * alive, hv, {})
    meta["world"] = dict(center=[0.0, 0.0, 0.0], half=0.016)    # placed where the user clicks
    meta["mesh"] = "skin"
    meta["source"] = "Procedural skin patch: epidermis, dermis, small-vessel network"
    meta["types"] = [dict(name=a, color=b) for a, b in TYPES["skin"]]
    return labels, meta


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "eval"))
    from metrics import function_score

    out = ROOT / "targets" / "data"
    for organ in sys.argv[1:] or list(ORGANS):
        labels, meta = build_skin() if organ == "skin" else build(organ)
        np.save(out / f"{organ}.npy", labels)
        (out / f"{organ}.json").write_text(json.dumps(meta, indent=2))
        print(organ, meta["counts"], "seed", meta["seed"], "hilum", meta["hilum"])
        print("  ", {k: round(v, 3) for k, v in function_score(labels, labels, meta["hilum"]).items()})
