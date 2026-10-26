"""Procedural organ targets for REGEN 3D.

Each organ is a voxel grid of labels:
    0 empty, 1 capsule (A), 2 parenchyma (B), 3 vessel (C), 4 drain (D)
plus a small JSON with the seed voxel, the hilum and a few counts.

Shapes are signed distance functions in normalised coordinates [-1, 1]^3
(y up), displaced by smooth noise so they look organic. The vascular tree is
grown with a space-colonization algorithm from the hilum.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

EMPTY, CAPSULE, PARENCHYMA, VESSEL, DRAIN = 0, 1, 2, 3, 4
LABELS = ["empty", "capsule", "parenchyma", "vessel", "drain"]


# ---------------------------------------------------------------- SDF helpers
def grid_coords(n: int) -> np.ndarray:
    c = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    z, y, x = np.meshgrid(c, c, c, indexing="ij")
    return np.stack([x, y, z], axis=-1)  # (z, y, x, 3) with xyz in last axis


def sd_ellipsoid(p, center, radii):
    q = (p - np.asarray(center)) / np.asarray(radii)
    k0 = np.linalg.norm(q, axis=-1)
    k1 = np.linalg.norm(q / np.asarray(radii), axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)


def sd_sphere(p, center, r):
    return np.linalg.norm(p - np.asarray(center), axis=-1) - r


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def value_noise(p, freq, rng_seed):
    """Cheap smooth 3D noise: trilinear interpolation of a random lattice."""
    rng = np.random.default_rng(rng_seed)
    g = 16
    lattice = rng.standard_normal((g, g, g))
    lattice = ndimage.gaussian_filter(lattice, 1.0, mode="wrap")
    lattice /= lattice.std()
    q = (p * freq * 0.5 + 0.5) * (g - 1)
    coords = [q[..., 2].ravel(), q[..., 1].ravel(), q[..., 0].ravel()]
    v = ndimage.map_coordinates(lattice, coords, order=1, mode="wrap")
    return v.reshape(p.shape[:-1])


# ------------------------------------------------------- space colonization
@dataclass
class Tree:
    nodes: list = field(default_factory=list)  # positions
    parent: list = field(default_factory=list)


def space_colonization(inside_fn, root, direction, n_points, rng, step=0.035,
                       influence=0.32, kill=0.07, max_iter=400, trunk_len=0.12):
    """Grow a branching tree from `root` towards attraction points filling the organ."""
    pts = []
    while len(pts) < n_points:
        cand = rng.uniform(-1, 1, size=(n_points * 4, 3))
        cand = cand[inside_fn(cand)]
        pts.extend(cand.tolist())
    pts = np.array(pts[:n_points])

    tree = Tree()
    direction = np.asarray(direction, float)
    direction /= np.linalg.norm(direction)
    pos = np.asarray(root, float)
    tree.nodes.append(pos.copy())
    tree.parent.append(-1)
    # straight trunk entering through the hilum
    for _ in range(int(trunk_len / step)):
        pos = pos + direction * step
        tree.nodes.append(pos.copy())
        tree.parent.append(len(tree.nodes) - 2)

    for _ in range(max_iter):
        if len(pts) == 0:
            break
        nodes = np.array(tree.nodes)
        d = np.linalg.norm(pts[:, None, :] - nodes[None, :, :], axis=-1)
        closest = d.argmin(axis=1)
        close_d = d[np.arange(len(pts)), closest]
        active = close_d < influence
        if not active.any():
            # nothing in reach: extend the node closest to any point
            i = np.unravel_index(d.argmin(), d.shape)
            closest = np.array([i[1]])
            pts_sel = pts[[i[0]]]
        else:
            pts_sel = pts[active]
            closest = closest[active]
        grew = False
        for ni in np.unique(closest):
            dirs = pts_sel[closest == ni] - nodes[ni]
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
            g = dirs.sum(0)
            if np.linalg.norm(g) < 1e-6:
                g = dirs[0]
            g /= np.linalg.norm(g)
            new = nodes[ni] + g * step
            tree.nodes.append(new)
            tree.parent.append(int(ni))
            grew = True
        nodes = np.array(tree.nodes)
        d = np.linalg.norm(pts[:, None, :] - nodes[None, :, :], axis=-1).min(axis=1)
        pts = pts[d > kill]
        if not grew:
            break
    return tree


def tree_radii(tree: Tree, r_leaf: float, r_max: float, exponent=2.5):
    n = len(tree.nodes)
    leaves = np.zeros(n)
    children = [[] for _ in range(n)]
    for i, p in enumerate(tree.parent):
        if p >= 0:
            children[p].append(i)
    order = []
    stack = [0]
    while stack:
        i = stack.pop()
        order.append(i)
        stack.extend(children[i])
    for i in reversed(order):
        leaves[i] = 1.0 if not children[i] else sum(leaves[c] for c in children[i])
    return np.minimum(r_leaf * leaves ** (1.0 / exponent), r_max)


def rasterize_tree(tree: Tree, radii, n: int) -> np.ndarray:
    """Mark voxels within radius of every segment. Coordinates normalised, radii in voxels."""
    out = np.zeros((n, n, n), bool)
    nodes = np.array(tree.nodes)
    vox = (nodes + 1.0) * 0.5 * n - 0.5  # to voxel coords (x, y, z)
    for i, p in enumerate(tree.parent):
        if p < 0:
            continue
        a, b, r = vox[p], vox[i], radii[i]
        lo = np.floor(np.minimum(a, b) - r - 1).astype(int).clip(0, n - 1)
        hi = np.ceil(np.maximum(a, b) + r + 1).astype(int).clip(0, n - 1)
        xs, ys, zs = (np.arange(lo[k], hi[k] + 1) for k in range(3))
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        q = np.stack([xx, yy, zz], -1).astype(float)
        ab = b - a
        t = np.clip(((q - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
        dist = np.linalg.norm(q - (a + t[..., None] * ab), axis=-1)
        sub = out[lo[2]:hi[2] + 1, lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        sub |= dist <= r
    return out


# ------------------------------------------------------------- organ shapes
def liver_sdf(p, seed):
    right = sd_ellipsoid(p, (-0.20, 0.02, 0.0), (0.64, 0.44, 0.52))
    left = sd_ellipsoid(p, (0.42, 0.12, -0.02), (0.46, 0.25, 0.38))
    d = smin(right, left, 0.18)
    d = smax(d, -(p[..., 1] + 0.30 - 0.18 * p[..., 0]), 0.12)  # flattened underside
    d += 0.045 * value_noise(p, 2.2, seed)
    return d


def kidney_sdf(p, seed):
    d = sd_ellipsoid(p, (-0.08, 0.0, 0.0), (0.44, 0.76, 0.36))
    d = smax(d, -sd_ellipsoid(p, (0.44, 0.0, 0.0), (0.26, 0.32, 0.40)), 0.10)
    d += 0.03 * value_noise(p, 2.5, seed)
    return d


def heart_sdf(p, seed):
    # ventricles taper to an apex pointing down-left; atria sit on the broad base
    q = p.copy()
    ang = np.radians(-28)
    x, y = p[..., 0], p[..., 1]
    q[..., 0] = np.cos(ang) * x - np.sin(ang) * y
    q[..., 1] = np.sin(ang) * x + np.cos(ang) * y
    prof = np.clip((q[..., 1] + 0.74) / 0.95, 0, 1)
    r = 0.56 * np.sqrt(prof)
    side = np.linalg.norm(np.stack([q[..., 0], q[..., 2] * 1.18], -1), axis=-1) - r
    vent = smax(side, q[..., 1] - 0.22, 0.18)
    atr_l = sd_ellipsoid(q, (0.26, 0.30, -0.10), (0.30, 0.24, 0.28))
    atr_r = sd_ellipsoid(q, (-0.28, 0.28, 0.02), (0.30, 0.26, 0.30))
    d = smin(vent, smin(atr_l, atr_r, 0.10), 0.14)
    groove = 0.035 * np.exp(-((q[..., 0] - 0.06 + 0.25 * q[..., 1]) ** 2) / 0.004) * (q[..., 2] > 0)
    d = d + groove + 0.025 * value_noise(p, 2.4, seed)
    return d, q


def lung_sdf(p, seed):
    y = p[..., 1]
    t = np.clip((0.78 - y) / 1.55, 0, 1)
    r = 0.16 + 0.46 * np.sqrt(t)
    d = np.linalg.norm(p[..., [0, 2]] - np.array([-0.06, 0.0]), axis=-1) / np.array(1.0) - r
    d = np.maximum(d, np.abs(y) - 0.78)
    d = smax(d, p[..., 0] - 0.30, 0.10)  # flat medial face
    d = smax(d, -sd_sphere(p, (0.0, -1.55, 0.0), 0.88), 0.12)  # diaphragm dome
    d += 0.035 * value_noise(p, 2.4, seed)
    return d


def blob_sdf(p, seed):
    return sd_sphere(p, (0, 0, 0), 0.62) + 0.08 * value_noise(p, 2.0, seed)


# ------------------------------------------------------------- organ builder
ORGANS = {
    "blob": dict(n=32),
    "liver": dict(n=48, hilum=(0.05, -0.40, 0.06), hilum_dir=(0, 1, 0), drain_dir=(0.2, 1, -0.1),
                  points=300, stub=[(0.05, -0.62, 0.10)]),
    "kidney": dict(n=48, hilum=(0.30, 0.0, 0.0), hilum_dir=(-1, 0, 0), drain_dir=(-1, -0.15, 0),
                   points=240, stub=[(0.62, -0.05, 0.0)], drain_stub=[(0.50, -0.20, 0.0), (0.56, -0.70, 0.02)]),
    "heart": dict(n=48, hilum=(0.02, 0.50, 0.0), hilum_dir=(0, -1, 0.15), drain_dir=(0.1, -1, 0),
                  points=300, surface_tree=True,
                  stub=[(-0.10, 0.80, 0.10), (-0.32, 0.86, 0.10)],
                  drain_stub=[(0.05, 0.84, -0.05), (0.30, 0.92, -0.05), (0.50, 0.80, -0.05), (0.56, 0.55, -0.05)]),
    "lung": dict(n=48, hilum=(0.34, 0.05, 0.0), hilum_dir=(-1, 0, 0), drain_dir=(-1, 0.1, 0),
                 points=320, stub=[(0.62, 0.10, 0.05)], drain_stub=[(0.55, 0.25, -0.06), (0.62, 0.80, -0.06)]),
}


def rasterize_polyline(points, radius_vox: float, n: int) -> np.ndarray:
    t = Tree()
    for i, pt in enumerate(points):
        t.nodes.append(np.asarray(pt, float))
        t.parent.append(i - 1)
    return rasterize_tree(t, np.full(len(points), radius_vox), n)


def to_norm(v, n):
    return (np.asarray(v, float) + 0.5) / n * 2.0 - 1.0


def build(name: str, seed: int = 7):
    cfg = ORGANS[name]
    n = cfg["n"]
    rng = np.random.default_rng(seed)
    p = grid_coords(n)
    cavity = np.zeros((n, n, n), bool)

    if name == "blob":
        body = blob_sdf(p, seed) < 0
        labels = np.where(body, PARENCHYMA, EMPTY).astype(np.uint8)
        return finalize(name, labels, None, cfg)

    if name == "liver":
        sdf = liver_sdf(p, seed)
    elif name == "kidney":
        sdf = kidney_sdf(p, seed)
    elif name == "heart":
        sdf, q = heart_sdf(p, seed)
        cav = sd_ellipsoid(q, (0.06, -0.10, 0.0), (0.20, 0.34, 0.20))
        cavity = cav < 0
    else:
        sdf = lung_sdf(p, seed)
    body = (sdf < 0) & ~cavity
    body = ndimage.binary_opening(body, iterations=1)
    lbl, k = ndimage.label(body)
    if k > 1:
        sizes = ndimage.sum(body, lbl, range(1, k + 1))
        body = lbl == (np.argmax(sizes) + 1)

    inner = ndimage.binary_erosion(body, iterations=2)
    if cfg.get("surface_tree"):
        inner &= ~ndimage.binary_erosion(body, iterations=6)

    def inside(pts):
        vox = np.clip(((pts + 1) * 0.5 * n).astype(int), 0, n - 1)
        return inner[vox[:, 2], vox[:, 1], vox[:, 0]]

    # vascular tree (C) grown from the hilum
    hilum = np.asarray(cfg["hilum"], float)
    hdir = np.asarray(cfg["hilum_dir"], float)
    root = hilum - hdir / np.linalg.norm(hdir) * 0.10
    tree = space_colonization(inside, root, hdir, cfg["points"], rng, step=0.04, kill=0.09)
    radii = tree_radii(tree, r_leaf=0.8, r_max=2.1)
    vessel = rasterize_tree(tree, radii, n)

    # drainage vessel (D): a central tube with a few short branches
    ddir = np.asarray(cfg["drain_dir"], float)
    droot = hilum + np.cross(hdir, (0, 0, 1) if abs(hdir[2]) < 0.9 else (1, 0, 0)) * 0.10
    droot = droot - ddir / np.linalg.norm(ddir) * 0.10
    dtree = space_colonization(inside, droot, ddir, max(40, cfg["points"] // 10), rng,
                               step=0.045, influence=0.5, kill=0.16, max_iter=60, trunk_len=0.25)
    dradii = tree_radii(dtree, r_leaf=0.9, r_max=2.2, exponent=2.0)
    drain = rasterize_tree(dtree, dradii, n) & ~vessel
    stub = rasterize_polyline([root] + list(cfg["stub"]), 2.0, n)
    vessel |= stub
    if cfg.get("drain_stub"):
        dstub = rasterize_polyline([droot] + list(cfg["drain_stub"]), 2.2, n)
        drain |= dstub & ~vessel

    alive = body | vessel | drain
    # keep vessels outside the body only near the hilum (the stub)
    hv = np.round((hilum + 1) * 0.5 * n - 0.5).astype(int)
    zz, yy, xx = np.indices((n, n, n))
    near_hilum = (xx - hv[0]) ** 2 + (yy - hv[1]) ** 2 + (zz - hv[2]) ** 2 <= (0.16 * n) ** 2
    alive &= body | near_hilum | stub | drain
    vessel &= alive
    drain &= alive

    shell = alive & ~ndimage.binary_erosion(alive, iterations=1)
    labels = np.full((n, n, n), EMPTY, np.uint8)
    labels[alive] = PARENCHYMA
    labels[shell] = CAPSULE
    # vessels stay under the capsule except at the hilum (and on the heart surface)
    open_shell = near_hilum | (cfg.get("surface_tree", False) & shell) | ~body
    labels[drain & (~shell | open_shell)] = DRAIN
    labels[vessel & (~shell | open_shell)] = VESSEL
    return finalize(name, labels, hv, cfg)


def finalize(name, labels, hilum_vox, cfg):
    n = labels.shape[0]
    alive = labels > 0
    # seed: parenchyma voxel closest to the centre of mass
    com = np.array(ndimage.center_of_mass(alive))
    cand = np.argwhere(labels == PARENCHYMA)
    if len(cand) == 0:
        cand = np.argwhere(alive)
    deep = ndimage.distance_transform_edt(alive)
    score = np.linalg.norm(cand - com, axis=1) - 0.5 * deep[tuple(cand.T)]
    sz, sy, sx = cand[score.argmin()]
    meta = dict(
        name=name, n=int(n), channels=16,
        seed=[int(sx), int(sy), int(sz)],
        hilum=None if hilum_vox is None else [int(v) for v in hilum_vox],
        counts={LABELS[i]: int((labels == i).sum()) for i in range(5)},
        labels=LABELS,
    )
    return labels, meta


def save(name, out_dir: Path):
    labels, meta = build(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", labels)
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return labels, meta


if __name__ == "__main__":
    import sys
    here = Path(__file__).resolve().parent / "data"
    names = sys.argv[1:] or list(ORGANS)
    for nm in names:
        _, m = save(nm, here)
        print(nm, m["counts"], "seed", m["seed"], "hilum", m["hilum"])
