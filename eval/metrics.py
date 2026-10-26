"""Geometric and functional metrics for REGEN 3D.

Geometric: 3D IoU of the alive mask, per-cell-type accuracy, Chamfer / Hausdorff
distance between extracted isosurfaces.
Functional: nutrient diffusion from perfused vessels (vessel voxels connected to
the hilum) and the share of parenchyma that receives enough nutrient; vascular
connectivity to the hilum.

The same diffusion rule runs in the viewer (web/src/shaders/fields.wgsl).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

ALIVE_T = 0.1
NUTRIENT_LAMBDA = 0.15
NUTRIENT_OK = 0.12
HILUM_RADIUS = 3.5
STRUCT26 = np.ones((3, 3, 3), bool)


def classify(state: np.ndarray) -> np.ndarray:
    """state (C, n, n, n) -> labels 0..4 (0 = dead)."""
    alive = state[0] > ALIVE_T
    cls = np.argmax(state[1:5], axis=0) + 1
    return np.where(alive, cls, 0).astype(np.uint8)


def iou(labels: np.ndarray, target: np.ndarray) -> float:
    a, b = labels > 0, target > 0
    return float((a & b).sum() / max((a | b).sum(), 1))


def type_accuracy(labels: np.ndarray, target: np.ndarray) -> dict:
    out = {}
    for t, name in [(1, "capsule"), (2, "parenchyma"), (3, "vessel"), (4, "drain")]:
        m = target == t
        out[name] = float((labels[m] == t).mean()) if m.any() else float("nan")
    return out


def hilum_mask(n: int, hilum) -> np.ndarray:
    zz, yy, xx = np.indices((n, n, n))
    hx, hy, hz = hilum
    return (xx - hx) ** 2 + (yy - hy) ** 2 + (zz - hz) ** 2 <= HILUM_RADIUS ** 2


def perfused_vessels(labels: np.ndarray, hilum) -> np.ndarray:
    """Vessel voxels in a 26-connected component touching the hilum."""
    ves = labels == 3
    if hilum is None or not ves.any():
        return np.zeros_like(ves)
    lab, k = ndimage.label(ves, STRUCT26)
    ids = np.unique(lab[hilum_mask(labels.shape[0], hilum) & ves])
    ids = ids[ids > 0]
    return np.isin(lab, ids)


def nutrient_field(labels: np.ndarray, perfused: np.ndarray, iters: int = 400,
                   lam: float = NUTRIENT_LAMBDA) -> np.ndarray:
    """Jacobi iterations of the steady-state diffusion used by the viewer."""
    tissue = labels > 0
    nut = perfused.astype(np.float32)
    t = tissue.astype(np.float32)
    for _ in range(iters):
        s = np.zeros_like(nut)
        tn = nut * t
        s[1:] += tn[:-1]; s[:-1] += tn[1:]
        s[:, 1:] += tn[:, :-1]; s[:, :-1] += tn[:, 1:]
        s[:, :, 1:] += tn[:, :, :-1]; s[:, :, :-1] += tn[:, :, 1:]
        nut = np.where(perfused, 1.0, np.where(tissue, s / (6.0 + lam), 0.0)).astype(np.float32)
    return nut


def function_score(labels: np.ndarray, target: np.ndarray, hilum) -> dict:
    perf = perfused_vessels(labels, hilum)
    nut = nutrient_field(labels, perf)
    par = labels == 2
    fed = (par & (nut >= NUTRIENT_OK)).sum()
    n_target_par = max((target == 2).sum(), 1)
    tv = max((target == 3).sum(), 1)
    return dict(
        function=float(fed / n_target_par),
        fed_of_present=float(fed / max(par.sum(), 1)),
        connectivity=float(perf.sum() / tv),
        vessel_present=float((labels == 3).sum() / tv),
    )


def surface_points(mask: np.ndarray) -> np.ndarray:
    from skimage import measure
    if mask.sum() < 8:
        return np.zeros((0, 3))
    v, *_ = measure.marching_cubes(mask.astype(np.float32), 0.5)
    return v


def surface_distance(labels: np.ndarray, target: np.ndarray) -> dict:
    """Chamfer (mean symmetric) and Hausdorff distances in voxels between alive isosurfaces."""
    from scipy.spatial import cKDTree
    a, b = surface_points(labels > 0), surface_points(target > 0)
    if len(a) == 0 or len(b) == 0:
        return dict(chamfer=float("inf"), hausdorff=float("inf"))
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    return dict(chamfer=float((da.mean() + db.mean()) / 2), hausdorff=float(max(da.max(), db.max())))


def all_metrics(state_or_labels: np.ndarray, target: np.ndarray, hilum, surface=False) -> dict:
    labels = state_or_labels if state_or_labels.ndim == 3 else classify(state_or_labels)
    m = dict(iou=iou(labels, target), alive=int((labels > 0).sum()))
    m.update({f"acc_{k}": v for k, v in type_accuracy(labels, target).items()})
    m.update(function_score(labels, target, hilum))
    if surface:
        m.update(surface_distance(labels, target))
    return m
