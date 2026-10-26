"""Build web/public/models/body.glb from BodyParts3D OBJ files.

Groups: skin, muscles, liver, heart, kidney_left, kidney_right, lung_left, lung_right.
BodyParts3D (isa tree) has no lung surface meshes, so each lung is a smooth
envelope grown around its bronchial tree, kept clear of the heart.

Output is in metres, Y up, centred on the body.

Anatomy: BodyParts3D, (c) The Database Center for Life Science, CC BY 4.0.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "assets_src"
OUT = ROOT / "web" / "public" / "models" / "body.glb"

GROUPS = {
    "skin": ["FMA7163"],
    "muscles": ["FMA5022"],
    "liver": ["FMA15739", "FMA15741", "FMA15742", "FMA15747", "FMA15743", "FMA15744", "FMA15745", "FMA15746", "FMA13365"],
    "heart": ["FMA13884", "FMA9531", "FMA9457"],
    "kidney_left": ["FMA7205"],
    "kidney_right": ["FMA7204"],
    "bronchi": ["FMA31739", "FMA7396", "FMA68418"],
}
FACES = {"skin": 28000, "muscles": 45000, "liver": 12000, "heart": 10000,
         "kidney_left": 4000, "kidney_right": 4000, "lung_left": 8000, "lung_right": 8000}


def element_map():
    m = defaultdict(set)
    for line in (SRC / "isa_element_parts.txt").read_text(encoding="utf-8").splitlines()[1:]:
        cid, _, fj = line.split("\t")
        m[cid].add(fj.strip())
    return m


def load_group(concepts, emap) -> trimesh.Trimesh:
    files = sorted(set().union(*(emap[c] for c in concepts)))
    parts = []
    for fj in files:
        p = SRC / "obj" / f"{fj}.obj"
        if p.exists():
            parts.append(trimesh.load(p, force="mesh", process=False))
        else:
            print("  missing", fj)
    mesh = trimesh.util.concatenate(parts)
    mesh.merge_vertices()
    return mesh


def decimate(mesh: trimesh.Trimesh, faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) > faces:
        mesh = mesh.simplify_quadric_decimation(face_count=faces)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


def envelope(points: np.ndarray, avoid: list[trimesh.Trimesh], pitch=4.0, grow=16.0, smooth=2.2):
    """Smooth closed surface around a point cloud (mm), carved away from `avoid` meshes."""
    lo = points.min(0) - grow * 2.5
    hi = points.max(0) + grow * 2.5
    shape = np.ceil((hi - lo) / pitch).astype(int) + 1
    grid = np.zeros(shape, bool)
    idx = np.clip(((points - lo) / pitch).astype(int), 0, shape - 1)
    grid[tuple(idx.T)] = True
    dist = ndimage.distance_transform_edt(~grid) * pitch
    solid = dist < grow
    # union with the convex hull of the tree: lungs are full lobes, not a cloud around the airways
    hull = trimesh.convex.convex_hull(points)
    zz = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3) * pitch + lo
    offs = np.einsum("ij,ij->i", hull.face_normals, hull.triangles[:, 0])
    inside = np.ones(len(zz), bool)
    for nrm, off in zip(hull.face_normals, offs):
        inside &= zz @ nrm <= off
    inside = inside.reshape(shape)
    solid |= ndimage.binary_dilation(inside, iterations=2)
    field = ndimage.gaussian_filter(solid.astype(np.float32), smooth)
    for m in avoid:
        av = np.zeros(shape, bool)
        pts = m.sample(60000)
        ai = ((pts - lo) / pitch).astype(int)
        ok = np.all((ai >= 0) & (ai < shape), 1)
        av[tuple(ai[ok].T)] = True
        av = ndimage.binary_fill_holes(ndimage.binary_dilation(av, iterations=3))
        field = np.minimum(field, ndimage.gaussian_filter((~av).astype(np.float32), 1.5))
    verts, faces, _, _ = measure.marching_cubes(field, 0.5)
    mesh = trimesh.Trimesh(verts * pitch + lo, faces[:, ::-1], process=True)
    trimesh.smoothing.filter_taubin(mesh, iterations=15)
    return mesh


def fuse(mesh: trimesh.Trimesh, pitch=1.6, smooth=1.6) -> trimesh.Trimesh:
    """Merge touching sub-meshes into one seamless closed surface (voxelise, fill, remesh)."""
    pts = mesh.sample(max(400000, len(mesh.faces) * 6))
    lo = pts.min(0) - 6 * pitch
    shape = np.ceil((pts.max(0) + 6 * pitch - lo) / pitch).astype(int) + 1
    grid = np.zeros(shape, bool)
    grid[tuple(((pts - lo) / pitch).astype(int).T)] = True
    grid = ndimage.binary_closing(grid, iterations=2)
    grid = ndimage.binary_fill_holes(grid)
    field = ndimage.gaussian_filter(grid.astype(np.float32), smooth)
    verts, faces, _, _ = measure.marching_cubes(field, 0.5)
    out = trimesh.Trimesh(verts * pitch + lo, faces[:, ::-1], process=True)
    trimesh.smoothing.filter_taubin(out, iterations=10)
    return out


def main():
    emap = element_map()
    meshes = {}
    for name, concepts in GROUPS.items():
        meshes[name] = load_group(concepts, emap)
        print(f"{name}: {len(meshes[name].faces)} faces")

    for k in ("liver", "heart"):
        meshes[k] = fuse(meshes[k])
        print(f"{k}: fused -> {len(meshes[k].faces)} faces")

    # lungs from the bronchial tree, split at the trachea's x
    br = meshes.pop("bronchi")
    pts = br.vertices
    mid_x = meshes["skin"].bounds.mean(0)[0]  # body midline
    for side, sel in (("lung_right", pts[:, 0] < mid_x), ("lung_left", pts[:, 0] >= mid_x)):
        meshes[side] = envelope(pts[sel], [meshes["heart"], meshes["liver"]], grow=12.0, smooth=2.6)
        print(f"{side}: {len(meshes[side].faces)} faces (envelope)")

    for k, m in meshes.items():
        meshes[k] = decimate(m, FACES[k])

    # mm, Z up  ->  metres, Y up, centred on the skin's bounding box
    skin = meshes["skin"]
    c = (skin.bounds[0] + skin.bounds[1]) / 2
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    T = rot @ trimesh.transformations.scale_matrix(0.001) @ trimesh.transformations.translation_matrix(-c)
    scene = trimesh.Scene()
    for k, m in meshes.items():
        m = m.copy()
        m.apply_transform(T)
        _ = m.vertex_normals
        scene.add_geometry(m, node_name=k, geom_name=k)
        print(f"  {k}: {len(m.faces)} faces, bounds {np.round(m.bounds, 3).tolist()}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(scene.export(file_type="glb"))
    print("wrote", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
