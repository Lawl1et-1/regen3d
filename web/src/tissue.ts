// Geometry helpers shared by Lab and Slice modes: picking the tissue surface,
// converting world-space cuts into voxel-space tool operations.
import { invert, transformDir, transformPoint, v3, type M4, type V3 } from './math';
import { TOOL_PLANE, type ToolOp } from './sim';

export interface Hit { local: V3; vox: V3; world: V3; t: number }

export const localToVox = (p: V3, n: number): V3 => p.map((c) => (c + 1) * n * 0.5 - 0.5) as V3;
export const voxToLocal = (p: V3, n: number): V3 => p.map((c) => ((c + 0.5) / n) * 2 - 1) as V3;

/** March a world ray through the volume's last known class grid; first alive voxel wins. */
export function raycastTissue(summary: Uint32Array | null, n: number, toWorld: M4, o: V3, d: V3): Hit | null {
  if (!summary) return null;
  const inv = invert(toWorld);
  const ol = transformPoint(inv, o);
  const dlu = transformDir(inv, d);
  const scale = v3.len(dlu);
  const dl = v3.scale(dlu, 1 / scale);
  let tn = 0, tf = 1e9;
  for (let k = 0; k < 3; k++) {
    const a = (-1 - ol[k]) / dl[k], b = (1 - ol[k]) / dl[k];
    tn = Math.max(tn, Math.min(a, b)); tf = Math.min(tf, Math.max(a, b));
  }
  if (tn >= tf) return null;
  const step = (2 / n) * 0.3;
  for (let t = tn; t < tf; t += step) {
    const p = v3.add(ol, v3.scale(dl, t));
    const g = localToVox(p, n).map(Math.round);
    if (g.some((c) => c < 0 || c >= n)) continue;
    if (summary[g[0] + g[1] * n + g[2] * n * n] & 0xff) {
      return { local: p, vox: localToVox(p, n), world: transformPoint(toWorld, p), t: t / scale };
    }
  }
  return null;
}

/** A world plane (normal + point) in the volume's local and voxel spaces. */
export function planeToVolume(nw: V3, pw: V3, toWorld: M4, n: number) {
  const inv = invert(toWorld);
  const nl = v3.norm(transformDir(inv, nw));
  const pl = transformPoint(inv, pw);
  const dLocal = v3.dot(nl, pl);
  const dVox = v3.dot(nl, localToVox(pl, n));
  return { nl, dLocal, dVox };
}

/** Count alive voxels on the positive side of a voxel-space plane. */
export function sideCounts(summary: Uint32Array, n: number, nv: V3, d: number): [number, number] {
  let pos = 0, neg = 0;
  for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    if (!(summary[x + y * n + z * n * n] & 0xff)) continue;
    if (nv[0] * x + nv[1] * y + nv[2] * z > d) pos++; else neg++;
  }
  return [pos, neg];
}

/** Tool op that removes the positive side (dot(p, n) > d) in voxel space. */
export function planeOp(nv: V3, d: number): ToolOp {
  return { kind: TOOL_PLANE, p0: [0, 0, 0, 0], p1: [nv[0], nv[1], nv[2], d] };
}

/** Plane through the camera containing two screen rays (a slice stroke). */
export function strokePlane(eye: V3, da: V3, db: V3): { n: V3; p: V3 } | null {
  const nrm = v3.cross(da, db);
  if (v3.len(nrm) < 1e-5) return null;
  return { n: v3.norm(nrm), p: eye };
}
