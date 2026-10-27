// CPU-side metrics from the GPU summary (class | nutrient << 8 | activity << 16 per voxel).
// Mirrors eval/metrics.py: IoU, per-type accuracy, function (fed parenchyma / target
// parenchyma), vessel connectivity to the hilum (26-connected BFS). The BFS distance
// becomes the perfusion texture that feeds nutrient diffusion and vessel pulses.
import type { Organ } from './sim';

export const NUTRIENT_OK = 0.12;
const HILUM_R = 3.5;

export interface Metrics {
  iou: number;
  acc: [number, number, number, number];
  func: number;
  connectivity: number;
  vesselPresent: number;
  alive: number;
  aliveTarget: number;
  seal: number;
  activity: number;
  starved: number;
}

export class MetricsEngine {
  readonly n: number;
  readonly target: Uint8Array;
  readonly targetCounts = [0, 0, 0, 0, 0];
  readonly hilum: Int32Array;
  readonly maxDist: number;
  readonly targetSeal: number;
  /** BFS distance from the hilum of every vessel voxel (-1 = no blood) */
  readonly dist: Int32Array;
  private queue: Int32Array;
  readonly perfusion: Uint8Array<ArrayBuffer>;
  private neighbors: Int32Array;

  constructor(organ: Organ) {
    const n = (this.n = organ.meta.n);
    this.target = organ.labels;
    for (const v of this.target) this.targetCounts[v]++;
    this.dist = new Int32Array(n * n * n);
    this.queue = new Int32Array(n * n * n);
    this.perfusion = new Uint8Array(n * n * n * 4);
    const nb: number[] = [];
    for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (dx || dy || dz) nb.push(dx, dy, dz);
    }
    this.neighbors = new Int32Array(nb);
    const h = organ.meta.hilum;
    const hil: number[] = [];
    if (h) {
      const r = Math.ceil(HILUM_R);
      for (let z = h[2] - r; z <= h[2] + r; z++) for (let y = h[1] - r; y <= h[1] + r; y++) for (let x = h[0] - r; x <= h[0] + r; x++) {
        if (x < 0 || y < 0 || z < 0 || x >= n || y >= n || z >= n) continue;
        if ((x - h[0]) ** 2 + (y - h[1]) ** 2 + (z - h[2]) ** 2 <= HILUM_R * HILUM_R) hil.push(x + y * n + z * n * n);
      }
    }
    this.hilum = new Int32Array(hil);
    const cls = new Uint32Array(this.target);
    this.maxDist = Math.max(1, this.bfs(cls));
    this.targetSeal = this.sealOf(cls);
  }

  /** BFS over vessel voxels from the hilum. Returns the max distance; fills this.dist (-1 = unreached). */
  private bfs(cls: Uint32Array): number {
    const n = this.n, nn = n * n;
    const dist = this.dist, q = this.queue, nb = this.neighbors;
    dist.fill(-1);
    let head = 0, tail = 0, maxd = 0;
    for (const i of this.hilum) {
      if ((cls[i] & 0xff) === 3) { dist[i] = 0; q[tail++] = i; }
    }
    while (head < tail) {
      const i = q[head++];
      const x = i % n, y = ((i / n) | 0) % n, z = (i / nn) | 0;
      const d = dist[i] + 1;
      for (let k = 0; k < nb.length; k += 3) {
        const xx = x + nb[k], yy = y + nb[k + 1], zz = z + nb[k + 2];
        if (xx < 0 || yy < 0 || zz < 0 || xx >= n || yy >= n || zz >= n) continue;
        const j = xx + yy * n + zz * nn;
        if (dist[j] < 0 && (cls[j] & 0xff) === 3) { dist[j] = d; q[tail++] = j; if (d > maxd) maxd = d; }
      }
    }
    return maxd;
  }

  private sealOf(cls: Uint32Array): number {
    const n = this.n, nn = n * n;
    let surf = 0, cap = 0;
    for (let z = 1; z < n - 1; z++) for (let y = 1; y < n - 1; y++) for (let x = 1; x < n - 1; x++) {
      const i = x + y * n + z * nn;
      const c = cls[i] & 0xff;
      if (!c) continue;
      if (!(cls[i + 1] & 0xff) || !(cls[i - 1] & 0xff) || !(cls[i + n] & 0xff) || !(cls[i - n] & 0xff) || !(cls[i + nn] & 0xff) || !(cls[i - nn] & 0xff)) {
        surf++;
        if (c === 1 || c >= 3) cap++;
      }
    }
    return surf ? cap / surf : 0;
  }

  compute(sum: Uint32Array): Metrics {
    const t = this.target;
    let inter = 0, uni = 0, alive = 0, fed = 0, starved = 0, ves = 0, act = 0;
    const hit = [0, 0, 0, 0];
    for (let i = 0; i < sum.length; i++) {
      const c = sum[i] & 0xff;
      const tc = t[i];
      if (c) alive++;
      if (c && tc) inter++;
      if (c || tc) uni++;
      if (tc && c === tc) hit[tc - 1]++;
      if (c === 2) {
        if (((sum[i] >> 8) & 0xff) >= NUTRIENT_OK * 255) fed++; else starved++;
      }
      if (c === 3) ves++;
      act += (sum[i] >> 16) & 0xff;
    }
    this.bfs(sum);
    let conn = 0;
    const perf = this.perfusion, dist = this.dist, md = this.maxDist;
    for (let i = 0; i < sum.length; i++) {
      const d = dist[i];
      if (d >= 0) { if (this.target[i] === 3) conn++; perf[4 * i] = 1 + Math.min(254, Math.round((d / md) * 220)); } else perf[4 * i] = 0;
      perf[4 * i + 1] = 0; perf[4 * i + 2] = 0;
    }
    const tc = this.targetCounts;
    return {
      iou: uni ? inter / uni : 0,
      acc: [hit[0] / Math.max(tc[1], 1), hit[1] / Math.max(tc[2], 1), hit[2] / Math.max(tc[3], 1), hit[3] / Math.max(tc[4], 1)],
      func: tc[2] ? fed / tc[2] : (alive ? 1 : 0),
      connectivity: tc[3] ? conn / tc[3] : 1,
      vesselPresent: tc[3] ? ves / tc[3] : 1,
      alive,
      aliveTarget: tc[1] + tc[2] + tc[3] + tc[4],
      seal: this.sealOf(sum) / Math.max(this.targetSeal, 1e-3),
      activity: act / 255 / Math.max(alive, 1),
      starved,
    };
  }
}
