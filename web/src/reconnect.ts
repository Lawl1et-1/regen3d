// Reconnection pulses: when a vessel stump that was cut off from the blood supply links up
// with the tree again, a light pulse travels from the junction along everything that just got
// its blood back. Found on the CPU by comparing two successive perfusion BFS results, written
// into the perfusion texture: g = distance from the junction (voxels), b = event slot.
import type { MetricsEngine } from './metrics';

export const PULSE_SLOTS = 4;
/** stump voxels (vessels that existed but had no blood) needed to call it a reconnection, not tip growth */
const MIN_STUMP = 3;

export class ReconnectPulses {
  private perfused: Uint8Array;
  private wasVessel: Uint8Array;
  private rel: Uint8Array;
  private slot: Uint8Array;
  /** start time of the pulse in each slot (-1 = none) */
  readonly times = [-1, -1, -1, -1];
  private next = 0;
  private armed = false;

  constructor(private metrics: MetricsEngine) {
    const nv = metrics.n ** 3;
    this.perfused = new Uint8Array(nv);
    this.wasVessel = new Uint8Array(nv);
    this.rel = new Uint8Array(nv);
    this.slot = new Uint8Array(nv);
  }

  /** Start watching from `sum` (the wounded organ right before the repair). */
  arm(sum: Uint32Array) {
    this.metrics.compute(sum);
    this.snapshot(sum);
    this.clear();
    this.armed = true;
  }

  disarm() {
    this.armed = false;
    this.clear();
  }

  get lastStart() { return Math.max(...this.times); }

  /** Call right after metrics.compute(sum): detects reconnections and fills g/b of the perfusion. */
  update(sum: Uint32Array, time: number) {
    if (this.armed) {
      const dist = this.metrics.dist;
      const fresh: number[] = [];
      let stumps = 0;
      for (let i = 0; i < dist.length; i++) {
        if (dist[i] >= 0 && !this.perfused[i]) { fresh.push(i); if (this.wasVessel[i]) stumps++; }
      }
      if (stumps >= MIN_STUMP) this.event(fresh, time);
      this.snapshot(sum);
    }
    const perf = this.metrics.perfusion, rel = this.rel, slot = this.slot;
    for (let i = 0; i < rel.length; i++) { perf[4 * i + 1] = rel[i]; perf[4 * i + 2] = slot[i] * 60; }
  }

  private event(fresh: number[], time: number) {
    const s = this.next;
    this.next = (s + 1) % PULSE_SLOTS;
    for (let i = 0; i < this.slot.length; i++) if (this.slot[i] === s + 1) { this.slot[i] = 0; this.rel[i] = 0; }
    const n = this.metrics.n, nn = n * n, dist = this.metrics.dist;
    // upstream first: each voxel is one further from the junction than its closest upstream neighbour
    fresh.sort((a, b) => dist[a] - dist[b]);
    for (const i of fresh) {
      const x = i % n, y = ((i / n) | 0) % n, z = (i / nn) | 0;
      let best = 255;
      for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy, zz = z + dz;
        if (xx < 0 || yy < 0 || zz < 0 || xx >= n || yy >= n || zz >= n) continue;
        const j = xx + yy * n + zz * nn;
        if (dist[j] !== dist[i] - 1) continue;
        const r = this.perfused[j] ? 0 : this.slot[j] === s + 1 ? this.rel[j] : 255;
        if (r < best) best = r;
      }
      this.rel[i] = best >= 255 ? 0 : Math.min(250, best + 1);
      this.slot[i] = s + 1;
    }
    this.times[s] = time;
  }

  private snapshot(sum: Uint32Array) {
    const dist = this.metrics.dist;
    for (let i = 0; i < dist.length; i++) {
      this.perfused[i] = dist[i] >= 0 ? 1 : 0;
      this.wasVessel[i] = (sum[i] & 0xff) === 3 ? 1 : 0;
    }
  }

  private clear() {
    this.rel.fill(0);
    this.slot.fill(0);
    this.times.fill(-1);
    this.next = 0;
  }
}
