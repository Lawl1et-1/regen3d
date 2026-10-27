// Simulated healing potential: how well the trained model repairs one specific injury,
// measured on the injury itself rather than on the whole organ.
//
// Every term is measured from a simulated repair against the healthy organ:
//   T  tissue reconstruction  quality of the wound repair: its voxels back with the correct cell
//                             type / (wound + tissue overgrowing the healthy shape around it)
//   V  vascular recovery      share of the vessels beyond the wound that the injury cut off from
//                             blood and that have it again (the wound's own vessels are part of T)
//   F  blood flow restored    1 - flowCut * (1 - V): the blood supply still missing at the end.
//                             Cutting a trunk that fed 20% of the tree and never reconnecting it
//                             costs 20%, a twig almost nothing: essential vessels weigh more.
//   I  collateral impairment  uninjured functional tissue that ends up without enough blood
//   D  residual damage        voxels of the organ not back to their healthy type (+ overgrowth)
// Per simulated repair:  H = T * F * (1 - I) * (1 - D)
// T is relative to the wound (how well it is rebuilt), 1 - D is the organ's structural integrity
// (how much of it is still wrong): they overlap on purpose, so the same quality of repair weighs
// more on a larger injury. The product is the heuristic part (equal weights, nothing fitted): an
// unrepaired wound gives 0 and rebuilding 25% of a wound can never score above 25%. There are no
// real healing outcomes to calibrate against, so H is a simulated healing potential, not a
// medical or calibrated probability. Uncertainty is reported separately, never folded in.
//
// The REPAIRABLE verdict answers another question (does the simulated repair close the wound and
// bring blood back?) with its own cut-offs, so it does not flip with small changes of H.
import { NUTRIENT_OK, type MetricsEngine } from './metrics';

/** Injuries larger than this share of the organ are beyond those seen in training. */
export const TRAINED_MAX_SHARE = 0.25;
/** Fewer lost vessel voxels than this: no vessel was really cut, V does not apply. */
const MIN_LOST_VESSELS = 6;
const OVERGROWTH_RADIUS = 2;
/** Verdict cut-offs (heuristic): the wound regrows (any cell type) and blood comes back. */
export const REFILL_OK = 0.8;
export const FLOW_OK = 0.8;

export interface HealingContext {
  n: number;
  /** voxels the injury removed (alive in the healthy organ, dead right after it) */
  lesion: Uint8Array;
  lesionCount: number;
  /** empty space near the wound, where tissue growing counts as overgrowth */
  overgrowthZone: Uint8Array;
  /** vessel voxels outside the wound that had blood before the injury and lost it */
  lostVessels: Int32Array;
  vesselsWithBlood: number;
  /** uninjured functional tissue fed by blood in the healthy organ */
  fedOutside: number;
  healthy: Uint32Array;
  organVoxels: number;
  /** severity of the injury */
  share: number;          // of the organ's tissue
  depthVoxels: number;    // deepest point of the lesion below the healthy surface
  flowLost: number;       // share of the perfused vascular tree beyond the wound that lost blood flow
  hilumHit: boolean;
}

export interface OutcomeScore {
  tissue: number;
  vessels: number | null;
  impairment: number;
  residual: number;
  /** share of the wound regrown with any tissue, and blood supply restored (for the verdict) */
  refilled: number;
  flow: number;
  potential: number;
}

export type UncertaintyLevel = 'low' | 'moderate' | 'high';

export interface HealingEstimate {
  /** mean H over the simulated repairs (0..1), shown rounded */
  potential: number;
  min: number;
  max: number;
  tissue: number;
  vessels: number | null;
  impairment: number;
  residual: number;
  refilled: number;
  flow: number;
  /** the simulated repairs close the wound and bring blood back (REFILL_OK, FLOW_OK) */
  repairable: boolean;
  uncertainty: UncertaintyLevel;
  uncertaintyWhy: string[];
  /** display text, rounded to what the evidence supports */
  label: string;
}

/** Everything the scores are measured against, from the healthy and the freshly injured organ. */
export function healingContext(metrics: MetricsEngine, healthy: Uint32Array, injured: Uint32Array,
  hilum: [number, number, number] | null): HealingContext {
  const n = metrics.n, nv = n * n * n;
  const lesion = new Uint8Array(nv);
  let lesionCount = 0, organVoxels = 0;
  for (let i = 0; i < nv; i++) {
    if (healthy[i] & 0xff) {
      organVoxels++;
      if (!(injured[i] & 0xff)) { lesion[i] = 1; lesionCount++; }
    }
  }
  const near = dilate(lesion, n, OVERGROWTH_RADIUS);
  const overgrowthZone = new Uint8Array(nv);
  for (let i = 0; i < nv; i++) if (near[i] && !(healthy[i] & 0xff)) overgrowthZone[i] = 1;

  // blood flow: perfused vessels before and right after the injury
  metrics.compute(healthy);
  const fedOutside = countFedOutside(healthy, lesion);
  const before = Uint8Array.from(metrics.dist, (d) => (d >= 0 ? 1 : 0));
  metrics.compute(injured);
  const lost: number[] = [];
  let withBlood = 0;
  for (let i = 0; i < nv; i++) {
    if (!before[i] || metrics.target[i] !== 3) continue;
    withBlood++;
    if (metrics.dist[i] < 0 && !lesion[i]) lost.push(i);
  }

  // depth: distance (voxels) from the outside of the healthy organ, deepest lesion voxel
  const depth = surfaceDistance(healthy, n);
  let depthVoxels = 0;
  for (let i = 0; i < nv; i++) if (lesion[i] && depth[i] > depthVoxels) depthVoxels = depth[i];

  let hilumHit = false;
  if (hilum) {
    const [hx, hy, hz] = hilum;
    let hit = 0;
    for (let z = hz - 3; z <= hz + 3; z++) for (let y = hy - 3; y <= hy + 3; y++) for (let x = hx - 3; x <= hx + 3; x++) {
      if (x >= 0 && y >= 0 && z >= 0 && x < n && y < n && z < n && lesion[x + y * n + z * n * n]) hit++;
    }
    hilumHit = hit > 6;
  }

  return {
    n, lesion, lesionCount, overgrowthZone, lostVessels: new Int32Array(lost), vesselsWithBlood: withBlood,
    fedOutside, healthy, organVoxels,
    share: lesionCount / Math.max(1, organVoxels), depthVoxels, flowLost: lost.length / Math.max(1, withBlood), hilumHit,
  };
}

/** Score one repaired state (class | nutrient << 8 per voxel). Runs the perfusion BFS on it. */
export function scoreOutcome(ctx: HealingContext, metrics: MetricsEngine, sum: Uint32Array): OutcomeScore {
  metrics.compute(sum);
  const nv = sum.length;
  if (!ctx.lesionCount) return { tissue: 1, vessels: null, impairment: 0, residual: 0, refilled: 1, flow: 1, potential: 1 };
  let correct = 0, over = 0, wrong = 0, alive = 0;
  for (let i = 0; i < nv; i++) {
    const c = sum[i] & 0xff, h = ctx.healthy[i] & 0xff;
    if (ctx.lesion[i] && c) { alive++; if (c === h) correct++; }
    if (ctx.overgrowthZone[i] && c) over++;
    if (h && c !== h) wrong++;
    else if (!h && c && ctx.overgrowthZone[i]) wrong++;
  }
  const tissue = correct / (ctx.lesionCount + over);
  let vessels: number | null = null;
  if (ctx.lostVessels.length >= MIN_LOST_VESSELS) {
    let back = 0;
    for (const i of ctx.lostVessels) if (metrics.dist[i] >= 0) back++;
    vessels = back / ctx.lostVessels.length;
  }
  const impairment = ctx.fedOutside > 0 ? clamp01((ctx.fedOutside - countFedOutside(sum, ctx.lesion)) / ctx.fedOutside) : 0;
  const residual = wrong / Math.max(1, ctx.organVoxels);
  const flow = 1 - ctx.flowLost * (1 - (vessels ?? 1));
  const potential = tissue * flow * (1 - impairment) * (1 - residual);
  return { tissue, vessels, impairment, residual, refilled: alive / ctx.lesionCount, flow, potential };
}

/** Combine the simulated repairs: mean potential, component medians, and a separate uncertainty. */
export function estimate(ctx: HealingContext, scores: OutcomeScore[]): HealingEstimate {
  const hs = scores.map((s) => s.potential);
  const mean = hs.reduce((a, b) => a + b, 0) / Math.max(1, hs.length);
  const min = Math.min(...hs), max = Math.max(...hs);
  const spread = max - min;
  const why: string[] = [];
  const pts = Math.max(1, Math.round(spread * 100));
  if (spread >= 0.05) why.push(`the simulated repairs end ${pts} points apart`);
  else why.push(`the simulated repairs agree within ${pts} point${pts > 1 ? 's' : ''}`);
  const outside = ctx.share > TRAINED_MAX_SHARE;
  if (outside) why.push(`the injury (${Math.round(ctx.share * 100)}% of the organ) is larger than any seen in training`);
  if (ctx.hilumHit) why.push('the injury reaches the hilum, where the vessels enter');
  const uncertainty: UncertaintyLevel = spread >= 0.15 || outside ? 'high' : spread >= 0.05 || ctx.hilumHit ? 'moderate' : 'low';

  // only as precise as the evidence: 5-point steps, a 10-point range when uncertain
  const r5 = (v: number) => Math.round(v * 20) * 5;
  const r10 = (v: number, f: (x: number) => number) => f(v * 10) * 10;
  const label = !ctx.lesionCount ? 'no injury'
    : uncertainty === 'high' ? `${r10(min, Math.floor)}–${Math.max(r10(max, Math.ceil), r10(min, Math.floor) + 10)}%`
      : `≈ ${r5(mean)}%`;
  const vs = scores.map((s) => s.vessels).filter((v): v is number => v !== null);
  return {
    potential: mean, min, max,
    tissue: median(scores.map((s) => s.tissue)),
    vessels: vs.length ? median(vs) : null,
    impairment: median(scores.map((s) => s.impairment)),
    residual: median(scores.map((s) => s.residual)),
    refilled: median(scores.map((s) => s.refilled)),
    flow: median(scores.map((s) => s.flow)),
    repairable: median(scores.map((s) => s.refilled)) >= REFILL_OK && median(scores.map((s) => s.flow)) >= FLOW_OK,
    uncertainty, uncertaintyWhy: why, label,
  };
}

/** Measured reasons a repair falls short (the two most telling). */
export function shortfalls(ctx: HealingContext, e: HealingEstimate): string[] {
  const out: { v: number; text: string }[] = [];
  if (e.tissue < 0.8) out.push({ v: 1 - e.tissue, text: `Only ${Math.round(e.tissue * 100)}% of the removed tissue grows back with the right cell types.` });
  if (e.vessels !== null && e.vessels < 0.8) {
    out.push({ v: 1 - e.vessels, text: ctx.hilumHit
      ? `The injury reaches the hilum: only ${Math.round(e.vessels * 100)}% of the vessels that lost blood flow reconnect.`
      : `Only ${Math.round(e.vessels * 100)}% of the vessels that lost blood flow reconnect.` });
  }
  if (e.residual > 0.05) out.push({ v: e.residual, text: `${Math.round(e.residual * 100)}% of the organ is still not back to its healthy state.` });
  if (e.impairment > 0.1) out.push({ v: e.impairment, text: `${Math.round(e.impairment * 100)}% of the uninjured functional tissue ends up without enough blood.` });
  return out.sort((a, b) => b.v - a.v).slice(0, 2).map((o) => o.text);
}

/** Functional tissue (class 2) outside the wound with enough nutrient to count as fed. */
function countFedOutside(sum: Uint32Array, lesion: Uint8Array) {
  const ok = NUTRIENT_OK * 255;
  let c = 0;
  for (let i = 0; i < sum.length; i++) if (!lesion[i] && (sum[i] & 0xff) === 2 && ((sum[i] >> 8) & 0xff) >= ok) c++;
  return c;
}

function clamp01(v: number) { return Math.min(1, Math.max(0, v)); }

function median(xs: number[]) {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2;
}

/** 6-connected dilation by `r` voxels. */
export function dilate(m: Uint8Array, n: number, r: number): Uint8Array {
  let cur = m;
  for (let it = 0; it < r; it++) {
    const out = new Uint8Array(cur);
    for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const i = x + y * n + z * n * n;
      if (cur[i]) continue;
      if ((x > 0 && cur[i - 1]) || (x < n - 1 && cur[i + 1]) || (y > 0 && cur[i - n]) || (y < n - 1 && cur[i + n]) ||
        (z > 0 && cur[i - n * n]) || (z < n - 1 && cur[i + n * n])) out[i] = 1;
    }
    cur = out;
  }
  return cur;
}

/** 6-connected BFS distance of every organ voxel from the outside (1 = on the surface). */
function surfaceDistance(grid: Uint32Array, n: number): Int32Array {
  const nv = n * n * n, nn = n * n;
  const d = new Int32Array(nv).fill(-1);
  const q = new Int32Array(nv);
  let head = 0, tail = 0;
  for (let i = 0; i < nv; i++) if (!(grid[i] & 0xff)) { d[i] = 0; q[tail++] = i; }
  while (head < tail) {
    const i = q[head++];
    const x = i % n, y = ((i / n) | 0) % n, z = (i / nn) | 0;
    const nb = [x > 0 ? i - 1 : -1, x < n - 1 ? i + 1 : -1, y > 0 ? i - n : -1, y < n - 1 ? i + n : -1, z > 0 ? i - nn : -1, z < n - 1 ? i + nn : -1];
    for (const j of nb) if (j >= 0 && d[j] < 0) { d[j] = d[i] + 1; q[tail++] = j; }
  }
  return d;
}
