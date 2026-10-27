// Edge cases for the simulated healing potential (src/healing.ts), on the real organ grids.
// Repaired states are built synthetically (no GPU): the scoring, the perfusion BFS and the
// metrics are the ones the viewer runs. Nutrient is approximated as tissue within a few
// voxels of a perfused vessel, which is what the simulated diffusion does.
//   npm test
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { MetricsEngine } from '../src/metrics';
import { estimate, healingContext, scoreOutcome, type HealingContext, type OutcomeScore } from '../src/healing';

const NUTRIENT_REACH = 6;
let failed = 0;

function check(name: string, fn: () => void) {
  try { fn(); console.log(`  ok    ${name}`); } catch (e) { failed++; console.log(`  FAIL  ${name}\n        ${(e as Error).message}`); }
}

function load(name: string) {
  const meta = JSON.parse(readFileSync(`public/organs/${name}.json`, 'utf8'));
  const labels = new Uint8Array(readFileSync(`public/organs/${name}.labels.bin`));
  const metrics = new MetricsEngine({ meta, labels } as never);
  return { n: meta.n as number, hilum: meta.hilum as [number, number, number], labels, metrics };
}

type Organ = ReturnType<typeof load>;

/** class grid -> summary (class | nutrient << 8): tissue near a perfused vessel is fed. */
function summary(o: Organ, cls: Uint8Array): Uint32Array {
  const n = o.n, nn = n * n, nv = cls.length;
  o.metrics.compute(Uint32Array.from(cls));
  const d = new Int32Array(nv).fill(-1);
  const q = new Int32Array(nv);
  let head = 0, tail = 0;
  for (let i = 0; i < nv; i++) if (o.metrics.dist[i] >= 0) { d[i] = 0; q[tail++] = i; }
  while (head < tail) {
    const i = q[head++];
    if (d[i] >= NUTRIENT_REACH) continue;
    const x = i % n, y = ((i / n) | 0) % n, z = (i / nn) | 0;
    for (const j of [x > 0 ? i - 1 : -1, x < n - 1 ? i + 1 : -1, y > 0 ? i - n : -1, y < n - 1 ? i + n : -1, z > 0 ? i - nn : -1, z < n - 1 ? i + nn : -1]) {
      if (j >= 0 && d[j] < 0 && cls[j]) { d[j] = d[i] + 1; q[tail++] = j; }
    }
  }
  const s = new Uint32Array(nv);
  for (let i = 0; i < nv; i++) s[i] = cls[i] | ((d[i] >= 0 ? 255 : 0) << 8);
  return s;
}

const coords = (o: Organ, i: number) => [i % o.n, ((i / o.n) | 0) % o.n, (i / (o.n * o.n)) | 0];

function sphere(o: Organ, c: number[], r: number): Uint8Array {
  const m = new Uint8Array(o.labels.length);
  for (let i = 0; i < m.length; i++) {
    if (!o.labels[i]) continue;
    const [x, y, z] = coords(o, i);
    if ((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2 <= r * r) m[i] = 1;
  }
  return m;
}

function injure(o: Organ, lesion: Uint8Array) {
  const cls = Uint8Array.from(o.labels);
  for (let i = 0; i < cls.length; i++) if (lesion[i]) cls[i] = 0;
  return cls;
}

/** Restore a share of the lesion, from its walls inwards (how tissue regrows). */
function partial(o: Organ, injured: Uint8Array, lesion: Uint8Array, share: number, vesselsAs = 3): Uint8Array {
  const n = o.n, nn = n * n;
  const order: number[] = [];
  const seen = new Uint8Array(lesion.length);
  let front: number[] = [];
  for (let i = 0; i < lesion.length; i++) {
    if (!lesion[i]) continue;
    const x = i % n, y = ((i / n) | 0) % n, z = (i / nn) | 0;
    const nb = [x > 0 ? i - 1 : -1, x < n - 1 ? i + 1 : -1, y > 0 ? i - n : -1, y < n - 1 ? i + n : -1, z > 0 ? i - nn : -1, z < n - 1 ? i + nn : -1];
    if (nb.some((j) => j >= 0 && injured[j])) { front.push(i); seen[i] = 1; }
  }
  while (front.length) {
    order.push(...front);
    const next: number[] = [];
    for (const i of front) {
      const x = i % n, y = ((i / n) | 0) % n, z = (i / nn) | 0;
      for (const j of [x > 0 ? i - 1 : -1, x < n - 1 ? i + 1 : -1, y > 0 ? i - n : -1, y < n - 1 ? i + n : -1, z > 0 ? i - nn : -1, z < n - 1 ? i + nn : -1]) {
        if (j >= 0 && lesion[j] && !seen[j]) { seen[j] = 1; next.push(j); }
      }
    }
    front = next;
  }
  const out = Uint8Array.from(injured);
  const k = Math.round(order.length * share);
  for (let t = 0; t < k; t++) { const i = order[t]; out[i] = o.labels[i] === 3 ? vesselsAs : o.labels[i]; }
  return out;
}

/** Old score: a run "succeeded" when the whole organ was > 90% shape and > 90% connected. */
function oldSuccess(o: Organ, cls: Uint8Array) {
  const m = o.metrics.compute(summary(o, cls));
  return m.iou > 0.9 && m.connectivity > 0.9;
}

function setup(o: Organ, lesion: Uint8Array) {
  const injured = injure(o, lesion);
  const ctx = healingContext(o.metrics, summary(o, o.labels), summary(o, injured), o.hilum);
  const score = (cls: Uint8Array) => scoreOutcome(ctx, o.metrics, summary(o, cls));
  return { injured, ctx, score };
}

function show(name: string, ctx: HealingContext, s: OutcomeScore) {
  const p = (v: number | null) => (v === null ? '  n/a' : `${(v * 100).toFixed(0).padStart(4)}%`);
  console.log(`        ${name.padEnd(30)} H ${p(s.potential)}  T ${p(s.tissue)}  V ${p(s.vessels)}  I ${p(s.impairment)}  D ${p(s.residual)}` +
    `   | injury ${(ctx.share * 100).toFixed(1)}%, depth ${ctx.depthVoxels}, flow cut ${(ctx.flowLost * 100).toFixed(0)}%`);
}

for (const organ of ['liver', 'kidney']) {
  const o = load(organ);
  const n = o.n, nv = o.labels.length;
  console.log(`\n${organ}`);

  // landmarks: a surface voxel far from the hilum, the organ centre, a main vessel near the hilum
  const surf: number[] = [];
  for (let i = 0; i < nv; i++) {
    if (!o.labels[i] || o.labels[i] === 3) continue;
    const [x, y, z] = coords(o, i);
    if ([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]].some(([a, b, c]) => !o.labels[(x + a) + (y + b) * n + (z + c) * n * n])) surf.push(i);
  }
  const dh = (i: number) => { const [x, y, z] = coords(o, i); return Math.hypot(x - o.hilum[0], y - o.hilum[1], z - o.hilum[2]); };
  const far = surf.reduce((a, b) => (dh(b) > dh(a) ? b : a));
  // a main vessel: the small lesion on the tree (under 2% of the organ) that cuts the most blood flow
  o.metrics.compute(Uint32Array.from(o.labels));
  const cands: number[] = [];
  for (let i = 0; i < nv; i++) if (o.labels[i] === 3 && o.metrics.dist[i] >= 3 && o.metrics.dist[i] <= 16) cands.push(i);
  let trunk = cands[0], best = -1;
  for (const i of cands.filter((_, k) => k % Math.ceil(cands.length / 120) === 0)) {
    const les = sphere(o, coords(o, i), 4);
    let size = 0; for (const v of les) size += v;
    const { ctx } = setup(o, les);
    if (size < 0.03 * o.labels.filter(Boolean).length && ctx.flowLost > best) { best = ctx.flowLost; trunk = i; }
  }
  let cx = 0, cy = 0, cz = 0, cnt = 0;
  for (let i = 0; i < nv; i++) if (o.labels[i]) { const [x, y, z] = coords(o, i); cx += x; cy += y; cz += z; cnt++; }
  const centre = [cx / cnt, cy / cnt, cz / cnt];

  check('no injury: potential 100%, labelled as no injury', () => {
    const { ctx, score } = setup(o, new Uint8Array(nv));
    const s = score(o.labels);
    assert.equal(ctx.lesionCount, 0);
    assert.equal(s.potential, 1);
    assert.equal(estimate(ctx, [s]).label, 'no injury');
  });

  const shallow = sphere(o, coords(o, far), 3);
  const sh = setup(o, shallow);
  check('small superficial cut, not repaired: near 0 (the old score passed it)', () => {
    const s = sh.score(sh.injured);
    show('superficial, no repair', sh.ctx, s);
    assert.ok(s.potential <= 0.05, `potential ${s.potential}`);
    assert.ok(oldSuccess(o, sh.injured), 'the old whole-organ score should have counted this as a success');
  });
  check('small superficial cut, fully repaired: near 100%', () => {
    const s = sh.score(o.labels);
    show('superficial, full repair', sh.ctx, s);
    assert.ok(s.potential >= 0.95, `potential ${s.potential}`);
  });

  // deep cut: a slab from one side of the organ down past its centre
  const deep = new Uint8Array(nv);
  for (let i = 0; i < nv; i++) {
    const [x, y] = coords(o, i);
    if (o.labels[i] && Math.abs(x - centre[0]) < 2.5 && y > centre[1] - 4) deep[i] = 1;
  }
  const dp = setup(o, deep);
  check('deep cut: deeper and more blood flow cut than the superficial one', () => {
    assert.ok(dp.ctx.depthVoxels > sh.ctx.depthVoxels + 3, `depth ${dp.ctx.depthVoxels} vs ${sh.ctx.depthVoxels}`);
    assert.ok(dp.ctx.flowLost > sh.ctx.flowLost, `flow ${dp.ctx.flowLost} vs ${sh.ctx.flowLost}`);
  });
  check('deep cut: none < half < full repair, and no repair scores 0', () => {
    const none = dp.score(dp.injured), half = dp.score(partial(o, dp.injured, deep, 0.5)), full = dp.score(o.labels);
    show('deep, no repair', dp.ctx, none);
    show('deep, half repaired', dp.ctx, half);
    show('deep, full repair', dp.ctx, full);
    assert.ok(none.potential <= 0.02, `none ${none.potential}`);
    assert.ok(half.potential > none.potential && half.potential < full.potential);
    assert.ok(full.potential >= 0.95, `full ${full.potential}`);
  });
  check('half repaired: a deep cut scores lower than a superficial one', () => {
    const d = dp.score(partial(o, dp.injured, deep, 0.5)).potential;
    const s = sh.score(partial(o, sh.injured, shallow, 0.5)).potential;
    assert.ok(d < s, `deep ${d} vs superficial ${s}`);
  });
  check('incomplete repair: potential rises steadily with the share rebuilt', () => {
    const hs = [0, 0.25, 0.5, 0.75, 1].map((f) => dp.score(partial(o, dp.injured, deep, f)).potential);
    console.log(`        deep cut, 0/25/50/75/100% rebuilt   H ${hs.map((h) => `${Math.round(h * 100)}%`).join(' / ')}`);
    for (let i = 1; i < hs.length; i++) assert.ok(hs[i] >= hs[i - 1] - 1e-9, hs.join(', '));
  });

  // severed vessels: a small lesion on a main vessel near the hilum
  const sv = sphere(o, coords(o, trunk), 4);
  const sev = setup(o, sv);
  check('severed main vessel, tissue regrown but vessel not: clearly lower, though the organ looks whole', () => {
    const s = sev.score(partial(o, sev.injured, sv, 1, 2));
    show('severed vessel, tissue only', sev.ctx, s);
    const m = o.metrics.compute(summary(o, partial(o, sev.injured, sv, 1, 2)));
    assert.ok(sev.ctx.flowLost > 0.05, `flow cut ${sev.ctx.flowLost}`);
    assert.ok(m.iou > 0.97, `whole-organ shape ${m.iou}`);
    assert.ok(s.vessels !== null && s.vessels < 0.2, `vessels ${s.vessels}`);
    // clearly below the same injury reconnected, even though the old score saw a whole organ
    assert.ok(s.potential < sev.score(o.labels).potential - 0.15, `potential ${s.potential}`);
  });
  check('severed main vessel, reconnected: high', () => {
    const s = sev.score(o.labels);
    show('severed vessel, reconnected', sev.ctx, s);
    assert.ok(s.potential >= 0.95, `potential ${s.potential}`);
  });

  check('no cliff: scores change smoothly between neighbouring outcomes', () => {
    const hs = [0.7, 0.75, 0.8].map((f) => dp.score(partial(o, dp.injured, deep, f)).potential);
    for (let i = 1; i < hs.length; i++) assert.ok(Math.abs(hs[i] - hs[i - 1]) < 0.25, hs.join(', '));
  });
  check('uncertain predictions: half the repairs fail -> mid score, high uncertainty, shown as a range', () => {
    const full = dp.score(o.labels), none = dp.score(dp.injured);
    const e = estimate(dp.ctx, [full, full, full, full, none, none, none, none]);
    console.log(`        4 full + 4 failed repairs           ${e.label}, uncertainty ${e.uncertainty}`);
    assert.ok(e.potential > 0.4 && e.potential < 0.6);
    assert.equal(e.uncertainty, 'high');
    assert.ok(e.label.includes('–'), e.label);
  });
  check('consistent predictions: low uncertainty, rounded to 5 points', () => {
    const s = sh.score(o.labels);
    const e = estimate(sh.ctx, Array(8).fill(s));
    assert.equal(e.uncertainty, 'low');
    assert.match(e.label, /^≈ \d+%$/);
    assert.equal(Number(e.label.replace(/\D/g, '')) % 5, 0);
  });
  check('overgrowth outside the healthy shape lowers tissue reconstruction', () => {
    const cls = Uint8Array.from(o.labels);
    for (let i = 0; i < nv; i++) if (sh.ctx.overgrowthZone[i]) cls[i] = 2;
    assert.ok(sh.score(cls).tissue < 0.9);
  });
}

console.log(failed ? `\n${failed} check(s) failed` : '\nall checks passed');
process.exit(failed ? 1 : 0);
