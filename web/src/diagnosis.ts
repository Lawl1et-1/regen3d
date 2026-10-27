// AI diagnosis of a lesion: the trained NCA simulates the repair several times
// (it is stochastic, so every rollout differs). Each simulated repair is scored on the
// lesion itself (healing.ts); together they give a simulated healing potential, its
// uncertainty, a predicted repair time and an uncertainty map. Nothing is scripted:
// every number comes from the rollouts.
import type { Metrics, MetricsEngine } from './metrics';
import type { Simulation } from './sim';
import { runWithBlood } from './blood';
import { dilate, estimate, healingContext, scoreOutcome, shortfalls, type HealingContext, type HealingEstimate, type OutcomeScore } from './healing';

export const ROLLOUTS = 8;
export const ROLLOUT_STEPS = 120;
/** Healing potential from which the repair is attempted live (a heuristic cut-off, not a probability). */
export const REPAIR_THRESHOLD = 0.5;
/** Live repair speed: slow while the wound closes so the phases can be followed, then faster. */
export const liveStepsPerSec = (step: number) => (step < 24 ? 2.5 : 8);
/** Seconds the live repair takes to reach `steps`. */
export const liveSeconds = (steps: number) => Math.min(steps, 24) / 2.5 + Math.max(0, steps - 24) / 8;

export interface RolloutResult {
  /** healing potential of this simulated repair >= REPAIR_THRESHOLD */
  success: boolean;
  score: OutcomeScore;
  final: Metrics;
  curve: { step: number; iou: number }[];
  healedAt: number;
}

export interface DiagnosisResult {
  repairable: boolean;
  /** simulated healing potential and its separate uncertainty (healing.ts) */
  healing: HealingEstimate;
  healingCtx: HealingContext;
  predictedSteps: number;
  shape: number;            // median shape recovered (IoU with the healthy organ)
  vessels: number;          // median share of the vascular tree linked to the hilum
  lesionShare: number;      // share of the tissue the lesion removed
  rollouts: RolloutResult[];
  reasons: string[];        // why the repair falls short (measured, not scripted)
  /** rgba8 per voxel: r uncertainty, g critical area, b mean predicted tissue, a mean predicted vessel */
  aux: Uint8Array<ArrayBuffer>;
  /** voxels removed by the lesion, and the part of it open to the outside (the wound mouth) */
  lesion: Uint8Array;
  mouth: Uint8Array;
  /** class grid right after the cut (the starting point of every repair) */
  postCut: Uint32Array;
}

export interface Progress { done: number; running: boolean; rollouts: RolloutResult[] }

/** Yield to rendering between chunks; falls back to a timer when frames are paused (background tab). */
export const nextFrame = () => new Promise<void>((r) => {
  let done = false;
  const go = () => { if (!done) { done = true; r(); } };
  requestAnimationFrame(go);
  setTimeout(go, 60);
});

export class Diagnosis {
  private token = 0;
  progress: Progress = { done: 0, running: false, rollouts: [] };
  result: DiagnosisResult | null = null;
  onProgress?: (p: Progress) => void;
  onResult?: (r: DiagnosisResult) => void;

  constructor(private device: GPUDevice, private main: Simulation, private fc: Simulation,
    private metrics: MetricsEngine, private hilum: [number, number, number] | null) {}

  cancel() { this.token++; this.progress = { done: 0, running: false, rollouts: [] }; this.result = null; }

  /** Analyse the current (cut) state of `main`. `healthy` is the class grid before the lesion. */
  async run(healthy: Uint32Array) {
    const token = ++this.token;
    const n = this.main.n;
    const nv = n * n * n;
    this.result = null;
    this.progress = { done: 0, running: true, rollouts: [] };
    this.onProgress?.(this.progress);

    const alive = new Float32Array(nv);     // how many rollouts end with tissue here
    const vessel = new Float32Array(nv);    // ...with a perfused vessel here
    let lesion: Uint8Array | null = null;
    let postCut: Uint32Array | null = null;
    let ctx: HealingContext | null = null;

    for (let k = 0; k < ROLLOUTS; k++) {
      // fresh copy of the lesioned organ (with its blood supply), with its own random stream
      const enc = this.device.createCommandEncoder();
      this.fc.copyFrom(enc, this.main);
      this.device.queue.submit([enc.finish()]);
      this.fc.blood = this.main.blood;
      this.fc.writeParams(7919 * (k + 1));
      this.fc.setStepIndex(100000 * (k + 1));
      const curve: { step: number; iou: number }[] = [];
      const end = { m: null as Metrics | null, sum: null as Uint32Array | null };
      await runWithBlood(this.device, this.fc, this.metrics, ROLLOUT_STEPS, (s, sum, m) => {
        end.m = m;
        end.sum = sum;
        curve.push({ step: s, iou: m.iou });
        if (k === 0 && s === 0) {
          postCut = sum;
          lesion = new Uint8Array(nv);
          for (let i = 0; i < nv; i++) if ((healthy[i] & 0xff) && !(sum[i] & 0xff)) lesion[i] = 1;
        }
      }, () => token === this.token);
      if (token !== this.token) return;
      const final = end.m!;
      const lastSum = end.sum!;
      // the rollout is healed once it reaches 98% of where it ends up
      const healedAt = curve.find((c) => c.iou >= final.iou * 0.98)?.step ?? ROLLOUT_STEPS;
      const perf = this.metrics.perfusion;
      for (let i = 0; i < nv; i++) {
        if (lastSum[i] & 0xff) alive[i]++;
        if (perf[4 * i]) vessel[i]++;
      }
      // score this simulated repair on the lesion (the BFS runs again, so after reading the perfusion)
      ctx ??= healingContext(this.metrics, healthy, postCut ?? healthy, this.hilum);
      const score = scoreOutcome(ctx, this.metrics, lastSum);
      const success = score.potential >= REPAIR_THRESHOLD;
      this.progress.rollouts.push({ success, score, final, curve, healedAt });
      this.progress.done = k + 1;
      this.onProgress?.(this.progress);
    }

    // ------------------------------------------------------------ aggregate
    const R = this.progress.rollouts;
    const healing = estimate(ctx!, R.map((r) => r.score));
    const repairable = healing.potential >= REPAIR_THRESHOLD;
    const med = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2; };
    const shape = med(R.map((r) => r.final.iou));
    const vessels = med(R.map((r) => r.final.connectivity));
    const predictedSteps = Math.round(med(R.map((r) => r.healedAt)));

    // uncertainty: where rollouts disagree; critical: tissue / vessels that most rollouts never bring back
    const target = this.metrics.target;
    const unc = new Float32Array(nv);
    const crit = new Float32Array(nv);
    for (let i = 0; i < nv; i++) {
      const p = alive[i] / ROLLOUTS;
      const q = vessel[i] / ROLLOUTS;
      let u = 4 * p * (1 - p);
      if (target[i] === 3) u = Math.max(u, 4 * q * (1 - q));
      unc[i] = u;
      if (target[i] && (p < 0.5 || (target[i] === 3 && q < 0.5))) crit[i] = 1;
    }
    // only the wound matters for the critical area (ignore 1-voxel boundary noise)
    if (lesion) {
      const near = dilate(lesion, n, 2);
      for (let i = 0; i < nv; i++) { if (!near[i]) { unc[i] *= 0.3; if (target[i] !== 3) crit[i] = 0; } }
    }
    const uD = blurMax(unc, n, 1), cD = blurMax(crit, n, 2);
    const lesionNear = lesion ? dilate(lesion, n, 1) : new Uint8Array(nv);
    const aux = new Uint8Array(nv * 4);
    for (let i = 0; i < nv; i++) {
      aux[i * 4] = Math.round(Math.min(1, uD[i]) * 255);
      aux[i * 4 + 1] = repairable ? (lesionNear[i] ? 255 : 0) : Math.round(Math.min(1, cD[i]) * 255);
      aux[i * 4 + 2] = Math.round((alive[i] / ROLLOUTS) * 255);
      aux[i * 4 + 3] = Math.round((vessel[i] / ROLLOUTS) * 255);
    }

    // why it falls short: measured facts only, the two most telling
    const reasons = shortfalls(ctx!, healing);
    if (!repairable && !reasons.length) reasons.push(`The simulated repairs disagree: healing potential ${healing.label}.`);

    // wound mouth: lesion voxels within 2 voxels of the outside of the healthy organ
    const outside = new Uint8Array(nv);
    for (let i = 0; i < nv; i++) if (!(healthy[i] & 0xff)) outside[i] = 1;
    const nearOut = dilate(outside, n, 2);
    const les = lesion ?? new Uint8Array(nv);
    const mouth = new Uint8Array(nv);
    for (let i = 0; i < nv; i++) if (les[i] && nearOut[i]) mouth[i] = 1;

    this.result = {
      lesion: les, mouth, postCut: postCut ?? healthy,
      repairable, healing, healingCtx: ctx!,
      predictedSteps, shape, vessels, lesionShare: ctx!.share,
      rollouts: R, reasons, aux,
    };
    this.progress.running = false;
    this.onProgress?.(this.progress);
    this.onResult?.(this.result);
  }
}

/** Max over a (2r+1)^3 box, softened with distance: spreads a value onto the nearby wound walls. */
function blurMax(v: Float32Array, n: number, r: number): Float32Array {
  let cur = v;
  for (let it = 0; it < r; it++) {
    const out = new Float32Array(cur);
    for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const i = x + y * n + z * n * n;
      let m = cur[i];
      if (x > 0) m = Math.max(m, cur[i - 1] * 0.8); if (x < n - 1) m = Math.max(m, cur[i + 1] * 0.8);
      if (y > 0) m = Math.max(m, cur[i - n] * 0.8); if (y < n - 1) m = Math.max(m, cur[i + n] * 0.8);
      if (z > 0) m = Math.max(m, cur[i - n * n] * 0.8); if (z < n - 1) m = Math.max(m, cur[i + n * n] * 0.8);
      out[i] = m;
    }
    cur = out;
  }
  return cur;
}
