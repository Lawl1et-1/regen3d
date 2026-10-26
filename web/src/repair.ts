// Live repair phases, detected from the simulation state (never timed):
//   Sealing    the wound mouth (lesion voxels open to the outside) is closed again
//   Vessels    the vascular tree is linked to the hilum again
//   Refilling  the removed volume is filled with tissue again
//   Finishing  the shape has converged to its final form
import type { DiagnosisResult } from './diagnosis';
import type { Metrics } from './metrics';

export const PHASES = ['Sealing', 'Vessels', 'Refilling', 'Finishing'] as const;
export const REPAIR_MAX_STEPS = 160;

export interface RepairSample { step: number; iou: number; connectivity: number; func: number; mouth: number; fill: number }
export interface RepairState { step: number; done: boolean[]; active: number; sample: RepairSample; finished: boolean }
export interface RepairOutcome { shape: number; vessels: number; healedAt: number; steps: number }

export class RepairTracker {
  readonly samples: RepairSample[] = [];
  readonly done = [false, false, false, false];
  private lesionIdx: Int32Array;
  private mouthIdx: Int32Array;
  finished = false;

  constructor(readonly forecast: DiagnosisResult) {
    const idx = (m: Uint8Array) => {
      const out: number[] = [];
      for (let i = 0; i < m.length; i++) if (m[i]) out.push(i);
      return new Int32Array(out);
    };
    this.lesionIdx = idx(forecast.lesion);
    this.mouthIdx = idx(forecast.mouth);
  }

  /** Feed a measurement (class grid + metrics) taken `step` steps into the repair. */
  update(step: number, sum: Uint32Array, m: Metrics): RepairState {
    const share = (ix: Int32Array) => {
      if (!ix.length) return 1;
      let a = 0;
      for (const i of ix) if (sum[i] & 0xff) a++;
      return a / ix.length;
    };
    const s: RepairSample = { step, iou: m.iou, connectivity: m.connectivity, func: m.func, mouth: share(this.mouthIdx), fill: share(this.lesionIdx) };
    this.samples.push(s);
    const f = this.forecast;
    if (s.mouth >= 0.85) this.done[0] = true;
    if (s.connectivity >= Math.min(0.9, f.vessels - 0.02)) this.done[1] = true;
    if (s.fill >= 0.9) this.done[2] = true;
    // converged: close to the forecast shape and no longer changing
    const prev = this.samples.filter((x) => x.step <= step - 12).pop();
    const still = prev ? Math.abs(s.iou - prev.iou) < 0.003 : false;
    if (this.done[0] && this.done[2] && still && s.iou >= f.shape - 0.01) this.done[3] = true;
    if (this.done[3] || step >= REPAIR_MAX_STEPS) this.finished = true;
    const active = this.done.findIndex((d) => !d);
    return { step, done: [...this.done], active, sample: s, finished: this.finished };
  }

  outcome(): RepairOutcome {
    const last = this.samples[this.samples.length - 1];
    const healed = this.samples.find((x) => x.iou >= last.iou * 0.98) ?? last;
    return { shape: last.iou, vessels: last.connectivity, healedAt: healed.step, steps: last.step };
  }
}
