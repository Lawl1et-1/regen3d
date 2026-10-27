// A loaded organ: kernels, the live simulation, metrics readback and history.
import { MetricsEngine, type Metrics } from './metrics';
import { ReconnectPulses } from './reconnect';
import { loadOrgan, SimKernels, Simulation, type Organ } from './sim';

export const STEPS_PER_SEC = 30; // 1x speed

export interface Sample { step: number; t: number; m: Metrics }

export class Scene {
  readonly sim: Simulation;
  readonly metrics: MetricsEngine;
  readonly pulses: ReconnectPulses;
  latest: Metrics | null = null;
  latestSummary: Uint32Array | null = null;
  history: Sample[] = [];
  paused = false;
  speed = 1;
  private acc = 0;
  private pendingStep = 0;
  private readFrame = 0;
  private reading = false;
  injuryStep = -1;
  injuryTime = -1;
  nutrientIters = 6;

  private constructor(readonly device: GPUDevice, readonly organ: Organ, readonly kernels: SimKernels) {
    this.sim = new Simulation(kernels, organ.meta.name);
    this.metrics = new MetricsEngine(organ);
    this.pulses = new ReconnectPulses(this.metrics);
  }

  static async load(device: GPUDevice, name: string) {
    const organ = await loadOrgan(name);
    return new Scene(device, organ, new SimKernels(device, organ));
  }

  get name() { return this.organ.meta.name; }
  get n() { return this.organ.meta.n; }

  /** Grow from the seed, or jump straight to a grown organ by pre-running steps. */
  reset(pregrow = 0) {
    this.sim.reset();
    this.history = [];
    this.injuryStep = -1;
    if (pregrow > 0) {
      for (let done = 0; done < pregrow; done += 25) {
        const enc = this.device.createCommandEncoder();
        this.sim.encodeSteps(enc, Math.min(25, pregrow - done));
        this.device.queue.submit([enc.finish()]);
      }
    }
  }

  stepOnce() { this.pendingStep++; }

  markInjury(time: number) {
    this.injuryStep = this.sim.steps;
    this.injuryTime = time;
  }

  /** Steps to run this frame given speed and dt (capped so a slow frame never snowballs). */
  stepsThisFrame(dt: number): number {
    if (this.paused) { const s = this.pendingStep; this.pendingStep = 0; return s; }
    this.acc += dt * STEPS_PER_SEC * this.speed;
    const k = Math.min(Math.floor(this.acc), Math.ceil(16 * Math.max(1, this.speed)));
    this.acc = Math.min(this.acc - k, 2);
    return k;
  }

  /** Encode simulation work for one frame. Returns the number of steps taken. */
  encode(enc: GPUCommandEncoder, dt: number, steps?: number): number {
    const k = steps ?? this.stepsThisFrame(dt);
    this.sim.encodeTools();
    this.sim.encodeSteps(enc, k);
    this.sim.prepareFields();
    this.sim.encodeFields(enc, this.nutrientIters);
    this.sim.encodePack(enc);
    return k;
  }

  /**
   * Every few frames, encode a copy of the summary; call the returned function after
   * submit to read it back and update metrics + the perfusion texture.
   */
  maybeRead(enc: GPUCommandEncoder, time: number, force = false) {
    this.readFrame++;
    if (this.reading || (!force && this.readFrame % 4 !== 0)) return null;
    const read = this.sim.readSummary(enc);
    if (!read) return null;
    this.reading = true;
    return async () => {
      const sum = await read();
      this.reading = false;
      const m = this.metrics.compute(sum);
      this.pulses.update(sum, time);
      this.sim.uploadPerfusion(this.metrics.perfusion);
      this.latest = m;
      this.latestSummary = sum;
      this.history.push({ step: this.sim.steps, t: time, m });
      if (this.history.length > 600) this.history.splice(0, this.history.length - 600);
    };
  }

  destroy() { this.sim.destroy(); }
}
