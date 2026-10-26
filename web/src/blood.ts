// Blood supply of the simulated world. Nutrient diffuses from vessels that are still
// connected to the hilum (found by a BFS over the read-back class grid); the NCA mask
// pass refuses new tissue where blood does not reach and lets starved tissue die.
// These helpers keep that field up to date while a simulation runs off-screen.
import { nextFrame } from './diagnosis';
import type { Metrics, MetricsEngine } from './metrics';
import type { Simulation } from './sim';

const CHUNK = 2;          // steps between two refreshes of the blood supply
const NUT_ITERS = 4;      // diffusion iterations per refresh

/** Measure the current state: class grid + metrics, and refresh perfusion from it. */
export async function measure(device: GPUDevice, sim: Simulation, metrics: MetricsEngine) {
  const enc = device.createCommandEncoder();
  sim.encodePack(enc);
  const read = sim.readSummary(enc)!;
  device.queue.submit([enc.finish()]);
  const sum = await read();
  const m = metrics.compute(sum);
  sim.uploadPerfusion(metrics.perfusion);
  return { sum, m };
}

/** Compute the healthy blood supply of the current (healthy) state and switch the constraint on. */
export async function establishBlood(device: GPUDevice, sim: Simulation, metrics: MetricsEngine) {
  sim.setBlood(false);
  await measure(device, sim, metrics);
  sim.prepareFields();
  const enc = device.createCommandEncoder();
  sim.encodeFields(enc, 160);
  sim.captureNutRef(enc);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  sim.setBlood(true);
}

/**
 * Run `steps` NCA steps with the blood supply refreshed every couple of steps.
 * `onMeasure` gets every measurement (step index, class grid, metrics); return true to stop.
 */
export async function runWithBlood(device: GPUDevice, sim: Simulation, metrics: MetricsEngine, steps: number,
  onMeasure: (step: number, sum: Uint32Array, m: Metrics) => boolean | void, alive: () => boolean = () => true) {
  sim.prepareFields();
  let r = await measure(device, sim, metrics);
  if (!alive() || onMeasure(0, r.sum, r.m)) return;
  for (let s = 0; s < steps;) {
    const k = Math.min(CHUNK, steps - s);
    const enc = device.createCommandEncoder();
    sim.encodeSteps(enc, k);
    sim.encodeFields(enc, NUT_ITERS);
    device.queue.submit([enc.finish()]);
    s += k;
    r = await measure(device, sim, metrics);
    if (!alive() || onMeasure(s, r.sum, r.m)) return;
    await nextFrame();
    if (!alive()) return;
  }
}
