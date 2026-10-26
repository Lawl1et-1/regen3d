// "Without AI": a simple, non-learned repair for comparison. Generic automatic hole
// filling, as found in mesh/volume tools: repeatedly, any empty voxel surrounded by a
// majority of tissue (>= 14 of its 26 neighbours) is filled with the most common cell
// type among those neighbours. Flat or convex surfaces never grow; gaps and holes fill.
// It knows nothing about the organ: no shape memory, no notion of vessels, so it closes
// gaps but cannot rebuild what was there.
const NEED = 14;
const MAX_ITERS = 14;

/** Returns the label grid after each iteration (frames of the fill), starting with the lesioned organ. */
export function holeFillFrames(classes: Uint32Array, n: number): Uint8Array[] {
  let cur = new Uint8Array(classes.length);
  for (let i = 0; i < classes.length; i++) cur[i] = classes[i] & 0xff;
  const frames = [cur];
  const nn = n * n;
  for (let it = 0; it < MAX_ITERS; it++) {
    const next = new Uint8Array(cur);
    let changed = 0;
    for (let z = 1; z < n - 1; z++) for (let y = 1; y < n - 1; y++) for (let x = 1; x < n - 1; x++) {
      const i = x + y * n + z * nn;
      if (cur[i]) continue;
      const votes = [0, 0, 0, 0, 0];
      let alive = 0;
      for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const c = cur[i + dx + dy * n + dz * nn];
        if (c) { alive++; votes[c]++; }
      }
      if (alive < NEED) continue;
      let best = 1;
      for (let c = 2; c <= 4; c++) if (votes[c] > votes[best]) best = c;
      next[i] = best;
      changed++;
    }
    cur = next;
    frames.push(cur);
    if (!changed) break;
  }
  return frames;
}

/** Pack a label grid like the GPU summary (class in the low byte) so MetricsEngine can score it. */
export function asSummary(labels: Uint8Array): Uint32Array {
  const s = new Uint32Array(labels.length);
  for (let i = 0; i < labels.length; i++) s[i] = labels[i];
  return s;
}
