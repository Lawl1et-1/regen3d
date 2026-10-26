// WebGPU 3D NCA simulation: state buffers, NCA step + alive mask, tools,
// nutrient/toxin fields, packing to textures and async summary readback.
import ncaSrc from './shaders/nca.wgsl?raw';
import maskSrc from './shaders/mask.wgsl?raw';
import fieldsSrc from './shaders/fields.wgsl?raw';
import packSrc from './shaders/pack.wgsl?raw';

export const CH = 16;
export const TOOL_SPHERE = 1;
export const TOOL_STROKE = 2;
export const TOOL_PLANE = 3;
export const TOOL_TOXIN = 4;
export const TOOL_WEDGE = 5;

export interface OrganMeta {
  name: string;
  n: number;
  seed: [number, number, number];
  hilum: [number, number, number] | null;
  counts: Record<string, number>;
  hidden: number;
  fire_rate: number;
  trained_iters: number;
  w1: number[];
  b1: number[];
  w2: number[];
  world?: { center: [number, number, number]; half: number };
  /** hidden channels that carry the wound signal (shown by the AI Brain view) */
  brain_channels?: [number, number, number];
}

export interface Organ {
  meta: OrganMeta;
  labels: Uint8Array;
}

export async function loadOrgan(name: string): Promise<Organ> {
  const base = `${import.meta.env.BASE_URL}organs/${name}`;
  const [meta, labels] = await Promise.all([
    fetch(`${base}.json`).then((r) => {
      if (!r.ok) throw new Error(`missing organ ${name}`);
      return r.json() as Promise<OrganMeta>;
    }),
    fetch(`${base}.labels.bin`).then((r) => r.arrayBuffer()),
  ]);
  return { meta, labels: new Uint8Array(labels) };
}

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

/** Pipelines and weights shared by every simulation instance of one organ. */
export class SimKernels {
  readonly n: number;
  readonly hidden: number;
  readonly w1: GPUBuffer;
  readonly b1: GPUBuffer;
  readonly w2t: GPUBuffer;
  readonly stepPipe: GPUComputePipeline;
  readonly maskPipe: GPUComputePipeline;
  readonly fieldLayout: GPUBindGroupLayout;
  readonly toolPipe: GPUComputePipeline;
  readonly nutPipe: GPUComputePipeline;
  readonly toxPipe: GPUComputePipeline;
  readonly packPipe: GPUComputePipeline;

  constructor(readonly device: GPUDevice, readonly organ: Organ) {
    const m = organ.meta;
    this.n = m.n;
    this.hidden = m.hidden;
    const mk = (data: Float32Array<ArrayBuffer>) => {
      const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: STORAGE });
      device.queue.writeBuffer(b, 0, data);
      return b;
    };
    this.w1 = mk(new Float32Array(m.w1));
    this.b1 = mk(new Float32Array(m.b1));
    // w2 is (C, hidden) row-major; the shader wants (hidden, C) so each hidden unit adds 4 vec4
    const w2t = new Float32Array(m.hidden * CH);
    for (let c = 0; c < CH; c++) for (let h = 0; h < m.hidden; h++) w2t[h * CH + c] = m.w2[c * m.hidden + h];
    this.w2t = mk(w2t);

    const mod = (code: string, label: string) => device.createShaderModule({ code, label });
    this.stepPipe = device.createComputePipeline({
      layout: 'auto', compute: { module: mod(ncaSrc, 'nca'), entryPoint: 'step' },
    });
    this.maskPipe = device.createComputePipeline({
      layout: 'auto', compute: { module: mod(maskSrc, 'mask'), entryPoint: 'mask' },
    });
    const st = (binding: number): GPUBindGroupLayoutEntry => ({
      binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' },
    });
    this.fieldLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        st(1), st(2), st(3), st(4), st(5), st(6), st(7),
        { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
      ],
    });
    const fmod = mod(fieldsSrc, 'fields');
    const fl = device.createPipelineLayout({ bindGroupLayouts: [this.fieldLayout] });
    this.toolPipe = device.createComputePipeline({ layout: fl, compute: { module: fmod, entryPoint: 'tool' } });
    this.nutPipe = device.createComputePipeline({ layout: fl, compute: { module: fmod, entryPoint: 'nutrient' } });
    this.toxPipe = device.createComputePipeline({ layout: fl, compute: { module: fmod, entryPoint: 'toxin' } });
    this.packPipe = device.createComputePipeline({
      layout: 'auto', compute: { module: mod(packSrc, 'pack'), entryPoint: 'pack' },
    });
  }
}

export interface ToolOp {
  kind: number;
  p0: [number, number, number, number];
  p1?: [number, number, number, number];
  p3?: [number, number, number, number];
  p4?: [number, number, number, number];
  amount?: number;
}

/** One simulated tissue volume (the organ, a detached piece, or a forecast copy). */
export class Simulation {
  readonly n: number;
  readonly voxels: number;
  readonly state: GPUBuffer;
  readonly next: GPUBuffer;
  readonly pre: GPUBuffer;
  readonly act: GPUBuffer;
  readonly wound: GPUBuffer;
  readonly growth: GPUBuffer;
  readonly toxA: GPUBuffer;
  readonly toxB: GPUBuffer;
  readonly nutA: GPUBuffer;
  readonly nutB: GPUBuffer;
  readonly ctrl: GPUBuffer;
  readonly summary: GPUBuffer;
  readonly tex: GPUTexture[];
  readonly perfTex: GPUTexture;
  /** Diagnosis overlay (rgba8): uncertainty, critical area, predicted tissue, predicted vessels. */
  readonly auxTex: GPUTexture;
  private readonly params: GPUBuffer;
  private readonly fieldU: GPUBuffer;
  private readonly packU: GPUBuffer;
  private readonly stepBG: GPUBindGroup;
  private readonly maskBG: GPUBindGroup;
  private readonly fieldBG: GPUBindGroup;
  private readonly packBG: GPUBindGroup;
  private readonly staging: GPUBuffer[];
  private stagingBusy = [false, false];
  private toolQueue: ToolOp[] = [];
  steps = 0;
  hiddenSel = 0;
  frozen = false;
  toxinActive = 0;

  constructor(readonly k: SimKernels, readonly label = 'sim') {
    const d = k.device;
    const n = (this.n = k.n);
    const v = (this.voxels = n * n * n);
    const buf = (bytes: number) => d.createBuffer({ size: bytes, usage: STORAGE, label });
    this.state = buf(v * CH * 4);
    this.next = buf(v * CH * 4);
    this.pre = buf(v * 4);
    this.act = buf(v * 4);
    this.wound = buf(v * 4);
    this.growth = buf(v * 4);
    this.toxA = buf(v * 4);
    this.toxB = buf(v * 4);
    this.nutA = buf(v * 4);
    this.nutB = buf(v * 4);
    this.summary = buf(v * 4);
    this.ctrl = buf(16);
    this.params = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.fieldU = d.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.packU = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.staging = [0, 1].map(() =>
      d.createBuffer({ size: v * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
    this.tex = [0, 1, 2].map((i) => d.createTexture({
      label: `${label}-tex${i}`, size: [n, n, n], dimension: '3d', format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    }));
    this.auxTex = d.createTexture({
      size: [n, n, n], dimension: '3d', format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.perfTex = d.createTexture({
      size: [n, n, n], dimension: '3d', format: 'r8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
    });

    const e = (binding: number, buffer: GPUBuffer): GPUBindGroupEntry => ({ binding, resource: { buffer } });
    this.stepBG = d.createBindGroup({
      layout: k.stepPipe.getBindGroupLayout(0),
      entries: [e(0, this.params), e(1, this.state), e(2, this.next), e(3, k.w1), e(4, k.b1), e(5, k.w2t),
        e(6, this.pre), e(7, this.act), e(8, this.ctrl)],
    });
    this.maskBG = d.createBindGroup({
      layout: k.maskPipe.getBindGroupLayout(0),
      entries: [e(0, this.params), e(1, this.next), e(2, this.state), e(3, this.pre), e(4, this.wound),
        e(5, this.toxA), e(6, this.ctrl), e(7, this.growth)],
    });
    this.fieldBG = d.createBindGroup({
      layout: k.fieldLayout,
      entries: [e(0, this.fieldU), e(1, this.state), e(2, this.wound), e(3, this.toxA), e(4, this.toxB),
        e(5, this.nutA), e(6, this.nutB), e(7, this.act), { binding: 8, resource: this.perfTex.createView() }],
    });
    this.packBG = d.createBindGroup({
      layout: k.packPipe.getBindGroupLayout(0),
      entries: [e(0, this.packU), e(1, this.state), e(2, this.wound), e(3, this.nutA), e(4, this.act),
        { binding: 7, resource: this.tex[0].createView() },
        { binding: 8, resource: this.tex[1].createView() },
        { binding: 9, resource: this.tex[2].createView() },
        e(10, this.summary)],
    });
    this.writeParams();
    this.reset();
  }

  private get wg() { return Math.ceil(this.n / 4); }

  writeParams(seed = 1234) {
    const b = new ArrayBuffer(32);
    const u = new Uint32Array(b);
    const f = new Float32Array(b);
    u[0] = this.n; u[1] = this.k.hidden; u[2] = seed; u[3] = this.hiddenSel;
    f[4] = this.k.organ.meta.fire_rate; f[5] = 0.985; f[6] = 0.35; u[7] = this.frozen ? 1 : 0;
    this.k.device.queue.writeBuffer(this.params, 0, b);
  }

  setHiddenSel(i: number) {
    this.hiddenSel = i;
    const b = this.k.organ.meta.brain_channels ?? [5 + i, 6, 7];
    this.k.device.queue.writeBuffer(this.packU, 0, new Uint32Array([this.n, b[0], b[1], b[2]]));
  }

  /** Clear everything and plant the single seed cell. */
  reset(stepIndex = 0) {
    const d = this.k.device;
    const s = new Float32Array(this.voxels * CH);
    const [sx, sy, sz] = this.k.organ.meta.seed;
    const i = (sx + sy * this.n + sz * this.n * this.n) * CH;
    s[i] = 1;
    for (let c = 5; c < CH; c++) s[i + c] = 1;
    d.queue.writeBuffer(this.state, 0, s);
    const z = new Float32Array(this.voxels);
    for (const b of [this.act, this.wound, this.growth, this.toxA, this.toxB, this.nutA, this.nutB]) d.queue.writeBuffer(b, 0, z);
    this.setStepIndex(stepIndex);
    this.setHiddenSel(this.hiddenSel);
    this.toxinActive = 0;
  }

  /** Fill the state with the target organ: alive, one-hot cell types, hidden channels at 0. */
  setFromLabels(labels: Uint8Array) {
    const s = new Float32Array(this.voxels * CH);
    for (let i = 0; i < this.voxels; i++) {
      const l = labels[i];
      if (!l) continue;
      s[i * CH] = 1;
      s[i * CH + l] = 1;
    }
    const d = this.k.device;
    d.queue.writeBuffer(this.state, 0, s);
    const z = new Float32Array(this.voxels);
    for (const b of [this.act, this.wound, this.growth, this.toxA, this.toxB, this.nutA, this.nutB]) d.queue.writeBuffer(b, 0, z);
  }

  /** The step counter seeds the stochastic update mask, so it makes runs reproducible. */
  setStepIndex(i: number) {
    this.steps = i;
    this.k.device.queue.writeBuffer(this.ctrl, 0, new Uint32Array([i, 0, 0, 0]));
  }

  copyFrom(enc: GPUCommandEncoder, o: Simulation) {
    const pairs: [GPUBuffer, GPUBuffer][] = [[o.state, this.state], [o.act, this.act], [o.wound, this.wound],
      [o.growth, this.growth], [o.toxA, this.toxA], [o.nutA, this.nutA], [o.ctrl, this.ctrl]];
    for (const [a, b] of pairs) enc.copyBufferToBuffer(a, 0, b, 0, a.size);
    enc.copyTextureToTexture({ texture: o.perfTex }, { texture: this.perfTex }, [this.n, this.n, this.n]);
    this.steps = o.steps;
    this.toxinActive = o.toxinActive;
  }

  get pendingTools() { return this.toolQueue.length; }

  queueTool(op: ToolOp) { this.toolQueue.push(op); if (op.kind === TOOL_TOXIN) this.toxinActive = 400; }

  private writeField(kind: number, p0: number[], p1: number[], p2: number[], p3 = [0, 0, 0, 0], p4 = [0, 0, 0, 0]) {
    const b = new ArrayBuffer(96);
    const u = new Uint32Array(b);
    const f = new Float32Array(b);
    u[0] = this.n; u[1] = kind; u[2] = this.hiddenSel; u[3] = 0;
    f.set(p0, 4); f.set(p1, 8); f.set(p2, 12); f.set(p3, 16); f.set(p4, 20);
    return b;
  }

  /**
   * Encode one frame of work: pending tools, `steps` NCA steps, field updates and packing.
   * Tools need their own uniform contents, so each one is submitted separately.
   */
  encodeTools() {
    const d = this.k.device;
    for (const op of this.toolQueue) {
      const enc = d.createCommandEncoder();
      d.queue.writeBuffer(this.fieldU, 0, this.writeField(op.kind, op.p0, op.p1 ?? [0, 0, 0, 0], [0.15, 0.12, op.amount ?? 3, 0.996], op.p3, op.p4));
      const p = enc.beginComputePass();
      p.setPipeline(this.k.toolPipe);
      p.setBindGroup(0, this.fieldBG);
      p.dispatchWorkgroups(this.wg, this.wg, this.wg);
      p.end();
      d.queue.submit([enc.finish()]);
    }
    this.toolQueue.length = 0;
  }

  encodeSteps(enc: GPUCommandEncoder, count: number) {
    if (count <= 0) return;
    const wg = this.wg;
    for (let s = 0; s < count; s++) {
      const p = enc.beginComputePass();
      p.setPipeline(this.k.stepPipe);
      p.setBindGroup(0, this.stepBG);
      p.dispatchWorkgroups(wg, wg, wg);
      p.setPipeline(this.k.maskPipe);
      p.setBindGroup(0, this.maskBG);
      p.dispatchWorkgroups(wg, wg, wg);
      p.end();
      if (this.toxinActive > 0) {
        this.encodeToxin(enc);
        this.toxinActive--;
      }
    }
    this.steps += count;
  }

  private encodeToxin(enc: GPUCommandEncoder) {
    const p = enc.beginComputePass();
    p.setPipeline(this.k.toxPipe);
    p.setBindGroup(0, this.fieldBG);
    p.dispatchWorkgroups(this.wg, this.wg, this.wg);
    p.end();
    enc.copyBufferToBuffer(this.toxB, 0, this.toxA, 0, this.toxA.size);
  }

  /** Field uniforms for nutrient/toxin (written once per frame before encodeFields). */
  prepareFields() {
    this.k.device.queue.writeBuffer(this.fieldU, 0, this.writeField(0, [0, 0, 0, 0], [0, 0, 0, 0], [0.15, 0.12, 0, 0.996]));
  }

  encodeFields(enc: GPUCommandEncoder, nutrientIters: number) {
    for (let i = 0; i < nutrientIters; i++) {
      const p = enc.beginComputePass();
      p.setPipeline(this.k.nutPipe);
      p.setBindGroup(0, this.fieldBG);
      p.dispatchWorkgroups(this.wg, this.wg, this.wg);
      p.end();
      enc.copyBufferToBuffer(this.nutB, 0, this.nutA, 0, this.nutA.size);
    }
  }

  encodePack(enc: GPUCommandEncoder) {
    const p = enc.beginComputePass();
    p.setPipeline(this.k.packPipe);
    p.setBindGroup(0, this.packBG);
    p.dispatchWorkgroups(this.wg, this.wg, this.wg);
    p.end();
  }

  /** Copy the summary to a free staging buffer; resolves with class|nutrient|activity per voxel. */
  readSummary(enc: GPUCommandEncoder): (() => Promise<Uint32Array>) | null {
    const slot = this.stagingBusy[0] ? (this.stagingBusy[1] ? -1 : 1) : 0;
    if (slot < 0) return null;
    this.stagingBusy[slot] = true;
    const sb = this.staging[slot];
    enc.copyBufferToBuffer(this.summary, 0, sb, 0, sb.size);
    return async () => {
      await sb.mapAsync(GPUMapMode.READ);
      const out = new Uint32Array(sb.getMappedRange().slice(0));
      sb.unmap();
      this.stagingBusy[slot] = false;
      return out;
    };
  }

  uploadAux(data: Uint8Array<ArrayBuffer> | null) {
    const n = this.n;
    this.k.device.queue.writeTexture({ texture: this.auxTex }, data ?? new Uint8Array(n * n * n * 4),
      { bytesPerRow: n * 4, rowsPerImage: n }, [n, n, n]);
  }

  uploadPerfusion(data: Uint8Array<ArrayBuffer>) {
    const n = this.n;
    this.k.device.queue.writeTexture({ texture: this.perfTex }, data, { bytesPerRow: n, rowsPerImage: n }, [n, n, n]);
  }

  /** Read the full state (for snapshots / replays). */
  async readState(): Promise<Float32Array> {
    const d = this.k.device;
    const sb = d.createBuffer({ size: this.state.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(this.state, 0, sb, 0, this.state.size);
    d.queue.submit([enc.finish()]);
    await sb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(sb.getMappedRange().slice(0));
    sb.destroy();
    return out;
  }

  writeState(s: Float32Array<ArrayBuffer>) {
    this.k.device.queue.writeBuffer(this.state, 0, s);
  }

  setFrozen(f: boolean) {
    this.frozen = f;
    this.writeParams();
  }

  destroy() {
    for (const b of [this.state, this.next, this.pre, this.act, this.wound, this.growth, this.toxA, this.toxB,
      this.nutA, this.nutB, this.summary, this.ctrl, this.params, this.fieldU, this.packU, ...this.staging]) b.destroy();
    for (const t of [...this.tex, this.perfTex, this.auxTex]) t.destroy();
  }
}
