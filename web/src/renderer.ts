// Frame renderer: ray-marched volumes -> HDR target, GPU particles, bloom chain,
// composite (DoF, ACES, vignette, grain) to the canvas.
import rayCode from './shaders/raymarch.wgsl?raw';
import postCode from './shaders/post.wgsl?raw';
import partCode from './shaders/particles.wgsl?raw';
import { invert, mat4, type M4, type V3 } from './math';
import type { Simulation } from './sim';

/** What the renderer needs from a camera (filled from the Three.js camera). */
export interface CamLike { viewProj: Float32Array; invViewProj: Float32Array; eye: V3; dist: number }

export const MODES = ['Realistic', 'Cell types', 'X-ray', 'Hidden signals', 'Nutrient', 'Histology'] as const;

export interface Cut { n: V3; d: number; age: number }
export interface Injury { p: V3; age: number }

export interface VolumeView {
  sim: Simulation;
  toWorld: M4;
  ghost?: Simulation | null;
  ghostAmt?: number;
  ghostTint?: [number, number, number, number];
  dissolve?: number;
  cuts?: Cut[];
  injuries?: Injury[];
  compare?: boolean;
  tissue?: [number, number, number, number];
  /** diagnosis overlay strengths: uncertainty (orange), critical area (red) */
  diag?: [number, number, number?];
}

export interface FrameOpts {
  mode: number;
  time: number;
  hiddenGain: number;
  signalGain: number;
  pulseFront: number;
  pulseAmt: number;
  clip: { on: boolean; n: V3; d: number };
  flash: number;
  exposure: number;
  growthRate: number;
  debrisRate: number;
  sparks: { a: V3; b: V3; on: boolean };
  pieceVel: V3;
  viewport?: [number, number, number, number];
}

const PARTICLES = { growth: 6000, debris: 9000, sparks: 1500 };
const HDR: GPUTextureFormat = 'rgba16float';

export class Renderer {
  private volLayout: GPUBindGroupLayout;
  private volMain: GPURenderPipeline;
  private volPiece: GPURenderPipeline;
  private postLayout: GPUBindGroupLayout;
  private prefilter: GPURenderPipeline;
  private down: GPURenderPipeline;
  private up: GPURenderPipeline;
  private composite: GPURenderPipeline;
  private partUpdate: GPUComputePipeline;
  private partRender: GPURenderPipeline;
  private sampler: GPUSampler;
  private uMain: GPUBuffer;
  private uPiece: GPUBuffer;
  private uPost: GPUBuffer;
  private uPart: GPUBuffer;
  private parts: GPUBuffer;
  private dummy2d: GPUTexture;
  private hdr!: GPUTexture;
  private depth!: GPUTexture;
  private bloom: GPUTexture[] = [];
  private size = [0, 0];
  renderScale = 0.8;
  frame = 0;
  bloomStrength = 0.55;
  aperture = 0.18;
  particlesOn = true;

  constructor(private device: GPUDevice, private ctx: GPUCanvasContext, format: GPUTextureFormat) {
    const d = device;
    const tex3 = (binding: number): GPUBindGroupLayoutEntry => ({
      binding, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' },
    });
    this.volLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        tex3(2), tex3(3), tex3(4), tex3(5), tex3(6), tex3(7),
        { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        tex3(9),
      ],
    });
    const rmod = d.createShaderModule({ code: rayCode, label: 'raymarch' });
    const vl = d.createPipelineLayout({ bindGroupLayouts: [this.volLayout] });
    this.volMain = d.createRenderPipeline({
      layout: vl, vertex: { module: rmod, entryPoint: 'vs' },
      fragment: { module: rmod, entryPoint: 'fsMain', targets: [{ format: HDR }, { format: HDR }] },
    });
    this.volPiece = d.createRenderPipeline({
      layout: vl, vertex: { module: rmod, entryPoint: 'vs' },
      fragment: {
        module: rmod, entryPoint: 'fsPiece',
        targets: [{ format: HDR, blend: { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } }],
      },
    });

    this.postLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      ],
    });
    const pmod = d.createShaderModule({ code: postCode, label: 'post' });
    const pl = d.createPipelineLayout({ bindGroupLayouts: [this.postLayout] });
    const post = (entry: string, format: GPUTextureFormat, blend?: GPUBlendState) => d.createRenderPipeline({
      layout: pl, vertex: { module: pmod, entryPoint: 'vs' },
      fragment: { module: pmod, entryPoint: entry, targets: [{ format, blend }] },
    });
    const add: GPUBlendState = { color: { srcFactor: 'one', dstFactor: 'one' }, alpha: { srcFactor: 'one', dstFactor: 'one' } };
    this.prefilter = post('fsPrefilter', HDR);
    this.down = post('fsDown', HDR);
    this.up = post('fsUp', HDR, add);
    this.composite = post('fsComposite', format);

    const [updSrc, renderPart] = partCode.split('// ---- render');
    const header = updSrc.slice(0, updSrc.indexOf('@group'));
    this.partUpdate = d.createComputePipeline({
      layout: 'auto', compute: { module: d.createShaderModule({ code: updSrc, label: 'particles' }), entryPoint: 'update' },
    });
    const prmod = d.createShaderModule({ code: header + renderPart, label: 'particles-render' });
    this.partRender = d.createRenderPipeline({
      layout: 'auto', vertex: { module: prmod, entryPoint: 'vsP' },
      fragment: { module: prmod, entryPoint: 'fsP', targets: [{ format: HDR, blend: add }] },
    });

    this.sampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
    const ub = (size: number) => d.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.uMain = ub(576);
    this.uPiece = ub(576);
    this.uPost = ub(32);
    this.uPart = ub(288);
    const np = PARTICLES.growth + PARTICLES.debris + PARTICLES.sparks;
    this.parts = d.createBuffer({ size: np * 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.dummy2d = d.createTexture({ size: [1, 1], format: HDR, usage: GPUTextureUsage.TEXTURE_BINDING });
  }

  private ensureTargets(w: number, h: number) {
    const sw = Math.max(1, Math.round(w * this.renderScale));
    const sh = Math.max(1, Math.round(h * this.renderScale));
    if (this.size[0] === sw && this.size[1] === sh) return;
    this.size = [sw, sh];
    this.hdr?.destroy(); this.depth?.destroy();
    for (const b of this.bloom) b.destroy();
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    this.hdr = this.device.createTexture({ size: [sw, sh], format: HDR, usage });
    this.depth = this.device.createTexture({ size: [sw, sh], format: HDR, usage });
    this.bloom = [];
    let bw = sw, bh = sh;
    for (let i = 0; i < 5; i++) {
      bw = Math.max(1, bw >> 1); bh = Math.max(1, bh >> 1);
      this.bloom.push(this.device.createTexture({ size: [bw, bh], format: HDR, usage }));
    }
  }

  private writeVolumeUniforms(buf: GPUBuffer, cam: CamLike, v: VolumeView, o: FrameOpts, w: number, h: number) {
    const f = new Float32Array(140);
    f.set(cam.invViewProj, 0);
    f.set(cam.viewProj, 16);
    f.set(invert(v.toWorld), 32);
    f.set(v.toWorld, 48);
    f.set([...cam.eye, o.time], 64);
    f.set([w, h, v.sim.n, o.mode], 68);
    f.set([-0.45, 0.75, 0.5, o.exposure], 72);
    f.set([...o.clip.n, o.clip.d], 76);
    f.set([o.hiddenGain, v.ghostAmt ?? 0, v.dissolve ?? 0, 0.12], 80);
    f.set([o.signalGain, o.pulseFront, o.pulseAmt, this.frame % 64], 84);
    f.set([o.clip.on && o.mode === 5 ? 1 : 0, v.compare ? 1 : 0, (v.cuts?.length ?? 0) > 0 ? 1 : 0, 0], 88);
    const cuts = v.cuts ?? [];
    for (let i = 0; i < 4; i++) {
      const c = cuts[i];
      if (c) { f.set([...c.n, c.d], 92 + i * 4); f[108 + i] = c.age; } else f[108 + i] = -1;
    }
    const inj = v.injuries ?? [];
    for (let i = 0; i < 4; i++) {
      const j = inj[i];
      if (j) f.set([...j.p, j.age], 112 + i * 4); else f.set([0, 0, 0, -1], 112 + i * 4);
    }
    f.set(v.ghostTint ?? [0, 0, 0, 0], 128);
    f.set(v.tissue ?? [0.86, 0.40, 0.37, 1], 132);
    f.set([v.diag?.[0] ?? 0, v.diag?.[1] ?? 0, o.time, v.diag?.[2] ?? 0], 136);
    this.device.queue.writeBuffer(buf, 0, f);
  }

  private volumeBG(v: VolumeView, buf: GPUBuffer, depthIn: GPUTexture) {
    const s = v.sim;
    const g = v.ghost ?? s;
    return this.device.createBindGroup({
      layout: this.volLayout,
      entries: [
        { binding: 0, resource: { buffer: buf } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: s.tex[0].createView() },
        { binding: 3, resource: s.tex[1].createView() },
        { binding: 4, resource: s.tex[2].createView() },
        { binding: 5, resource: s.perfTex.createView() },
        { binding: 6, resource: g.tex[0].createView() },
        { binding: 7, resource: g.tex[1].createView() },
        { binding: 8, resource: depthIn.createView() },
        { binding: 9, resource: s.auxTex.createView() },
      ],
    });
  }

  private postBG(src: GPUTexture, bloom?: GPUTexture, depth?: GPUTexture) {
    return this.device.createBindGroup({
      layout: this.postLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uPost } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: src.createView() },
        { binding: 3, resource: (bloom ?? this.dummy2d).createView() },
        { binding: 4, resource: (depth ?? this.dummy2d).createView() },
      ],
    });
  }

  private fullscreen(enc: GPUCommandEncoder, pipe: GPURenderPipeline, bg: GPUBindGroup, target: GPUTextureView, load: GPULoadOp) {
    const p = enc.beginRenderPass({ colorAttachments: [{ view: target, loadOp: load, storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    p.setPipeline(pipe); p.setBindGroup(0, bg); p.draw(3); p.end();
  }

  /** Render one frame. `main` is the healing organ; `piece` an optional second volume. */
  render(enc: GPUCommandEncoder, cam: CamLike, main: VolumeView, piece: VolumeView | null, o: FrameOpts,
    dt: number, out?: GPUTextureView) {
    const canvas = this.ctx.canvas as HTMLCanvasElement;
    const w = canvas.width, h = canvas.height;
    this.ensureTargets(w, h);
    this.frame++;
    const [sw, sh] = this.size;
    this.writeVolumeUniforms(this.uMain, cam, main, o, sw, sh);

    const mp = enc.beginRenderPass({
      colorAttachments: [
        { view: this.hdr.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
        { view: this.depth.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [1e4, 0, 0, 0] },
      ],
    });
    mp.setPipeline(this.volMain);
    mp.setBindGroup(0, this.volumeBG(main, this.uMain, this.dummy2d));
    mp.draw(3);
    mp.end();

    if (piece) {
      this.writeVolumeUniforms(this.uPiece, cam, piece, o, sw, sh);
      const pp = enc.beginRenderPass({ colorAttachments: [{ view: this.hdr.createView(), loadOp: 'load', storeOp: 'store' }] });
      pp.setPipeline(this.volPiece);
      pp.setBindGroup(0, this.volumeBG(piece, this.uPiece, this.depth));
      pp.draw(3);
      pp.end();
    }

    if (this.particlesOn) this.particles(enc, cam, main, piece, o, dt, sw / sh);

    // bloom
    this.device.queue.writeBuffer(this.uPost, 0, new Float32Array([
      o.exposure, this.bloomStrength, cam.dist - 0.2, this.aperture, o.time, o.flash, 0.9, 0.025,
    ]));
    this.fullscreen(enc, this.prefilter, this.postBG(this.hdr), this.bloom[0].createView(), 'clear');
    for (let i = 1; i < this.bloom.length; i++) {
      this.fullscreen(enc, this.down, this.postBG(this.bloom[i - 1]), this.bloom[i].createView(), 'clear');
    }
    for (let i = this.bloom.length - 1; i > 0; i--) {
      this.fullscreen(enc, this.up, this.postBG(this.bloom[i]), this.bloom[i - 1].createView(), 'load');
    }
    const target = out ?? this.ctx.getCurrentTexture().createView();
    const cp = enc.beginRenderPass({ colorAttachments: [{ view: target, loadOp: o.viewport ? 'load' : 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    cp.setPipeline(this.composite);
    cp.setBindGroup(0, this.postBG(this.hdr, this.bloom[0], this.depth));
    if (o.viewport) cp.setViewport(o.viewport[0], o.viewport[1], o.viewport[2], o.viewport[3], 0, 1);
    cp.draw(3);
    cp.end();
  }

  private particles(enc: GPUCommandEncoder, cam: CamLike, main: VolumeView, piece: VolumeView | null, o: FrameOpts, dt: number, aspect: number) {
    const f = new Float32Array(72);
    f.set(cam.viewProj, 0);
    f.set(main.toWorld, 16);
    f.set(piece?.toWorld ?? mat4(), 32);
    f.set([...cam.eye, o.time], 48);
    f.set([Math.min(dt, 0.05), main.sim.n, o.growthRate, piece ? o.debrisRate : 0], 52);
    f.set([...o.sparks.a, o.sparks.on ? 1 : 0], 56);
    f.set([...o.sparks.b, this.frame], 60);
    f.set([...o.pieceVel, 1], 64);
    f.set([PARTICLES.growth, PARTICLES.debris, PARTICLES.sparks, aspect], 68);
    this.device.queue.writeBuffer(this.uPart, 0, f);
    const ps = piece?.sim ?? main.sim;
    const cbg = this.device.createBindGroup({
      layout: this.partUpdate.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uPart } },
        { binding: 1, resource: { buffer: this.parts } },
        { binding: 2, resource: main.sim.tex[0].createView() },
        { binding: 3, resource: main.sim.tex[2].createView() },
        { binding: 4, resource: ps.tex[0].createView() },
      ],
    });
    const total = PARTICLES.growth + PARTICLES.debris + PARTICLES.sparks;
    const cp = enc.beginComputePass();
    cp.setPipeline(this.partUpdate); cp.setBindGroup(0, cbg); cp.dispatchWorkgroups(Math.ceil(total / 64)); cp.end();
    const rbg = this.device.createBindGroup({
      layout: this.partRender.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uPart } },
        { binding: 1, resource: { buffer: this.parts } },
        { binding: 2, resource: this.depth.createView() },
      ],
    });
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: this.hdr.createView(), loadOp: 'load', storeOp: 'store' }] });
    rp.setPipeline(this.partRender); rp.setBindGroup(0, rbg); rp.draw(6, total); rp.end();
  }
}
