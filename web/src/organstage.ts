// The isolated organ as a living voxel volume on the GPU (WebGPU), drawn over the
// Three.js view with the same camera. Handles the Fruit Ninja-style swipe: a glowing
// trail follows the gesture and, where it crossed the organ, a wedge incision is cut.
import type * as THREE from 'three';
import { initGPU } from './gpu';
import { invert, mat4, v3, type M4, type V3 } from './math';
import { Renderer, type CamLike, type FrameOpts } from './renderer';
import { Scene } from './scene';
import { TOOL_WEDGE } from './sim';
import { raycastTissue } from './tissue';

interface TrailPt { x: number; y: number; t: number }
export interface CutInfo { depthMm: number; lengthMm: number }

export class OrganStage {
  scene!: Scene;
  private renderer!: Renderer;
  private device!: GPUDevice;
  readonly toWorld: M4 = mat4();
  private cam: CamLike = { viewProj: new Float32Array(16), invViewProj: new Float32Array(16), eye: [0, 0, 0], dist: 1 };
  private trail: TrailPt[] = [];
  private stroke: { pts: TrailPt[]; hits: { x: number; y: number; t: number; world: V3; vox: V3 }[] } | null = null;
  private cutAge = -1;
  private flash = 0;
  private last = performance.now();
  private time = 0;
  ready = false;
  private needPack = true;
  private needRead = true;
  private trailDrawn = false;
  private lastVP = new Float32Array(16);
  onCut?: (info: CutInfo) => void;
  tissue: [number, number, number, number] = [0.36, 0.085, 0.065, 1];

  private constructor(readonly canvas: HTMLCanvasElement, readonly overlay: CanvasRenderingContext2D) {}

  static async create(canvas: HTMLCanvasElement, overlay: HTMLCanvasElement, organ: string) {
    const st = new OrganStage(canvas, overlay.getContext('2d')!);
    const gpu = await initGPU(canvas);
    if (!gpu) return null;
    st.device = gpu.device;
    st.renderer = new Renderer(gpu.device, gpu.ctx, gpu.format);
    st.renderer.particlesOn = false;
    st.renderer.aperture = 0;
    st.renderer.bloomStrength = 0.35;
    st.renderer.renderScale = 0.65;
    st.scene = await Scene.load(gpu.device, organ);
    const w = st.scene.organ.meta.world;
    if (w) {
      st.toWorld.set([w.half, 0, 0, 0, 0, w.half, 0, 0, 0, 0, w.half, 0, w.center[0], w.center[1], w.center[2], 1]);
    }
    st.reset();
    st.resize();
    addEventListener('resize', () => st.resize());
    return st;
  }

  resize() {
    const dpr = Math.min(devicePixelRatio || 1, 1.25);
    this.canvas.width = Math.round(innerWidth * dpr);
    this.canvas.height = Math.round(innerHeight * dpr);
    const o = this.overlay.canvas;
    o.width = Math.round(innerWidth * (devicePixelRatio || 1));
    o.height = Math.round(innerHeight * (devicePixelRatio || 1));
    this.overlay.setTransform(devicePixelRatio || 1, 0, 0, devicePixelRatio || 1, 0, 0);
    this.lastVP?.fill(NaN); // force a redraw at the new size
  }

  /** Healthy organ (the target tissue) with no lesion. */
  reset() {
    this.scene.sim.setFromLabels(this.scene.organ.labels);
    this.scene.latestSummary = null;
    this.cutAge = -1;
    this.needPack = true;
  }

  syncCamera(c: THREE.PerspectiveCamera, target: THREE.Vector3) {
    c.updateMatrixWorld();
    const vp = new Float32Array(16);
    const m = c.projectionMatrix.clone().multiply(c.matrixWorldInverse);
    vp.set(m.elements);
    this.cam.viewProj = vp;
    this.cam.invViewProj = invert(vp);
    this.cam.eye = [c.position.x, c.position.y, c.position.z];
    this.cam.dist = c.position.distanceTo(target);
  }

  // ------------------------------------------------------------------ swipe
  private ray(x: number, y: number) {
    const ndc = [(x / innerWidth) * 2 - 1, 1 - (y / innerHeight) * 2];
    const m = this.cam.invViewProj;
    const un = (z: number): V3 => {
      const w = m[3] * ndc[0] + m[7] * ndc[1] + m[11] * z + m[15];
      return [(m[0] * ndc[0] + m[4] * ndc[1] + m[8] * z + m[12]) / w, (m[1] * ndc[0] + m[5] * ndc[1] + m[9] * z + m[13]) / w, (m[2] * ndc[0] + m[6] * ndc[1] + m[10] * z + m[14]) / w];
    };
    const a = un(0), b = un(1);
    return { o: this.cam.eye, d: v3.norm(v3.sub(b, a)) };
  }

  private hit(x: number, y: number) {
    const r = this.ray(x, y);
    return raycastTissue(this.scene.latestSummary, this.scene.n, this.toWorld, r.o, r.d);
  }

  beginStroke(x: number, y: number) {
    const p = { x, y, t: performance.now() };
    this.stroke = { pts: [p], hits: [] };
    this.trail.push(p);
    this.addHit(p);
  }

  moveStroke(x: number, y: number) {
    const s = this.stroke;
    if (!s) return;
    const p = { x, y, t: performance.now() };
    const prev = s.pts[s.pts.length - 1];
    // sample the segment so fast swipes don't skip over thin parts of the organ
    const steps = Math.max(1, Math.ceil(Math.hypot(x - prev.x, y - prev.y) / 6));
    for (let i = 1; i <= steps; i++) {
      const q = { x: prev.x + ((x - prev.x) * i) / steps, y: prev.y + ((y - prev.y) * i) / steps, t: prev.t + ((p.t - prev.t) * i) / steps };
      const had = s.hits.length > 0;
      const h = this.addHit(q);
      // Fruit Ninja feel: each pass through the organ is cut as soon as the blade leaves it
      if (had && !h) { this.cut(); s.hits = []; }
    }
    s.pts.push(p);
    this.trail.push(p);
  }

  endStroke() {
    const s = this.stroke;
    if (s && s.hits.length > 1) this.cut();
    this.stroke = null;
  }

  private addHit(p: TrailPt) {
    const s = this.stroke!;
    const h = this.hit(p.x, p.y);
    if (h) s.hits.push({ ...p, world: h.world, vox: h.vox });
    return h;
  }

  /** Turn the stroke into a wedge incision in voxel space. */
  private cut() {
    const s = this.stroke!;
    const hits = s.hits;
    if (hits.length < 2) return;
    const n = this.scene.n;
    const first = hits[0], lastH = hits[hits.length - 1];
    const eye = this.cam.eye;
    // blade plane contains the two camera rays at the stroke's entry and exit
    const da = v3.norm(v3.sub(first.world, eye)), db = v3.norm(v3.sub(lastH.world, eye));
    let nrm = v3.cross(da, db);
    if (v3.len(nrm) < 1e-6) return;
    nrm = v3.norm(nrm);
    const ev = hits.map((h) => h.vox);
    const e: V3 = v3.scale(ev.reduce((a, b) => v3.add(a, b), [0, 0, 0] as V3), 1 / ev.length);
    const view = v3.norm(v3.add(da, db));                          // into the tissue
    const vDir = v3.norm(v3.sub(view, v3.scale(nrm, v3.dot(view, nrm))));
    const uDir = v3.norm(v3.cross(nrm, vDir));                    // along the stroke
    const sv = ev.map((p) => v3.dot(v3.sub(p, e), uDir));
    const s0 = Math.min(...sv) - 1.5, s1 = Math.max(...sv) + 1.5;
    // depth from swipe speed: a quick flick cuts deep, a slow drag stays shallow
    const dur = Math.max(30, lastH.t - first.t);
    const px = hits.reduce((a, h, i) => (i ? a + Math.hypot(h.x - hits[i - 1].x, h.y - hits[i - 1].y) : 0), 0);
    const speed = px / dur; // px per ms
    const thick = this.thickness(e, vDir);
    const frac = Math.min(0.85, 0.22 + speed * 0.35);
    const depth = Math.max(2.5, thick * frac);
    const w0 = 2.2;
    this.scene.sim.queueTool({
      kind: TOOL_WEDGE,
      p0: [e[0], e[1], e[2], depth],
      p1: [nrm[0], nrm[1], nrm[2], w0],
      p3: [vDir[0], vDir[1], vDir[2], s0],
      p4: [uDir[0], uDir[1], uDir[2], s1],
    });
    this.cutAge = 0;
    this.flash = 0.35;
    const mm = this.scene.organ.meta.world ? (this.scene.organ.meta.world.half * 2 * 1000) / n : 1;
    this.onCut?.({ depthMm: depth * mm, lengthMm: (s1 - s0) * mm });
  }

  /** Tissue thickness (voxels) along `dir` starting at voxel point `e`. */
  private thickness(e: V3, dir: V3) {
    const sum = this.scene.latestSummary;
    const n = this.scene.n;
    if (!sum) return n / 2;
    let last = 0;
    for (let t = 0; t < n * 1.8; t += 0.5) {
      const g = v3.add(e, v3.scale(dir, t)).map(Math.round);
      if (g.some((c) => c < 0 || c >= n)) break;
      if (sum[g[0] + g[1] * n + g[2] * n * n] & 0xff) last = t;
    }
    return Math.max(4, last);
  }

  // ------------------------------------------------------------------ frame
  /**
   * Work only happens when needed: simulation passes when the tissue changes (a cut,
   * or AI steps once trained), and a redraw when the camera moves or an effect fades.
   * A still organ costs nothing.
   */
  frame() {
    const now = performance.now();
    const dt = Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    this.time += dt;
    if (this.cutAge >= 0) this.cutAge += dt;
    this.flash = Math.max(0, this.flash - dt * 2.5);

    const sc = this.scene;
    const trained = sc.organ.meta.hidden > 0;
    const steps = trained ? sc.stepsThisFrame(dt) : 0;
    const simChanged = this.needPack || sc.sim.pendingTools > 0 || steps > 0;
    const vp = this.cam.viewProj;
    const camChanged = vp.some((v, i) => v !== this.lastVP[i]);
    const animating = this.flash > 0 || (this.cutAge >= 0 && this.cutAge < 1.5);
    if (simChanged || camChanged || animating || this.needRead) {
      const enc = this.device.createCommandEncoder();
      let read = null;
      if (simChanged) {
        sc.nutrientIters = trained ? 6 : 0;
        sc.encode(enc, dt, steps);
        this.needPack = false;
        this.needRead = true;
      }
      if (this.needRead) {
        read = sc.maybeRead(enc, this.time, true);
        if (read) this.needRead = false;
      }
      const opts: FrameOpts = {
        mode: 0, time: this.time, hiddenGain: 1, signalGain: 0, pulseFront: 2, pulseAmt: 0,
        clip: { on: false, n: [0, 0, 1], d: 0 }, flash: this.flash * 0.25, exposure: 1.0,
        growthRate: 0, debrisRate: 0, sparks: { a: [0, 0, 0], b: [0, 0, 0], on: false }, pieceVel: [0, 0, 0],
      };
      this.renderer.render(enc, this.cam, {
        sim: sc.sim, toWorld: this.toWorld, tissue: this.tissue,
        cuts: this.cutAge >= 0 ? [{ n: [0, 0, 1], d: 0, age: this.cutAge }] : [],
      }, null, opts, dt);
      this.device.queue.submit([enc.finish()]);
      read?.();
      this.lastVP.set(vp);
    }
    this.ready = !!sc.latestSummary;
    if (this.trail.length || this.trailDrawn) this.drawTrail(now);
  }

  private drawTrail(now: number) {
    const g = this.overlay;
    g.clearRect(0, 0, innerWidth, innerHeight);
    const life = 220;
    this.trail = this.trail.filter((p) => now - p.t < life);
    const pts = this.trail;
    this.trailDrawn = pts.length >= 2;
    if (pts.length < 2) return;
    g.save();
    g.globalCompositeOperation = 'lighter';
    g.lineCap = 'round';
    g.lineJoin = 'round';
    for (const pass of [{ w: 16, c: 'rgba(80,200,255,0.10)', blur: 22 }, { w: 7, c: 'rgba(120,225,255,0.45)', blur: 12 }, { w: 2.4, c: 'rgba(240,252,255,0.95)', blur: 4 }]) {
      g.shadowColor = 'rgba(95,225,255,0.9)';
      g.shadowBlur = pass.blur;
      g.strokeStyle = pass.c;
      for (let i = 1; i < pts.length; i++) {
        const a = pts[i - 1], b = pts[i];
        const k = 1 - (now - b.t) / life;
        g.lineWidth = pass.w * Math.max(0.05, k);
        g.globalAlpha = Math.max(0, k);
        g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
      }
    }
    g.restore();
  }
}
