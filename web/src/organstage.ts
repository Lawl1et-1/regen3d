// The isolated organ as a living voxel volume on the GPU (WebGPU), drawn over the
// Three.js view with the same camera. Handles the Fruit Ninja-style swipe: a glowing
// trail follows the gesture and, where it crossed the organ, a wedge incision is cut.
import type * as THREE from 'three';
import { initGPU } from './gpu';
import { invert, mat4, transformDir, v3, type M4, type V3 } from './math';
import { Renderer, type CamLike, type FrameOpts } from './renderer';
import { Scene } from './scene';
import { Diagnosis, liveStepsPerSec, type DiagnosisResult } from './diagnosis';
import { RepairTracker, type RepairOutcome, type RepairState } from './repair';
import { asSummary, holeFillFrames } from './baseline';
import type { Metrics } from './metrics';
import { loadCheckpoint, SimKernels, Simulation, TOOL_WEDGE, type ToolOp, type Weights } from './sim';
import { raycastTissue } from './tissue';

interface TrailPt { x: number; y: number; t: number }
export interface CutInfo { depthMm: number; lengthMm: number }
export interface LearnRow { iteration: number; metrics: Metrics | null; step: number; done: boolean }
export interface LearnProgress { rows: LearnRow[]; current: number; finished: boolean }
const LEARN_STEPS = 100;

/** Parenchyma colour of each organ in the realistic view. */
const TISSUE: Record<string, [number, number, number, number]> = {
  liver: [0.36, 0.085, 0.065, 1],
  heart: [0.44, 0.06, 0.07, 1],
  kidney: [0.40, 0.10, 0.085, 1],
  lung: [0.62, 0.34, 0.36, 1],
};

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
  diagnosis: Diagnosis | null = null;
  /** class grid of the healthy organ (before any lesion), the reference for the diagnosis */
  private healthy: Uint32Array | null = null;
  private lesioned = false;
  private diagPending = false;
  diag: [number, number] = [0, 0];
  /** idle -> (diagnosis) -> plan (hologram preview) -> repair (live) -> done */
  mode: 'idle' | 'plan' | 'repair' | 'done' = 'idle';
  private modeT = 0;
  private ghost = 0;
  private tracker: RepairTracker | null = null;
  private repairStart = 0;
  brain = false;
  onRepair?: (s: RepairState) => void;
  onRepairDone?: (o: RepairOutcome, forecast: DiagnosisResult) => void;
  onPlan?: () => void;
  /** which volume(s) are shown: the live organ, AI vs hole filling side by side, or a training checkpoint */
  view: 'live' | 'compare' | 'learn' = 'live';
  private cutOps: ToolOp[] = [];
  private lastResult: DiagnosisResult | null = null;
  private baseSim: Simulation | null = null;
  private learnKernels: SimKernels | null = null;
  private learnSim: Simulation | null = null;
  private ckptCache = new Map<number, Weights>();
  private viewToken = 0;
  private target: V3 = [0, 0, 0];
  private pieceWorld: M4 = mat4();
  private mainWorld: M4 = mat4();
  onCompare?: (c: { ai: Metrics; base: Metrics; frame: number; frames: number }) => void;
  onLearn?: (l: LearnProgress) => void;
  tissue: [number, number, number, number] = [0.36, 0.085, 0.065, 1];
  /** one simulation + diagnosis per organ, created on first use */
  private organs = new Map<string, { scene: Scene; diagnosis: Diagnosis | null }>();
  onDiagProgress?: (p: Parameters<NonNullable<Diagnosis['onProgress']>>[0]) => void;
  onDiagResult?: (r: DiagnosisResult) => void;

  private constructor(readonly canvas: HTMLCanvasElement, readonly overlay: CanvasRenderingContext2D) {}

  static async create(canvas: HTMLCanvasElement, overlay: HTMLCanvasElement) {
    const st = new OrganStage(canvas, overlay.getContext('2d')!);
    const gpu = await initGPU(canvas);
    if (!gpu) return null;
    st.device = gpu.device;
    st.renderer = new Renderer(gpu.device, gpu.ctx, gpu.format);
    st.renderer.particlesOn = false;
    st.renderer.aperture = 0;
    st.renderer.bloomStrength = 0.35;
    st.renderer.renderScale = 0.65;
    st.resize();
    addEventListener('resize', () => st.resize());
    return st;
  }

  /**
   * Show `organ` as the living volume, centred on `center` (the clicked mesh). `mirror`
   * flips it left-right: one kidney / lung model serves both sides of the body.
   */
  async useOrgan(organ: string, center: V3, mirror: boolean) {
    let o = this.organs.get(organ);
    if (!o) {
      const scene = await Scene.load(this.device, organ);
      let diagnosis: Diagnosis | null = null;
      if (scene.organ.meta.hidden > 0) {
        diagnosis = new Diagnosis(this.device, scene.sim, new Simulation(scene.kernels, 'forecast'), scene.metrics, scene.organ.meta.hilum);
        diagnosis.onProgress = (p) => this.onDiagProgress?.(p);
        diagnosis.onResult = (r) => this.onDiagResult?.(r);
      }
      o = { scene, diagnosis };
      this.organs.set(organ, o);
    }
    if (this.scene !== o.scene) {
      this.diagnosis?.cancel();
      // the comparison volumes belong to the previous organ
      this.baseSim?.destroy(); this.baseSim = null;
      this.learnSim?.destroy(); this.learnSim = null; this.learnKernels = null;
      this.ckptCache.clear();
    }
    this.scene = o.scene;
    this.diagnosis = o.diagnosis;
    this.tissue = TISSUE[organ] ?? TISSUE.liver;
    const h = o.scene.organ.meta.world?.half ?? 0.13;
    this.toWorld.set([mirror ? -h : h, 0, 0, 0, 0, h, 0, 0, 0, 0, h, 0, center[0], center[1], center[2], 1]);
    this.reset();
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
    const sim = this.scene.sim;
    sim.setFromLabels(this.scene.organ.labels);
    if (this.scene.organ.meta.hidden > 0) {
      // let the AI settle on the healthy organ first, exactly as in training
      const enc = this.device.createCommandEncoder();
      sim.setStepIndex(0);
      sim.encodeSteps(enc, 60);
      this.device.queue.submit([enc.finish()]);
    }
    this.diagnosis?.cancel();
    sim.uploadAux(null);
    this.diag = [0, 0];
    this.stopRepair();
    this.exitViews();
    this.cutOps = [];
    this.lastResult = null;
    this.healthy = null;
    this.lesioned = false;
    this.diagPending = false;
    this.scene.latestSummary = null;
    this.cutAge = -1;
    this.needPack = true;
    this.lastVP.fill(NaN);
  }

  /** Show the diagnosis overlay; when repairable, preview the plan then repair live. */
  showDiagnosis(r: DiagnosisResult) {
    this.lastResult = r;
    this.scene.sim.uploadAux(r.aux);
    this.diag = r.repairable ? [1, 0] : [0.6, 1];
    this.lastVP.fill(NaN);
    if (r.repairable) {
      this.tracker = new RepairTracker(r);
      this.mode = 'plan';
      this.modeT = 0;
      this.onPlan?.();
    }
  }

  stopRepair() {
    this.mode = 'idle';
    this.tracker = null;
    this.ghost = 0;
  }

  /** A lesion has been analysed and (if repairable) repaired: the comparisons can run. */
  get canCompare() { return !!this.lastResult && (this.mode === 'done' || !this.lastResult.repairable); }

  /** "Without AI": the AI's repair beside a non-learned hole filling of the same lesion. */
  async startCompare() {
    const r = this.lastResult;
    if (!r) return;
    const token = ++this.viewToken;
    this.view = 'compare';
    this.baseSim ??= new Simulation(this.scene.kernels, 'baseline');
    const frames = holeFillFrames(r.postCut, this.scene.n);
    const ai = this.scene.latest!;
    for (let f = 0; f < frames.length; f++) {
      this.baseSim.setFromLabels(frames[f]);
      const enc = this.device.createCommandEncoder();
      this.baseSim.encodePack(enc);
      this.device.queue.submit([enc.finish()]);
      const base = this.scene.metrics.compute(asSummary(frames[f]));
      this.onCompare?.({ ai, base, frame: f, frames: frames.length });
      await new Promise((res) => setTimeout(res, f === 0 ? 700 : 160));
      if (token !== this.viewToken) return;
    }
  }

  /** "Watch the AI learn": the same cut, repaired by the model saved at each stage of training. */
  async startLearn() {
    const its = this.scene.organ.meta.checkpoints ?? [];
    if (!its.length || !this.cutOps.length) return;
    const token = ++this.viewToken;
    this.view = 'learn';
    this.learnKernels ??= new SimKernels(this.device, this.scene.organ);
    this.learnSim ??= new Simulation(this.learnKernels, 'learn');
    const sim = this.learnSim;
    const rows: LearnRow[] = its.map((iteration) => ({ iteration, metrics: null, step: 0, done: false }));
    const emit = (current: number, finished = false) => this.onLearn?.({ rows, current, finished });
    for (let k = 0; k < its.length; k++) {
      let w = this.ckptCache.get(its[k]);
      if (!w) { w = await loadCheckpoint(this.scene.organ.meta.name, its[k]); this.ckptCache.set(its[k], w); }
      if (token !== this.viewToken) return;
      this.learnKernels.loadWeights(w);
      // same starting point as the live organ: healthy, settled by this model, then the same cuts
      sim.setFromLabels(this.scene.organ.labels);
      sim.setStepIndex(0);
      sim.writeParams(1234);
      let enc = this.device.createCommandEncoder();
      sim.encodeSteps(enc, 60);
      this.device.queue.submit([enc.finish()]);
      for (const op of this.cutOps) sim.queueTool(op);
      sim.encodeTools();
      for (let step = 0; step <= LEARN_STEPS; step += 2) {
        enc = this.device.createCommandEncoder();
        if (step > 0) sim.encodeSteps(enc, 2);
        sim.encodePack(enc);
        const read = step % 10 === 0 ? sim.readSummary(enc) : null;
        this.device.queue.submit([enc.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        if (read) {
          const sum = await read();
          rows[k].metrics = this.scene.metrics.compute(sum);
          rows[k].step = step;
        }
        if (token !== this.viewToken) return;
        emit(k);
        await new Promise((res) => requestAnimationFrame(() => res(null)));
        if (token !== this.viewToken) return;
      }
      rows[k].done = true;
      emit(k);
      await new Promise((res) => setTimeout(res, 900));
      if (token !== this.viewToken) return;
    }
    emit(its.length - 1, true);
  }

  /** Screen position of the left (AI) or right (hole filling) organ in the comparison. */
  screenOf(which: 'main' | 'piece'): [number, number] {
    const w = which === 'main' ? this.mainWorld : this.pieceWorld;
    const m = this.cam.viewProj;
    const top = (this.scene.organ.meta.world?.half ?? 0.13) * 0.85;
    const p: V3 = [w[12], w[13] + top, w[14]];
    const cw = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
    const x = (m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / cw;
    const y = (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / cw;
    return [(x * 0.5 + 0.5) * innerWidth, (0.5 - y * 0.5) * innerHeight];
  }

  exitViews() {
    this.viewToken++;
    this.view = 'live';
    this.lastVP.fill(NaN);
  }

  setBrain(on: boolean) { this.brain = on; this.lastVP.fill(NaN); }

  syncCamera(c: THREE.PerspectiveCamera, target: THREE.Vector3) {
    c.updateMatrixWorld();
    const vp = new Float32Array(16);
    const m = c.projectionMatrix.clone().multiply(c.matrixWorldInverse);
    vp.set(m.elements);
    this.cam.viewProj = vp;
    this.cam.invViewProj = invert(vp);
    this.cam.eye = [c.position.x, c.position.y, c.position.z];
    this.cam.dist = c.position.distanceTo(target);
    this.target = [target.x, target.y, target.z];
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
    // blade plane contains the two camera rays at the stroke's entry and exit (in voxel space)
    const inv = invert(this.toWorld);
    const vox = (d: V3) => v3.norm(transformDir(inv, d));
    const da = vox(v3.norm(v3.sub(first.world, eye))), db = vox(v3.norm(v3.sub(lastH.world, eye)));
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
    const op: ToolOp = {
      kind: TOOL_WEDGE,
      p0: [e[0], e[1], e[2], depth],
      p1: [nrm[0], nrm[1], nrm[2], w0],
      p3: [vDir[0], vDir[1], vDir[2], s0],
      p4: [uDir[0], uDir[1], uDir[2], s1],
    };
    this.scene.sim.queueTool(op);
    this.cutOps.push(op);
    this.cutAge = 0;
    this.flash = 0.35;
    this.lesioned = true;
    this.diagPending = true;
    this.diag = [0, 0];
    this.stopRepair();
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
    // plan: the hologram shows the predicted repair for a moment, then the AI repairs live
    this.modeT += dt;
    if (this.mode === 'plan' && this.modeT > 2.8) {
      this.mode = 'repair';
      this.modeT = 0;
      this.repairStart = sc.sim.steps;
      sc.paused = false;
    }
    const ghostGoal = this.mode === 'plan' || this.mode === 'repair' ? 1 : 0;
    this.ghost += Math.sign(ghostGoal - this.ghost) * Math.min(Math.abs(ghostGoal - this.ghost), dt * (ghostGoal ? 1.8 : 0.8));
    if (this.mode === 'repair') sc.speed = liveStepsPerSec(sc.sim.steps - this.repairStart) / 30;
    const steps = this.mode === 'repair' ? sc.stepsThisFrame(dt) : 0;
    const simChanged = this.needPack || sc.sim.pendingTools > 0 || steps > 0;
    const vp = this.cam.viewProj;
    const camChanged = vp.some((v, i) => v !== this.lastVP[i]);
    const animating = this.view !== 'live' || this.flash > 0 || (this.cutAge >= 0 && this.cutAge < 1.5) || this.mode === 'plan' || this.mode === 'repair' || this.ghost > 0;
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
        mode: this.brain ? 3 : 0, time: this.time, hiddenGain: 1, signalGain: 0, pulseFront: 2, pulseAmt: 0,
        clip: { on: false, n: [0, 0, 1], d: 0 }, flash: this.flash * 0.25, exposure: 1.0,
        growthRate: 0, debrisRate: 0, sparks: { a: [0, 0, 0], b: [0, 0, 0], on: false }, pieceVel: [0, 0, 0],
      };
      // side by side: the AI's repair on the left, hole filling on the right
      let mainWorld = this.toWorld;
      let piece = null;
      if (this.view === 'compare' && this.baseSim) {
        const e = this.cam.eye, t = this.target;
        const f = v3.norm(v3.sub(t, e));
        const right = v3.norm(v3.cross(f, [0, 1, 0]));
        const dx = (this.scene.organ.meta.world?.half ?? 0.13) * 1.12;
        this.mainWorld.set(this.toWorld); this.pieceWorld.set(this.toWorld);
        const up = v3.norm(v3.cross(right, f));   // lift both organs above the results bar
        for (let k = 0; k < 3; k++) {
          this.mainWorld[12 + k] += -right[k] * dx + up[k] * dx * 0.45;
          this.pieceWorld[12 + k] += right[k] * dx + up[k] * dx * 0.45;
        }
        mainWorld = this.mainWorld;
        piece = { sim: this.baseSim, toWorld: this.pieceWorld, tissue: this.tissue };
      }
      const shown = this.view === 'learn' && this.learnSim ? this.learnSim : sc.sim;
      const live = this.view === 'live';
      this.renderer.render(enc, this.cam, {
        sim: shown, toWorld: mainWorld, tissue: this.tissue,
        diag: !live ? [0, 0] : this.mode === 'idle' ? this.diag : this.mode === 'done' ? [0, 0] : [this.diag[0] * (1 - this.ghost), this.diag[1]],
        ghostAmt: live ? this.ghost : 0, ghostTint: [0.35, 0.62, 1.0, 1],
        cuts: live && this.cutAge >= 0 ? [{ n: [0, 0, 1], d: 0, age: this.cutAge }] : [],
      }, piece, opts, dt);
      this.device.queue.submit([enc.finish()]);
      const wasLesioned = this.lesioned;
      const tracker = this.mode === 'repair' ? this.tracker : null;
      const repairStep = sc.sim.steps - this.repairStart;
      read?.().then(() => {
        if (!wasLesioned && !this.lesioned) this.healthy = sc.latestSummary;
        // measure the live repair and detect its phases
        if (tracker && tracker === this.tracker && sc.latestSummary && sc.latest && !tracker.finished) {
          const st = tracker.update(repairStep, sc.latestSummary, sc.latest);
          this.onRepair?.(st);
          if (st.finished) {
            this.mode = 'done';
            this.onRepairDone?.(tracker.outcome(), tracker.forecast);
          }
        }
      });
      this.lastVP.set(vp);
      // the cut is now applied on the GPU: let the AI analyse it
      if (this.diagPending && sc.sim.pendingTools === 0 && this.diagnosis && this.healthy) {
        this.diagPending = false;
        void this.diagnosis.run(this.healthy);
      }
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
