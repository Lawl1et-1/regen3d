// Application shell: canvas sizing, frame loop, input routing, routes
// (#/body, #/lab/<organ>, #/slice/<organ>) and shared GPU resources.
import { OrbitCamera } from './camera';
import { Recorder } from './capture';
import type { Gpu } from './gpu';
import { Renderer } from './renderer';
import { Scene } from './scene';
import { h, ICON } from './ui';

export interface Controller {
  mount(ui: HTMLElement): void;
  unmount(): void;
  frame(dt: number, time: number): void;
  onPointer?(type: string, e: PointerEvent): void;
  onKey?(e: KeyboardEvent): void;
  onWheel?(e: WheelEvent): boolean;
}

export interface OrganInfo { name: string; label: string; n: number }

export class App {
  readonly device: GPUDevice;
  readonly canvas: HTMLCanvasElement;
  readonly overlay: CanvasRenderingContext2D;
  readonly camera = new OrbitCamera();
  readonly renderer: Renderer;
  readonly recorder: Recorder;
  readonly ui = document.getElementById('ui')!;
  organs: OrganInfo[] = [];
  private scenes = new Map<string, Scene>();
  private ctrl: Controller | null = null;
  private last = performance.now();
  time = 0;
  fps = 60;
  dprCap = 1.5;
  private routing = false;

  constructor(readonly gpu: Gpu) {
    this.device = gpu.device;
    this.canvas = gpu.canvas;
    const ov = document.getElementById('overlay') as HTMLCanvasElement;
    this.overlay = ov.getContext('2d')!;
    this.renderer = new Renderer(gpu.device, gpu.ctx, gpu.format);
    this.recorder = new Recorder(this.canvas);
    this.resize();
    addEventListener('resize', () => this.resize());
    const c = this.canvas;
    c.addEventListener('pointerdown', (e) => { c.setPointerCapture(e.pointerId); this.ctrl?.onPointer?.('down', e); });
    c.addEventListener('pointermove', (e) => this.ctrl?.onPointer?.('move', e));
    c.addEventListener('pointerup', (e) => this.ctrl?.onPointer?.('up', e));
    c.addEventListener('pointercancel', (e) => this.ctrl?.onPointer?.('up', e));
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (!this.ctrl?.onWheel?.(e)) this.camera.zoom(Math.exp(e.deltaY * 0.0012));
    }, { passive: false });
    addEventListener('keydown', (e) => {
      if ((e.target as HTMLElement).tagName === 'INPUT') return;
      this.ctrl?.onKey?.(e);
    });
    addEventListener('hashchange', () => this.route());
  }

  get aspect() { return this.canvas.width / this.canvas.height; }

  resize() {
    const dpr = Math.min(devicePixelRatio || 1, this.dprCap);
    this.canvas.width = Math.round(innerWidth * dpr);
    this.canvas.height = Math.round(innerHeight * dpr);
    const ov = this.overlay.canvas;
    ov.width = Math.round(innerWidth * (devicePixelRatio || 1));
    ov.height = Math.round(innerHeight * (devicePixelRatio || 1));
    this.overlay.setTransform(devicePixelRatio || 1, 0, 0, devicePixelRatio || 1, 0, 0);
  }

  clearOverlay() { this.overlay.clearRect(0, 0, innerWidth, innerHeight); }

  rayAt(x: number, y: number) {
    return this.camera.ray((x / innerWidth) * 2 - 1, 1 - (y / innerHeight) * 2);
  }

  async scene(name: string) {
    let s = this.scenes.get(name);
    if (!s) {
      s = await Scene.load(this.device, name);
      this.scenes.set(name, s);
    }
    return s;
  }

  navSeg(active: 'body' | 'lab' | 'slice', organ?: string) {
    const seg = h('div', { class: 'seg' });
    const cur = organ ?? (location.hash.split('/')[2] || this.organs[0]?.name);
    const items: [string, string, string][] = [
      ['body', `${ICON.body.replace('<svg', '<svg width="14" height="14"')}Body`, '#/body'],
      ['lab', 'Lab', `#/lab/${cur}`],
      ['slice', 'Slice mode', `#/slice/${cur}`],
    ];
    for (const [id, label, href] of items) {
      const b = h('button', {}, label);
      b.classList.toggle('on', id === active);
      b.onclick = () => { location.hash = href; };
      seg.appendChild(b);
    }
    return seg;
  }

  async start() {
    try {
      this.organs = await fetch(`${import.meta.env.BASE_URL}organs/index.json`).then((r) => r.json());
    } catch {
      this.organs = [];
    }
    await this.route();
    const loop = (now: number) => {
      const dt = Math.min(0.1, (now - this.last) / 1000);
      this.last = now;
      this.time += dt;
      this.fps = this.fps * 0.95 + (1 / Math.max(dt, 1e-3)) * 0.05;
      this.ctrl?.frame(dt, this.time);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  async route() {
    if (this.routing) return;
    this.routing = true;
    try {
      const [, kind, name] = location.hash.split('/');
      const organ = this.organs.find((o) => o.name === name)?.name ?? this.organs.find((o) => o.name !== 'blob')?.name ?? this.organs[0]?.name;
      if (!organ) throw new Error('No trained organs found in /organs. Run train/export.py first.');
      const loading = h('div', { class: 'loading' }, 'LOADING ORGAN…');
      this.ui.appendChild(loading);
      let next: Controller;
      if (kind === 'slice') {
        const { SliceMode } = await import('./slice');
        next = new SliceMode(this, await this.scene(organ));
      } else if (kind === 'lab' || !this.hasBody()) {
        const { Lab } = await import('./lab');
        next = new Lab(this, await this.scene(organ));
      } else {
        const { BodyView } = await import('./body');
        next = new BodyView(this);
      }
      loading.remove();
      this.ctrl?.unmount();
      this.clearOverlay();
      this.ctrl = next;
      next.mount(this.ui);
    } catch (err) {
      console.error(err);
      this.ui.innerHTML = `<div class="fatal"><div class="box glass"><h1>Could not start</h1><p>${(err as Error).message}</p></div></div>`;
    } finally {
      this.routing = false;
    }
  }

  hasBody() { return this.organs.some((o) => o.name !== 'blob'); }
}
