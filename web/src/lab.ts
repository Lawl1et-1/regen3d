// Lab mode: free viewer with scalpel / slice / toxin tools, six view modes,
// live HUD with sparklines, recording, cinematic orbit and time-lapse strip.
import type { App, Controller } from './app';
import { StripCapture } from './capture';
import { mat4, v3, type V3 } from './math';
import { MODES, type Cut, type FrameOpts, type Injury } from './renderer';
import type { Scene } from './scene';
import { TOOL_STROKE, TOOL_TOXIN } from './sim';
import { planeOp, planeToVolume, raycastTissue, sideCounts, strokePlane } from './tissue';
import { h, ICON, pct, Sparkline, toast } from './ui';

type Tool = 'orbit' | 'scalpel' | 'slice' | 'toxin';

export class Lab implements Controller {
  private root!: HTMLElement;
  private tool: Tool = 'scalpel';
  private radius = 3;
  mode = 0;
  private hiddenSel = 0;
  private clipN: V3 = [0, 0, 1];
  private clipD = 0;
  private injuries: Injury[] = [];
  private cuts: Cut[] = [];
  private signal = 0.25;
  private drag: { kind: 'orbit' | 'pan' | 'cut' | 'slice'; x: number; y: number; sx: number; sy: number; last?: V3; moved: boolean } | null = null;
  private hover: { x: number; y: number } | null = null;
  private cinematic = false;
  private strip = new StripCapture();
  private spark: Record<string, Sparkline> = {};
  private vals: Record<string, HTMLElement> = {};
  private els: Record<string, HTMLElement> = {};
  private pulse = { front: 2, amt: 0, lastConn: 0 };
  private readonly toWorld = mat4();

  constructor(private app: App, private scene: Scene) {}

  mount(ui: HTMLElement) {
    this.root = h('div');
    ui.appendChild(this.root);
    this.buildUI();
    this.setMode(0);
    this.app.camera.reset();
  }

  unmount() { this.root.remove(); this.strip.active = false; }

  // ------------------------------------------------------------------ UI
  private buildUI() {
    const r = this.root;
    const top = h('div', { class: 'topbar glass' });
    top.innerHTML = `<div class="brand">REGEN<b>3D</b></div><div class="organ-tag">${this.scene.name} · ${this.scene.n}³ · ${this.scene.organ.meta.trained_iters} it</div>`;
    top.appendChild(this.app.navSeg('lab'));
    r.appendChild(top);

    const rail = h('div', { class: 'rail glass' });
    const tools: [Tool, string, string][] = [
      ['orbit', ICON.orbit, 'Orbit (O)'], ['scalpel', ICON.scalpel, 'Scalpel — drag on tissue (S)'],
      ['slice', ICON.slice, 'Slice — drag a line across (X)'], ['toxin', ICON.toxin, 'Toxin — click to inject (T)'],
    ];
    for (const [id, icon, tip] of tools) {
      const b = h('button', { class: 'tool', 'data-tool': id, 'aria-label': tip }, `${icon}<span class="tip">${tip}</span>`);
      b.onclick = () => this.setTool(id);
      rail.appendChild(b);
    }
    rail.appendChild(h('div', { class: 'sep' }));
    const rad = h('input', { class: 'radius', type: 'range', min: '1', max: '8', step: '0.5', value: String(this.radius), 'aria-label': 'Brush radius' }) as HTMLInputElement;
    const rl = h('div', { class: 'rlabel' }, `r ${this.radius}`);
    rad.oninput = () => { this.radius = +rad.value; rl.textContent = `r ${this.radius}`; };
    rail.append(rad, rl);
    r.appendChild(rail);
    this.setTool(this.tool);

    const tr = h('div', { class: 'transport glass' });
    const btn = (icon: string, label: string, fn: () => void) => {
      const b = h('button', { class: 'ibtn', title: label, 'aria-label': label }, icon);
      b.onclick = fn;
      return b;
    };
    const play = btn(ICON.pause, 'Pause / play (Space)', () => this.togglePause());
    this.els.play = play;
    tr.append(
      btn(ICON.seed, 'Regrow from a single seed cell (R)', () => this.reset(false)),
      btn(ICON.reset, 'Reset to a grown organ (G)', () => this.reset(true)),
      play,
      btn(ICON.step, 'Single step (.)', () => this.scene.stepOnce()),
      h('div', { class: 'vsep' }),
    );
    const sp = h('div', { class: 'speed' });
    const speeds = [0.25, 0.5, 1, 2, 4, 8];
    const si = h('input', { type: 'range', min: '0', max: '5', step: '1', value: '2', 'aria-label': 'Speed' }) as HTMLInputElement;
    const so = h('output', {}, '1x');
    si.oninput = () => { this.scene.speed = speeds[+si.value]; so.textContent = `${speeds[+si.value]}x`; };
    sp.append('speed', si, so);
    tr.appendChild(sp);
    tr.appendChild(h('div', { class: 'vsep' }));
    const modes = h('div', { class: 'seg' });
    MODES.forEach((m, i) => {
      const b = h('button', { 'data-mode': String(i) }, `<kbd>${i + 1}</kbd>${m}`);
      b.onclick = () => this.setMode(i);
      modes.appendChild(b);
    });
    tr.appendChild(modes);
    r.appendChild(tr);

    const hud = h('div', { class: 'hud glass' });
    hud.appendChild(h('h3', {}, 'Tissue state <span id="hud-step"></span>'));
    const rows: [string, string, string][] = [
      ['iou', 'Shape (3D IoU)', '#5fe1ff'], ['func', 'Function (fed parenchyma)', '#6be6a6'],
      ['conn', 'Vessels linked to hilum', '#ff3d52'], ['alive', 'Living cells', '#ffbe55'],
    ];
    for (const [k, label, color] of rows) {
      const m = h('div', { class: 'metric' });
      const v = h('div', { class: 'v' }, '–');
      const sl = new Sparkline(color, 0, k === 'alive' ? 1.1 : 1);
      m.append(h('div', { class: 'k' }, label), v, sl.canvas);
      hud.appendChild(m);
      this.spark[k] = sl;
      this.vals[k] = v;
    }
    const foot = h('div', { class: 'foot' }, '<span id="hud-inj">no injury yet</span><span id="hud-fps"></span>');
    hud.appendChild(foot);
    r.appendChild(hud);
    this.els.step = hud.querySelector('#hud-step')!;
    this.els.inj = hud.querySelector('#hud-inj')!;
    this.els.fps = hud.querySelector('#hud-fps')!;
    if (!this.scene.organ.meta.hilum) { hud.querySelectorAll('.metric')[1].remove(); hud.querySelectorAll('.metric')[1].remove(); }

    const cap = h('div', { class: 'capture glass' });
    const rec = btn(ICON.rec, 'Record WebM', () => this.app.recorder.toggle());
    this.app.recorder.onChange = (on) => rec.classList.toggle('on', on);
    const cin = btn(ICON.film, 'Cinematic orbit (C)', () => this.toggleCinematic());
    this.els.cin = cin;
    const st = btn(ICON.strip, 'Time-lapse strip: next 300 steps', () => {
      this.strip.begin(this.scene.sim.steps);
      st.classList.add('on');
      toast('Capturing 6 frames over the next 300 steps…');
    });
    this.strip.onDone = () => { st.classList.remove('on'); toast('Time-lapse strip saved'); };
    cap.append(rec, cin, st);
    r.appendChild(cap);

    this.els.side = h('div');
    r.appendChild(this.els.side);
  }

  private setTool(t: Tool) {
    this.tool = t;
    this.root.querySelectorAll('.tool').forEach((b) => b.classList.toggle('on', (b as HTMLElement).dataset.tool === t));
    this.app.canvas.style.cursor = t === 'orbit' ? 'grab' : 'crosshair';
  }

  setMode(i: number) {
    this.mode = i;
    this.root.querySelectorAll('[data-mode]').forEach((b) => b.classList.toggle('on', (b as HTMLElement).dataset.mode === String(i)));
    const side = this.els.side;
    side.innerHTML = '';
    if (i === 1) {
      const lg = h('div', { class: 'legend glass' });
      for (const [name, col] of [['Capsule (A)', '#f2e073'], ['Parenchyma (B)', '#59bf8c'], ['Vessel (C)', '#eb2938'], ['Drain (D)', '#4d80fa']]) {
        lg.appendChild(h('div', { class: 'row' }, `<span class="sw" style="background:${col}"></span>${name}`));
      }
      side.appendChild(lg);
    } else if (i === 3) {
      const p = h('div', { class: 'panel glass' }, '<label>Hidden channel</label>');
      const chips = h('div', { class: 'chips' });
      for (let c = 0; c < 11; c++) {
        const b = h('button', {}, String(c + 5));
        b.classList.toggle('on', c === this.hiddenSel);
        b.onclick = () => { this.hiddenSel = c; this.scene.sim.setHiddenSel(c); chips.querySelectorAll('button').forEach((x, j) => x.classList.toggle('on', j === c)); };
        chips.appendChild(b);
      }
      p.appendChild(chips);
      p.appendChild(h('div', { class: 'note' }, 'Channels 5–15 carry no target: the rule invents them to coordinate growth. Cut the tissue and watch them light up around the wound.'));
      side.appendChild(p);
    } else if (i === 4) {
      const p = h('div', { class: 'panel glass' }, '<label>Nutrient field</label><div class="note">Steady-state diffusion from vessels connected to the hilum. Starved parenchyma pulses red.</div>');
      side.appendChild(p);
    } else if (i === 5) {
      const p = h('div', { class: 'panel glass' }, '<label>Section depth</label>');
      const s = h('input', { type: 'range', min: '-1', max: '1', step: '0.01', value: String(this.clipD) }) as HTMLInputElement;
      s.oninput = () => { this.clipD = +s.value; };
      const al = h('button', { class: 'textbtn' }, 'Align to view');
      al.onclick = () => this.alignClip();
      p.append(s, h('div', { style: 'height:8px' }), al);
      p.appendChild(h('div', { class: 'note' }, 'H&E-style stained section: nuclei in purple, cytoplasm in pink, vessel lumens with red cells.'));
      side.appendChild(p);
      this.alignClip();
    }
  }

  private alignClip() {
    const c = this.app.camera;
    this.clipN = v3.norm(v3.sub(c.eye, c.target));
  }

  private togglePause() {
    this.scene.paused = !this.scene.paused;
    this.els.play.innerHTML = this.scene.paused ? ICON.play : ICON.pause;
  }

  private toggleCinematic() {
    this.cinematic = !this.cinematic;
    this.els.cin.classList.toggle('on', this.cinematic);
  }

  private reset(grown: boolean) {
    this.scene.reset(grown ? 220 : 0);
    this.injuries = []; this.cuts = [];
    toast(grown ? 'Reset to a grown organ' : 'Planting a single seed cell');
  }

  // ------------------------------------------------------------------ input
  onKey(e: KeyboardEvent) {
    const k = e.key.toLowerCase();
    if (k >= '1' && k <= '6') this.setMode(+k - 1);
    else if (k === ' ') { e.preventDefault(); this.togglePause(); }
    else if (k === '.') this.scene.stepOnce();
    else if (k === 'r') this.reset(false);
    else if (k === 'g') this.reset(true);
    else if (k === 'c') this.toggleCinematic();
    else if (k === 'o') this.setTool('orbit');
    else if (k === 's') this.setTool('scalpel');
    else if (k === 'x') this.setTool('slice');
    else if (k === 't') this.setTool('toxin');
  }

  private pick(x: number, y: number) {
    const ray = this.app.rayAt(x, y);
    return raycastTissue(this.scene.latestSummary, this.scene.n, this.toWorld, ray.o, ray.d);
  }

  onPointer(type: string, e: PointerEvent) {
    const cam = this.app.camera;
    if (type === 'down') {
      const base = { x: e.clientX, y: e.clientY, sx: e.clientX, sy: e.clientY, moved: false };
      if (e.button === 2 || e.shiftKey) { this.drag = { kind: 'pan', ...base }; return; }
      if (this.tool === 'slice') { this.drag = { kind: 'slice', ...base }; return; }
      if (this.tool === 'scalpel' || this.tool === 'toxin') {
        const hit = this.pick(e.clientX, e.clientY);
        if (hit) {
          if (this.tool === 'toxin') { this.inject(hit.vox, hit.local); this.drag = null; return; }
          this.drag = { kind: 'cut', ...base, last: hit.vox };
          this.carve(hit.vox, hit.vox, hit.local);
          return;
        }
      }
      this.drag = { kind: 'orbit', ...base };
      this.app.canvas.style.cursor = 'grabbing';
    } else if (type === 'move') {
      this.hover = { x: e.clientX, y: e.clientY };
      const d = this.drag;
      if (!d) return;
      const dx = e.clientX - d.x, dy = e.clientY - d.y;
      d.x = e.clientX; d.y = e.clientY;
      if (Math.hypot(e.clientX - d.sx, e.clientY - d.sy) > 3) d.moved = true;
      if (d.kind === 'orbit') cam.orbit(dx, dy);
      else if (d.kind === 'pan') cam.pan(dx, dy);
      else if (d.kind === 'cut') {
        const hit = this.pick(e.clientX, e.clientY);
        if (hit && d.last && v3.len(v3.sub(hit.vox, d.last)) > 0.6) {
          this.carve(d.last, hit.vox, hit.local);
          d.last = hit.vox;
        }
      }
    } else if (type === 'up') {
      const d = this.drag;
      this.drag = null;
      this.app.canvas.style.cursor = this.tool === 'orbit' ? 'grab' : 'crosshair';
      if (d?.kind === 'slice' && d.moved) this.slice(d.sx, d.sy, e.clientX, e.clientY);
    }
  }

  private carve(a: V3, b: V3, local: V3) {
    this.scene.sim.queueTool({ kind: TOOL_STROKE, p0: [a[0], a[1], a[2], this.radius], p1: [b[0], b[1], b[2], 0] });
    this.injure(local);
  }

  private inject(vox: V3, local: V3) {
    this.scene.sim.queueTool({ kind: TOOL_TOXIN, p0: [vox[0], vox[1], vox[2], 2.2], amount: 3.2 });
    this.injure(local);
    toast('Toxin injected — the lesion spreads, then the tissue repairs');
  }

  private slice(x0: number, y0: number, x1: number, y1: number) {
    const cam = this.app.camera;
    const a = this.app.rayAt(x0, y0), b = this.app.rayAt(x1, y1);
    const pl = strokePlane(cam.eye, a.d, b.d);
    const sum = this.scene.latestSummary;
    if (!pl || !sum) return;
    const n = this.scene.n;
    let { nl, dLocal, dVox } = planeToVolume(pl.n, pl.p, this.toWorld, n);
    const [pos, neg] = sideCounts(sum, n, nl, dVox);
    if (pos === 0 || neg === 0) { toast('The cut missed the tissue'); return; }
    if (pos > neg) { nl = v3.scale(nl, -1) as V3; dVox = -dVox; dLocal = -dLocal; }
    this.scene.sim.queueTool(planeOp(nl, dVox));
    this.cuts.unshift({ n: nl, d: dLocal, age: 0 });
    this.cuts.length = Math.min(this.cuts.length, 4);
    const mid = this.app.rayAt((x0 + x1) / 2, (y0 + y1) / 2);
    const hit = raycastTissue(sum, n, this.toWorld, mid.o, mid.d);
    this.injure(hit ? hit.local : [0, 0, 0]);
  }

  private injure(local: V3) {
    const last = this.injuries[0];
    if (!last || last.age > 0.25 || v3.len(v3.sub(last.p, local)) > 0.15) {
      this.injuries.unshift({ p: local, age: 0 });
      this.injuries.length = Math.min(this.injuries.length, 4);
    }
    this.scene.markInjury(this.app.time);
    this.signal = 1.6;
  }

  // ------------------------------------------------------------------ frame
  frame(dt: number, time: number) {
    const { app, scene } = this;
    const cam = app.camera;
    if (this.cinematic) { cam.spin(0.22, dt); cam.idle = 0; }
    cam.update(dt, app.aspect, time);
    for (const j of this.injuries) j.age += dt;
    for (const c of this.cuts) c.age += dt;
    this.signal = Math.max(0.25, this.signal - dt * 0.05);

    // vessel reconnection pulse: when connectivity jumps, send a pulse from the hilum
    const m = scene.latest;
    if (m) {
      if (m.connectivity - this.pulse.lastConn > 0.04 && scene.injuryStep >= 0) { this.pulse.front = 0; this.pulse.amt = 1; }
      this.pulse.lastConn = m.connectivity;
    }
    this.pulse.front += dt * 0.45;
    if (this.pulse.front > 1.2) this.pulse.amt = 0;

    const enc = app.device.createCommandEncoder();
    scene.encode(enc, dt);
    const read = scene.maybeRead(enc, time);
    const opts: FrameOpts = {
      mode: this.mode, time, hiddenGain: 1.2, signalGain: this.signal,
      pulseFront: this.pulse.front, pulseAmt: this.pulse.amt,
      clip: { on: true, n: this.clipN, d: this.clipD }, flash: 0, exposure: 1.0,
      growthRate: 0.35, debrisRate: 0, sparks: { a: [0, 0, 0], b: [0, 0, 0], on: false }, pieceVel: [0, 0, 0],
    };
    app.renderer.render(enc, cam, { sim: scene.sim, toWorld: this.toWorld, injuries: this.injuries, cuts: this.cuts }, null, opts, dt);
    app.device.queue.submit([enc.finish()]);
    read?.();
    this.strip.maybeGrab(app.canvas, scene.sim.steps);
    this.drawOverlay();
    this.updateHUD(time);
  }

  private drawOverlay() {
    const g = this.app.overlay;
    this.app.clearOverlay();
    const d = this.drag;
    if (d?.kind === 'slice') {
      g.save();
      g.strokeStyle = 'rgba(95,225,255,0.9)'; g.lineWidth = 2; g.shadowColor = '#5fe1ff'; g.shadowBlur = 12;
      g.setLineDash([8, 6]);
      g.beginPath(); g.moveTo(d.sx, d.sy); g.lineTo(d.x, d.y); g.stroke();
      g.restore();
    }
    if (this.hover && (this.tool === 'scalpel' || this.tool === 'toxin') && !(d && d.kind !== 'cut')) {
      const hit = this.pick(this.hover.x, this.hover.y);
      if (hit) {
        const [px, py, pw] = this.app.camera.project(hit.world);
        const sx = (px * 0.5 + 0.5) * innerWidth, sy = (0.5 - py * 0.5) * innerHeight;
        const r = this.tool === 'toxin' ? 10 : ((this.radius * 2) / this.scene.n) * 1.65 / pw * innerHeight * 0.5;
        g.save();
        g.strokeStyle = this.tool === 'toxin' ? 'rgba(170,255,120,0.9)' : 'rgba(95,225,255,0.85)';
        g.lineWidth = 1.5; g.shadowColor = g.strokeStyle; g.shadowBlur = 8;
        g.beginPath(); g.arc(sx, sy, Math.max(4, r), 0, Math.PI * 2); g.stroke();
        g.restore();
      }
    }
  }

  private updateHUD(time: number) {
    const s = this.scene;
    this.els.step.textContent = `step ${s.sim.steps}`;
    this.els.fps.textContent = `${this.app.fps.toFixed(0)} fps`;
    const m = s.latest;
    if (!m) return;
    const hist = s.history.slice(-160);
    const mark = s.injuryStep >= 0 ? hist.findIndex((x) => x.step >= s.injuryStep) : -1;
    const set = (k: string, val: string, series: number[]) => {
      if (!this.vals[k]?.isConnected) return;
      this.vals[k].textContent = val;
      this.spark[k].draw(series, mark);
    };
    set('iou', pct(m.iou), hist.map((x) => x.m.iou));
    set('func', pct(m.func), hist.map((x) => x.m.func));
    set('conn', pct(m.connectivity), hist.map((x) => x.m.connectivity));
    set('alive', `${m.alive.toLocaleString()}`, hist.map((x) => x.m.alive / Math.max(m.aliveTarget, 1)));
    if (s.injuryStep >= 0) {
      this.els.inj.textContent = `since injury: ${s.sim.steps - s.injuryStep} steps · ${(time - s.injuryTime).toFixed(1)}s`;
    }
  }
}
