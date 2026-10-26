import './style.css';
import { BodyView3D, ORGANS, type OrganId } from './body3d';
import { AnalysisPanel } from './analysis';
import { OrganStage } from './organstage';
import { PHASES, type RepairState } from './repair';
import type { Metrics } from './metrics';
import type { LearnProgress } from './organstage';
import { h } from './ui';

const canvas = document.getElementById('three') as HTMLCanvasElement;
const gpuCanvas = document.getElementById('gpu') as HTMLCanvasElement;
const trailCanvas = document.getElementById('trail') as HTMLCanvasElement;
const ui = document.getElementById('ui')!;
const view = new BodyView3D(canvas);
if (import.meta.env.DEV) (window as unknown as { __view: BodyView3D }).__view = view;

/**
 * Each organ's trained model and the mesh it was trained on. Kidneys and lungs come in
 * pairs: the other side shows the same model mirrored.
 */
const MODELS: Record<OrganId, { model: string; mesh: string }> = {
  liver: { model: 'liver', mesh: 'liver' },
  heart: { model: 'heart', mesh: 'heart' },
  kidney: { model: 'kidney', mesh: 'kidney_left' },
  lung: { model: 'lung', mesh: 'lung_right' },
};
/** Organs whose AI is trained and exported (organs/index.json). */
let CUTTABLE: OrganId[] = [];
let stage: OrganStage | null = null;
let stageLoading: Promise<OrganStage | null> | null = null;
let stageOn = false;
/** the organ volume currently loaded in the stage (it switches when another organ is selected) */
let stageOrgan: string | null = null;

// ---------------------------------------------------------------- UI
const top = h('div', { class: 'topbar glass' }, '<div class="brand">REGEN<b>3D</b></div><div class="organ-tag">AI organ repair</div>');
const xrayBtn = h('button', { class: 'pill', 'aria-pressed': 'false' }, '<span class="dot"></span>X-ray');
xrayBtn.onclick = () => setXray(view.xrayTarget < 0.5);
top.appendChild(xrayBtn);
ui.appendChild(top);

const hint = h('div', { class: 'hint' }, 'Drag to rotate · scroll to zoom · click an organ');
ui.appendChild(hint);

// ---------------------------------------------------------------- proving the AI does the repair
const proof = h('div', { class: 'proof glass' });
ui.appendChild(proof);
const tagL = h('div', { class: 'cmp-tag ai' }, 'With AI');
const tagR = h('div', { class: 'cmp-tag noai' }, 'Without AI');
ui.append(tagL, tagR);
let viewOpen: 'compare' | 'learn' | null = null;
/** camera pull-back per view, so the organ(s) clear the results bar */
const ZOOM = { compare: 1.9, learn: 1.3 };
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function enableProofs() {
  const ok = !!stage?.canCompare;
  for (const id of ['btn-noai', 'btn-learn']) (document.getElementById(id) as HTMLButtonElement | null)?.toggleAttribute('disabled', !ok);
}

function openView(v: 'compare' | 'learn') {
  if (!stage || !stage.canCompare) return;
  if (viewOpen) closeView();
  viewOpen = v;
  document.body.classList.add('proofing');
  analysis.hide();
  showPhases(null);
  proof.innerHTML = '<p class="ai-note">Preparing…</p>';
  proof.classList.add('show');
  view.zoomBy(ZOOM[v]);
  if (v === 'compare') void stage.startCompare(); else void stage.startLearn();
}

function closeView() {
  if (!viewOpen) return;
  view.zoomBy(1 / ZOOM[viewOpen]);
  viewOpen = null;
  document.body.classList.remove('proofing');
  stage?.exitViews();
  proof.classList.remove('show');
  tagL.classList.remove('show'); tagR.classList.remove('show');
  analysis.restore();
}

function backBtn() {
  const b = h('button', { class: 'pill' }, '← Back to the organ');
  b.onclick = () => closeView();
  return b;
}

function showCompare(c: { ai: Metrics; base: Metrics; frame: number; frames: number }) {
  if (viewOpen !== 'compare') return;
  const row = (k: string, a: number, b: number) =>
    `<tr><th>${k}</th><td class="${a > 0.9 ? 'good' : 'poor'}">${pct(a)}</td><td class="${b > 0.9 ? 'good' : 'poor'}">${pct(b)}</td></tr>`;
  const filling = c.frame < c.frames - 1;
  proof.innerHTML = `
    <div class="ai-head"><span class="ai-title">With AI vs without AI</span><span class="ai-chip ${filling ? 'run' : 'ok'}">${filling ? `filling ${c.frame}/${c.frames - 1}` : 'done'}</span></div>
    <p class="ai-lead">Same lesion. Left: repaired by the trained network. Right: automatic hole filling, a simple non-learned method that fills any gap mostly surrounded by tissue.</p>
    <table class="fva"><thead><tr><th></th><th>With AI</th><th>Without AI</th></tr></thead><tbody>
      ${row('Shape recovered', c.ai.iou, c.base.iou)}
      ${row('Vessels linked to hilum', c.ai.connectivity, c.base.connectivity)}
    </tbody></table>`;
  proof.appendChild(backBtn());
  tagL.classList.add('show'); tagR.classList.add('show');
}

function showLearn(l: LearnProgress) {
  if (viewOpen !== 'learn') return;
  const rows = l.rows.map((r, i) => {
    const m = r.metrics;
    const ok = m ? m.iou > 0.9 && m.connectivity > 0.9 : false;
    const state = r.done ? (ok ? 'ok' : 'bad') : i === l.current ? 'run' : '';
    const nums = m ? `${pct(m.iou)} · ${pct(m.connectivity)}` : '—';
    return `<li class="${state}"><span class="roll ${state}"></span><b>iteration ${r.iteration}</b><span class="nums">${nums}</span><span class="st">${r.done ? (ok ? 'repairs' : 'fails') : i === l.current ? `step ${r.step}` : ''}</span></li>`;
  }).join('');
  proof.innerHTML = `
    <div class="ai-head"><span class="ai-title">Watch the AI learn</span><span class="ai-chip ${l.finished ? 'ok' : 'run'}">${l.finished ? 'done' : `stage ${l.current + 1}/${l.rows.length}`}</span></div>
    <p class="ai-lead">The same cut, repaired by the network saved at each stage of training (shape · vessels).</p>
    <ol class="learn">${rows}</ol>`;
  proof.appendChild(backBtn());
}

const brainNote = h('div', { class: 'brain-note glass' },
  '<b>AI Brain</b>Each cell carries 11 hidden numbers that no one designed: the network invented them to talk to its neighbours. '
  + 'Colours show three of them; brightness shows where they are changing. Cut the organ and watch the signal spread from the wound.');
ui.appendChild(brainNote);
const vesselNote = h('div', { class: 'brain-note glass' },
  '<b>Vessels</b>Tissue only grows where blood reaches it. Red: vessels connected to the hilum, with blood pulses flowing out. '
  + 'Cyan: vessels the AI is regrowing in the wound. Pale tissue is starving: it regrows only once a vessel reaches it, and dies if none does.');
ui.appendChild(vesselNote);

const label = h('div', { class: 'organ-label' });
ui.appendChild(label);

const focusPanel = h('div', { class: 'focus-panel glass' });
ui.appendChild(focusPanel);

const analysis = new AnalysisPanel(ui);

// repair phases, bottom of the screen (each one lights up when the simulation reaches it)
const phases = h('div', { class: 'phases glass' });
phases.innerHTML = `<ol>${PHASES.map((p, i) => `<li data-i="${i}"><span class="pdot"></span>${p}</li>`).join('<li class="pline" aria-hidden="true"></li>')}</ol><div class="pwhy"></div><div class="pstat"></div>`;
ui.appendChild(phases);
/** What happens in each phase (shown under the timeline while it is active). */
const PHASE_WHY = [
  'Cells at the cut edges divide and close the open wound first.',
  'Vessel stumps sprout and reconnect to the tree, so blood can reach the wound.',
  'Cells fill the gap behind the new vessels, only where blood arrives.',
  'The shape is refined until it stops changing.',
];

function showPhases(st: RepairState | null) {
  phases.classList.toggle('show', !!st);
  if (!st) return;
  phases.querySelectorAll<HTMLElement>('li[data-i]').forEach((li) => {
    const i = +li.dataset.i!;
    li.classList.toggle('done', st.done[i]);
    li.classList.toggle('active', !st.finished && i === st.active);
  });
  const sm = st.sample;
  phases.querySelector('.pwhy')!.textContent = st.finished ? '' : PHASE_WHY[st.active] ?? '';
  phases.querySelector('.pstat')!.textContent = st.finished
    ? `Repair complete · ${st.step} steps · shape ${(sm.iou * 100).toFixed(1)}% · vessels ${(sm.connectivity * 100).toFixed(1)}%`
    : `step ${st.step} · wound closed ${(sm.mouth * 100).toFixed(0)}% · refilled ${(sm.fill * 100).toFixed(0)}% · vessels ${(sm.connectivity * 100).toFixed(1)}%`;
}

function setXray(on: boolean) {
  view.xrayTarget = on ? 1 : 0;
  xrayBtn.classList.toggle('on', on);
  xrayBtn.setAttribute('aria-pressed', String(on));
}

function select(id: OrganId) {
  const node = view.pickedNode ?? MODELS[id].mesh;
  view.focus(id, node);
  stageOrgan = null;
  stage?.setBrain(false);
  stage?.setVessels(false);
  brainNote.classList.remove('show');
  vesselNote.classList.remove('show');
  label.classList.remove('show');
  top.classList.add('dim');
  const cuttable = CUTTABLE.includes(id);
  hint.textContent = cuttable
    ? 'Swipe across the organ to make a lesion · right-drag or two fingers to rotate · scroll to zoom'
    : 'Drag to rotate the organ · scroll to zoom';
  focusPanel.innerHTML = `<div class="fp-k">Selected organ</div><div class="fp-v">${ORGANS[id].label}</div>`;
  const actions = h('div', { class: 'fp-actions' });
  if (cuttable) {
    actions.appendChild(h('div', { class: 'fp-cut', id: 'cut-info' }, 'No lesion yet'));
    const reset = h('button', { class: 'pill' }, 'Reset organ');
    reset.onclick = () => { closeView(); stage?.reset(); setCutInfo(null); analysis.hide(); showPhases(null); enableProofs(); };
    const brain = h('button', { class: 'pill brain', 'aria-pressed': 'false' }, '<span class="dot"></span>AI Brain');
    const vessels = h('button', { class: 'pill', 'aria-pressed': 'false' }, '<span class="dot"></span>Vessels');
    const syncViews = () => {
      for (const [b, on] of [[brain, !!stage?.brain], [vessels, !!stage?.vessels]] as const) {
        b.classList.toggle('on', on);
        b.setAttribute('aria-pressed', String(on));
      }
      brainNote.classList.toggle('show', !!stage?.brain);
      vesselNote.classList.toggle('show', !!stage?.vessels);
    };
    brain.onclick = () => { stage?.setBrain(!stage.brain); syncViews(); };
    vessels.onclick = () => { stage?.setVessels(!stage.vessels); syncViews(); };
    actions.append(vessels, brain);
    const noAi = h('button', { class: 'pill', id: 'btn-noai', disabled: '' }, 'Without AI');
    noAi.onclick = () => openView('compare');
    const learn = h('button', { class: 'pill', id: 'btn-learn', disabled: '' }, 'Watch the AI learn');
    learn.onclick = () => openView('learn');
    actions.append(noAi, learn);
    actions.appendChild(reset);
    prepareStage(id, node);
  } else {
    actions.appendChild(h('div', { class: 'fp-note' }, 'This organ\'s AI is still training.'));
  }
  const back = h('button', { class: 'pill' }, '← Back to body');
  back.onclick = () => unselect();
  actions.appendChild(back);
  focusPanel.appendChild(actions);
  focusPanel.classList.add('show');
}

function setCutInfo(info: { depthMm: number; lengthMm: number } | null) {
  const el = document.getElementById('cut-info');
  if (el) el.textContent = info ? `Last cut · ${info.lengthMm.toFixed(0)} mm long · ${info.depthMm.toFixed(0)} mm deep` : 'No lesion yet';
}

function unselect() {
  closeView();
  showStage(false);
  analysis.hide();
  showPhases(null);
  view.unfocus();
  top.classList.remove('dim');
  focusPanel.classList.remove('show');
  hint.textContent = 'Drag to rotate · scroll to zoom · click an organ';
}

function prepareStage(id: OrganId, node: string) {
  const m = MODELS[id];
  const load = (s: OrganStage | null) => {
    if (!s) return;
    void s.useOrgan(m.model, view.nodeCenter(node), node !== m.mesh).then(() => {
      if (view.focused === id) stageOrgan = m.model;
    });
  };
  stageLoading ??= OrganStage.create(gpuCanvas, trailCanvas).then((s) => {
    stage = s;
    if (s) {
      s.onCut = (info) => { setCutInfo(info); showPhases(null); enableProofs(); };
      s.onPlan = () => analysis.plan();
      s.onRepair = (st) => showPhases(st);
      s.onRepairDone = (o, f) => { analysis.actual(o, f); enableProofs(); };
      s.onCompare = (c) => showCompare(c);
      s.onLearn = (l) => showLearn(l);
      s.onDiagProgress = (p) => analysis.progress(p);
      s.onDiagResult = (r) => { analysis.result(r); s.showDiagnosis(r); enableProofs(); };
    }
    if (import.meta.env.DEV) (window as unknown as { __stage: OrganStage | null }).__stage = s;
    return s;
  }).catch((e) => { console.error(e); return null; });
  void stageLoading.then(load);
}

function showStage(on: boolean) {
  stageOn = on;
  gpuCanvas.classList.toggle('show', on);
  document.body.classList.toggle('cutting', on);
  view.setCutControls(on);
  // keep drawing the meshes until the volume has faded in, so the swap is seamless
  if (on) setTimeout(() => { if (stageOn) view.renderOn = false; }, 500);
  else view.renderOn = true;
}

// ---------------------------------------------------------------- input
let down: { x: number; y: number } | null = null;
let blade = false;
canvas.addEventListener('pointerdown', (e) => {
  down = { x: e.clientX, y: e.clientY };
  if (stageOn && stage && !viewOpen && e.button === 0 && e.isPrimary) {
    blade = true;
    canvas.setPointerCapture(e.pointerId);
    stage.beginStroke(e.clientX, e.clientY);
  }
});
canvas.addEventListener('pointermove', (e) => {
  if (blade && stage) { stage.moveStroke(e.clientX, e.clientY); return; }
  if (view.focused || view.tweening) return;
  const id = view.pick(e.clientX, e.clientY);
  view.hovered = id;
  canvas.style.cursor = id ? 'pointer' : 'grab';
});
const up = (e: PointerEvent) => {
  if (blade) { blade = false; stage?.endStroke(); down = null; return; }
  if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) { down = null; return; }
  down = null;
  const id = view.pick(e.clientX, e.clientY);
  if (id) select(id);
};
canvas.addEventListener('pointerup', up);
canvas.addEventListener('pointercancel', up);
// a second finger means rotate/zoom, not cut
canvas.addEventListener('touchstart', (e) => { if (e.touches.length > 1 && blade) { blade = false; stage?.endStroke(); } }, { passive: true });
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
addEventListener('keydown', (e) => {
  if ((e.key === 'x' || e.key === 'X') && !view.focused) setXray(view.xrayTarget < 0.5);
  if (e.key === 'Escape' && view.focused) unselect();
});

view.onFrame = () => {
  const id = view.hovered;
  if (id && !view.focused) {
    const [x, y] = view.screenOf(id);
    label.textContent = ORGANS[id].label;
    label.style.transform = `translate(${x}px, ${y}px)`;
    label.classList.add('show');
  } else label.classList.remove('show');

  // once the zoom onto a cuttable organ ends, swap the mesh for the living GPU volume
  const f = view.focused;
  if (!stageOn && stage && f && stageOrgan === MODELS[f].model && !view.tweening) showStage(true);
  if (stage && viewOpen === 'compare') {
    const [lx, ly] = stage.screenOf('main'), [rx, ry] = stage.screenOf('piece');
    tagL.style.transform = `translate(${lx}px, ${ly}px)`;
    tagR.style.transform = `translate(${rx}px, ${ry}px)`;
  }
  if (stage) {
    stage.syncCamera(view.camera, view.controls.target);
    if (stageOn || gpuCanvas.classList.contains('show')) stage.frame();
  }
};

const organList = fetch(`${import.meta.env.BASE_URL}organs/index.json`).then((r) => r.json() as Promise<{ name: string }[]>)
  .then((list) => { CUTTABLE = (Object.keys(MODELS) as OrganId[]).filter((id) => list.some((o) => o.name === MODELS[id].model)); })
  .catch(() => { CUTTABLE = []; });

Promise.all([view.load(`${import.meta.env.BASE_URL}models/body.glb`), organList])
  .then(() => {
    const loop = () => { view.frame(); requestAnimationFrame(loop); };
    loop();
    if (import.meta.env.DEV) {
      // dev shortcuts: #liver, #kidney... opens that organ; __select('kidney', 'kidney_right') picks a side
      (window as unknown as { __select: (id: OrganId, node?: string) => void }).__select = (id, node) => { view.pickedNode = node ?? null; select(id); };
      const dev = location.hash.slice(1) as OrganId;
      if (dev in MODELS) select(dev);
    }
  })
  .catch((err) => {
    console.error(err);
    ui.appendChild(h('div', { class: 'fatal' }, `<div class="box glass"><h1>Could not load the body model</h1><p>${(err as Error).message}</p></div>`));
  });
