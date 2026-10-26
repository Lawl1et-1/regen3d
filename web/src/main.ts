import './style.css';
import { BodyView3D, ORGANS, type OrganId } from './body3d';
import { OrganStage } from './organstage';
import { h } from './ui';

const canvas = document.getElementById('three') as HTMLCanvasElement;
const gpuCanvas = document.getElementById('gpu') as HTMLCanvasElement;
const trailCanvas = document.getElementById('trail') as HTMLCanvasElement;
const ui = document.getElementById('ui')!;
const view = new BodyView3D(canvas);
if (import.meta.env.DEV) (window as unknown as { __view: BodyView3D }).__view = view;

/** Organs with a GPU volume that can be cut (the others come after approval). */
const CUTTABLE: OrganId[] = ['liver'];
let stage: OrganStage | null = null;
let stageLoading: Promise<OrganStage | null> | null = null;
let stageOn = false;

// ---------------------------------------------------------------- UI
const top = h('div', { class: 'topbar glass' }, '<div class="brand">REGEN<b>3D</b></div><div class="organ-tag">AI organ repair</div>');
const xrayBtn = h('button', { class: 'pill', 'aria-pressed': 'false' }, '<span class="dot"></span>X-ray');
xrayBtn.onclick = () => setXray(view.xrayTarget < 0.5);
top.appendChild(xrayBtn);
ui.appendChild(top);

const hint = h('div', { class: 'hint' }, 'Drag to rotate · scroll to zoom · click an organ');
ui.appendChild(hint);

const label = h('div', { class: 'organ-label' });
ui.appendChild(label);

const focusPanel = h('div', { class: 'focus-panel glass' });
ui.appendChild(focusPanel);

function setXray(on: boolean) {
  view.xrayTarget = on ? 1 : 0;
  xrayBtn.classList.toggle('on', on);
  xrayBtn.setAttribute('aria-pressed', String(on));
}

function select(id: OrganId) {
  view.focus(id);
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
    reset.onclick = () => { stage?.reset(); setCutInfo(null); };
    actions.appendChild(reset);
    prepareStage(id);
  } else {
    actions.appendChild(h('div', { class: 'fp-note' }, 'Lesions are available on the liver first; this organ gets its own AI after approval.'));
  }
  const back = h('button', { class: 'pill' }, '← Back to body');
  back.onclick = () => unselect();
  actions.appendChild(back);
  focusPanel.appendChild(actions);
  focusPanel.classList.add('show');
}

function setCutInfo(info: { depthMm: number; lengthMm: number } | null) {
  const el = document.getElementById('cut-info');
  if (el) el.textContent = info ? `Lesion · ${info.lengthMm.toFixed(0)} mm long · ${info.depthMm.toFixed(0)} mm deep` : 'No lesion yet';
}

function unselect() {
  showStage(false);
  view.unfocus();
  top.classList.remove('dim');
  focusPanel.classList.remove('show');
  hint.textContent = 'Drag to rotate · scroll to zoom · click an organ';
}

function prepareStage(id: OrganId) {
  if (!stageLoading) {
    stageLoading = OrganStage.create(gpuCanvas, trailCanvas, id).then((s) => {
      stage = s;
      if (s) s.onCut = (info) => setCutInfo(info);
      if (import.meta.env.DEV) (window as unknown as { __stage: OrganStage | null }).__stage = s;
      return s;
    }).catch((e) => { console.error(e); return null; });
  }
}

function showStage(on: boolean) {
  stageOn = on;
  gpuCanvas.classList.toggle('show', on);
  document.body.classList.toggle('cutting', on);
  view.setCutControls(on);
  // keep drawing the meshes until the volume has faded in, so the swap is seamless
  if (on) setTimeout(() => { if (stageOn) view.renderEnabled = false; }, 500);
  else view.renderEnabled = true;
}

// ---------------------------------------------------------------- input
let down: { x: number; y: number } | null = null;
let blade = false;
canvas.addEventListener('pointerdown', (e) => {
  down = { x: e.clientX, y: e.clientY };
  if (stageOn && stage && e.button === 0 && e.isPrimary) {
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
  if (!stageOn && stage && f && CUTTABLE.includes(f) && !view.tweening) showStage(true);
  if (stage) {
    stage.syncCamera(view.camera, view.controls.target);
    if (stageOn || gpuCanvas.classList.contains('show')) stage.frame();
  }
};

view.load(`${import.meta.env.BASE_URL}models/body.glb`)
  .then(() => {
    const loop = () => { view.frame(); requestAnimationFrame(loop); };
    loop();
    if (import.meta.env.DEV && location.hash === '#liver') select('liver');
  })
  .catch((err) => {
    console.error(err);
    ui.appendChild(h('div', { class: 'fatal' }, `<div class="box glass"><h1>Could not load the body model</h1><p>${(err as Error).message}</p></div>`));
  });
