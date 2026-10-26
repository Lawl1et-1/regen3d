import './style.css';
import { BodyView3D, ORGANS, type OrganId } from './body3d';
import { h } from './ui';

const canvas = document.getElementById('three') as HTMLCanvasElement;
const ui = document.getElementById('ui')!;
const view = new BodyView3D(canvas);

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
  hint.textContent = 'Drag to rotate the organ · scroll to zoom';
  focusPanel.innerHTML = `<div class="fp-k">Selected organ</div><div class="fp-v">${ORGANS[id].label}</div>`;
  const back = h('button', { class: 'pill' }, '← Back to body');
  back.onclick = () => unselect();
  focusPanel.appendChild(back);
  focusPanel.classList.add('show');
}

function unselect() {
  view.unfocus();
  top.classList.remove('dim');
  focusPanel.classList.remove('show');
  hint.textContent = 'Drag to rotate · scroll to zoom · click an organ';
}

// ---------------------------------------------------------------- input
let down: { x: number; y: number } | null = null;
canvas.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; });
canvas.addEventListener('pointermove', (e) => {
  if (view.focused || view.tweening) return;
  const id = view.pick(e.clientX, e.clientY);
  view.hovered = id;
  canvas.style.cursor = id ? 'pointer' : 'grab';
});
canvas.addEventListener('pointerup', (e) => {
  if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) { down = null; return; }
  down = null;
  const id = view.pick(e.clientX, e.clientY);
  if (id) select(id);
});
addEventListener('keydown', (e) => {
  if (e.key === 'x' || e.key === 'X') setXray(view.xrayTarget < 0.5);
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
};

view.load(`${import.meta.env.BASE_URL}models/body.glb`)
  .then(() => {
    const loop = () => { view.frame(); requestAnimationFrame(loop); };
    loop();
  })
  .catch((err) => {
    console.error(err);
    ui.appendChild(h('div', { class: 'fatal' }, `<div class="box glass"><h1>Could not load the body model</h1><p>${(err as Error).message}</p></div>`));
  });
