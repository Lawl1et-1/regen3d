// Small DOM helpers, icons and sparklines.
export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, string> = {}, html = ''): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (html) el.innerHTML = html;
  return el;
}

const svg = (d: string, extra = '') =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" ${extra}>${d}</svg>`;

export const ICON = {
  orbit: svg('<path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 4v5h-5"/>'),
  scalpel: svg('<path d="M3 21l9.5-9.5"/><path d="M12.5 11.5l6.8-6.8a2 2 0 0 1 2.8 2.8L14 15.6c-1 .9-2.4.6-3-.6"/>'),
  slice: svg('<path d="M4 20L20 4"/><path d="M14 4h6v6" opacity=".5"/><path d="M7 7l2 2M5 11l2 2M11 5l2 2" opacity=".55"/>'),
  toxin: svg('<path d="M12 3s6 6.4 6 10.5A6 6 0 0 1 6 13.5C6 9.4 12 3 12 3z"/><path d="M9.5 14.5a2.5 2.5 0 0 0 2.5 2.5"/>'),
  play: svg('<path d="M7 5l12 7-12 7z" fill="currentColor" stroke="none"/>'),
  pause: svg('<path d="M8 5v14M16 5v14" stroke-width="3"/>'),
  step: svg('<path d="M6 5l9 7-9 7z" fill="currentColor" stroke="none"/><path d="M18 5v14" stroke-width="2.4"/>'),
  reset: svg('<path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/>'),
  seed: svg('<circle cx="12" cy="12" r="2.5" fill="currentColor"/><circle cx="12" cy="12" r="7" opacity=".45"/>'),
  rec: svg('<circle cx="12" cy="12" r="6" fill="currentColor" stroke="none"/>'),
  film: svg('<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 5v14M17 5v14M3 9h4M3 15h4M17 9h4M17 15h4"/>'),
  strip: svg('<rect x="2.5" y="7" width="5" height="10" rx="1"/><rect x="9.5" y="7" width="5" height="10" rx="1"/><rect x="16.5" y="7" width="5" height="10" rx="1"/>'),
  body: svg('<circle cx="12" cy="4.5" r="2"/><path d="M8 21v-7l-2-1 1.5-5h9L18 13l-2 1v7"/>'),
};

export class Sparkline {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  constructor(private color: string, private min = 0, private max = 1) {
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d')!;
  }
  draw(values: number[], marker = -1) {
    const c = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, c.clientWidth * dpr), hgt = Math.max(1, c.clientHeight * dpr);
    if (c.width !== w || c.height !== hgt) { c.width = w; c.height = hgt; }
    const g = this.ctx;
    g.clearRect(0, 0, w, hgt);
    if (values.length < 2) return;
    const lo = this.min, hi = Math.max(this.max, ...values);
    const x = (i: number) => (i / (values.length - 1)) * w;
    const y = (v: number) => hgt - 2 - ((v - lo) / (hi - lo || 1)) * (hgt - 4);
    if (marker >= 0) {
      g.strokeStyle = 'rgba(255,255,255,0.18)';
      g.setLineDash([2, 3]);
      g.beginPath(); g.moveTo(x(marker), 0); g.lineTo(x(marker), hgt); g.stroke();
      g.setLineDash([]);
    }
    const grad = g.createLinearGradient(0, 0, 0, hgt);
    grad.addColorStop(0, this.color + '55');
    grad.addColorStop(1, this.color + '00');
    g.beginPath();
    values.forEach((v, i) => (i ? g.lineTo(x(i), y(v)) : g.moveTo(x(i), y(v))));
    g.lineTo(w, hgt); g.lineTo(0, hgt); g.closePath();
    g.fillStyle = grad; g.fill();
    g.beginPath();
    values.forEach((v, i) => (i ? g.lineTo(x(i), y(v)) : g.moveTo(x(i), y(v))));
    g.strokeStyle = this.color; g.lineWidth = 1.5 * dpr; g.stroke();
  }
}

export function toast(msg: string, ms = 2200) {
  const el = h('div', { class: 'toast glass' }, msg);
  document.getElementById('ui')!.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 450); }, ms);
}

export const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
