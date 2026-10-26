// WebM recording of the canvas and "time-lapse strip" export.
export function download(blob: Blob, name: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

export class Recorder {
  private rec: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  onChange?: (on: boolean) => void;
  constructor(private canvas: HTMLCanvasElement) {}
  get active() { return !!this.rec; }

  start() {
    if (this.rec) return;
    const stream = this.canvas.captureStream(60);
    const types = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
    const mimeType = types.find((t) => MediaRecorder.isTypeSupported(t)) ?? '';
    this.rec = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 16_000_000 });
    this.chunks = [];
    this.rec.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
    this.rec.onstop = () => {
      download(new Blob(this.chunks, { type: 'video/webm' }), `regen3d-${Date.now()}.webm`);
      this.rec = null;
      this.onChange?.(false);
    };
    this.rec.start(250);
    this.onChange?.(true);
  }

  stop() { this.rec?.stop(); }
  toggle() { if (this.rec) this.stop(); else this.start(); }
}

/** Collects frames at chosen simulation steps and lays them out side by side. */
export class StripCapture {
  frames: { img: HTMLCanvasElement; label: string }[] = [];
  targets: number[] = [];
  active = false;
  private start = 0;
  onDone?: () => void;

  begin(currentStep: number, offsets = [0, 15, 40, 80, 150, 300]) {
    this.frames = [];
    this.start = currentStep;
    this.targets = offsets.map((o) => currentStep + o);
    this.active = true;
  }

  /** Call right after a frame was submitted (the canvas still holds it). */
  maybeGrab(canvas: HTMLCanvasElement, step: number) {
    if (!this.active || !this.targets.length || step < this.targets[0]) return;
    this.targets.shift();
    const s = Math.min(1, 360 / canvas.height);
    const cw = Math.round(canvas.height * 0.8 * s), ch = Math.round(canvas.height * s);
    const c = document.createElement('canvas');
    c.width = cw; c.height = ch;
    const sx = (canvas.width - canvas.height * 0.8) / 2;
    c.getContext('2d')!.drawImage(canvas, sx, 0, canvas.height * 0.8, canvas.height, 0, 0, cw, ch);
    this.frames.push({ img: c, label: `+${step - this.start} steps` });
    if (!this.targets.length) this.finish();
  }

  private finish() {
    this.active = false;
    const pad = 10, top = 34, bottom = 30;
    const w = this.frames.reduce((a, f) => a + f.img.width + pad, pad);
    const hh = this.frames[0].img.height + top + bottom;
    const out = document.createElement('canvas');
    out.width = w; out.height = hh;
    const g = out.getContext('2d')!;
    g.fillStyle = '#06070a'; g.fillRect(0, 0, w, hh);
    g.fillStyle = '#e7ecf3'; g.font = '600 14px Inter, sans-serif';
    g.fillText('REGEN 3D — regrowth time-lapse', pad, 22);
    g.font = '12px JetBrains Mono, monospace';
    let x = pad;
    for (const f of this.frames) {
      g.drawImage(f.img, x, top);
      g.fillStyle = '#8b95a6';
      g.fillText(f.label, x + 4, top + f.img.height + 18);
      x += f.img.width + pad;
    }
    g.fillStyle = '#566070'; g.font = '10px Inter, sans-serif';
    g.fillText('Simulation of a learned model — not a medical tool.', w - 290, hh - 8);
    out.toBlob((b) => b && download(b, `regen3d-strip-${Date.now()}.png`));
    this.onDone?.();
  }
}
