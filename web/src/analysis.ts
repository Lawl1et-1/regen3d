// "AI Analysis" panel: shows the diagnosis while the rollouts run, then the verdict with the
// simulated healing potential and, separately, tissue and vascular recovery, residual damage,
// the model's uncertainty and the injury itself (see healing.ts for what each number means).
import { liveSeconds, ROLLOUTS, ROLLOUT_STEPS, type DiagnosisResult, type Progress } from './diagnosis';
import type { OutcomeScore } from './healing';
import type { RepairOutcome } from './repair';
import { h } from './ui';

const whole = (v: number) => `${Math.round(v * 100)}%`;
/** small shares: under 1% is shown as such rather than with false precision */
const small = (v: number) => (v > 0 && v < 0.01 ? '<1%' : whole(v));
const r5 = (v: number) => Math.round(v * 20) * 5;

export class AnalysisPanel {
  readonly el: HTMLElement;

  constructor(parent: HTMLElement) {
    this.el = h('aside', { class: 'ai-panel glass', 'aria-live': 'polite' });
    parent.appendChild(this.el);
  }

  hide() { this.wasShown = this.el.classList.contains('show') || this.wasShown; this.el.classList.remove('show'); }
  private wasShown = false;
  /** Show the panel again after a comparison view, if it had content. */
  restore() { if (this.wasShown && this.el.innerHTML) this.el.classList.add('show'); this.wasShown = false; }

  /** The hologram is showing the predicted repair; the live repair follows. */
  plan() {
    const chip = this.el.querySelector('.ai-chip');
    if (chip) { chip.className = 'ai-chip run'; chip.textContent = 'repairing'; }
    this.el.querySelector('.plan-note')?.remove();
    this.el.querySelector('.verdict')?.insertAdjacentHTML('afterend',
      '<p class="plan-note"><span class="sw blue"></span>Blue hologram: the shape the 8 simulations predict. Dotted lines: vessels that will reconnect.</p>');
  }

  /** Forecast vs actual, once the live repair has finished (both scored the same way). */
  actual(o: RepairOutcome, f: DiagnosisResult, a: OutcomeScore) {
    const chip = this.el.querySelector('.ai-chip');
    if (chip) { chip.className = 'ai-chip ok'; chip.textContent = 'repaired'; }
    this.el.querySelector('.fva')?.remove();
    const row = (k: string, x: string, y: string) => `<tr><th>${k}</th><td>${x}</td><td>${y}</td></tr>`;
    const e = f.healing;
    this.el.insertAdjacentHTML('beforeend', `
      <table class="fva"><thead><tr><th></th><th>Forecast</th><th>Actual</th></tr></thead><tbody>
        ${row('Healing potential', e.label, `≈ ${r5(a.potential)}%`)}
        ${row('Tissue rebuilt', whole(e.tissue), whole(a.tissue))}
        ${e.vessels !== null || a.vessels !== null ? row('Vascular recovery', e.vessels === null ? '—' : whole(e.vessels), a.vessels === null ? '—' : whole(a.vessels)) : ''}
        ${row('Time', `${f.predictedSteps} steps`, `${o.healedAt} steps`)}
      </tbody></table>`);
  }

  progress(p: Progress) {
    if (!p.running) return; // the result view takes over
    const dots = Array.from({ length: ROLLOUTS }, (_, i) => {
      const r = p.rollouts[i];
      const cls = r ? (r.success ? 'ok' : 'bad') : i === p.done && p.running ? 'run' : '';
      return `<span class="roll ${cls}" title="${r ? `simulated repair ${i + 1}: healing potential ${whole(r.score.potential)}` : `simulated repair ${i + 1}`}"></span>`;
    }).join('');
    this.el.innerHTML = `
      <div class="ai-head"><span class="ai-title">AI Analysis</span><span class="ai-chip run">analysing</span></div>
      <p class="ai-lead">Simulating ${ROLLOUTS} possible repairs of this exact lesion</p>
      <div class="rolls">${dots}</div>
      <div class="ai-bar"><span style="width:${(p.done / ROLLOUTS) * 100}%"></span></div>
      <p class="ai-note">The trained network repairs a copy of the organ ${ROLLOUTS} times, ${ROLLOUT_STEPS} steps each.
        It is stochastic, so each run ends a little differently.</p>`;
    this.el.classList.add('show');
  }

  /** `mm`: millimetres per voxel of this organ, for the injury depth. */
  result(r: DiagnosisResult, mm: number) {
    const e = r.healing, c = r.healingCtx;
    const dots = r.rollouts.map((x, i) =>
      `<span class="roll ${x.success ? 'ok' : 'bad'}" title="simulated repair ${i + 1}: healing potential ${whole(x.score.potential)}"></span>`).join('');
    const secs = liveSeconds(r.predictedSteps);
    const verdict = r.repairable
      ? `<div class="verdict ok"><b>REPAIRABLE</b><span>simulated healing potential ${e.label}</span></div>`
      : `<div class="verdict bad"><b>NOT REPAIRABLE</b><span>simulated healing potential ${e.label}</span></div>`;
    const reasons = r.reasons.length ? `<ul class="reasons">${r.reasons.map((x) => `<li>${x}</li>`).join('')}</ul>` : '';
    const legend = r.repairable
      ? '<div class="legend-row"><span class="sw orange"></span>Orange glow: where the simulated repairs disagree</div>'
      : '<div class="legend-row"><span class="sw red"></span>Red: the area that does not come back</div><div class="legend-row"><span class="sw orange"></span>Orange: where the simulated repairs disagree</div>';
    const unc = { low: 'Low', moderate: 'Moderate', high: 'High' }[e.uncertainty];
    this.el.innerHTML = `
      <div class="ai-head"><span class="ai-title">AI Analysis</span><span class="ai-chip ${r.repairable ? 'ok' : 'bad'}">done</span></div>
      ${verdict}
      ${reasons}
      <dl class="ai-stats">
        <dt>Healing potential</dt><dd>${e.label}<small>${whole(e.min)}–${whole(e.max)} over ${ROLLOUTS} simulated repairs</small></dd>
        <dt>Tissue rebuilt</dt><dd>${whole(e.tissue)}<small>of the removed volume, right cell types</small></dd>
        <dt>Vascular recovery</dt>${e.vessels === null
          ? '<dd>—<small>no blood flow cut beyond the wound</small></dd>'
          : `<dd>${whole(e.vessels)}<small>of the vessels cut off from blood (${small(c.flowLost)} of the supply)</small></dd>`}
        <dt>Residual damage</dt><dd>${small(e.residual)} of organ<small>uninjured tissue starving: ${small(e.impairment)}</small></dd>
        <dt>Model uncertainty</dt><dd class="unc ${e.uncertainty}">${unc}<small>${e.uncertaintyWhy.join('; ')}</small></dd>
        <dt>Injury</dt><dd>${small(c.share)} of tissue<small>${Math.max(1, Math.round(c.depthVoxels * mm))} mm deep · ${small(c.flowLost)} of blood flow cut</small></dd>
        ${r.repairable
          ? `<dt>Predicted time</dt><dd>${r.predictedSteps} steps<small>≈ ${secs.toFixed(1)} s at live speed</small></dd>`
          : `<dt>Repair stalls</dt><dd>after ${r.predictedSteps} steps<small>no further progress</small></dd>`}
      </dl>
      <div class="rolls">${dots}</div>
      ${legend}
      <p class="ai-note">Healing potential = tissue rebuilt × blood supply restored × uninjured tissue still fed,
        averaged over ${ROLLOUTS} simulated repairs of this injury. A heuristic score of the simulation, rounded to what
        it supports: not a medical or calibrated probability.</p>`;
    this.el.classList.add('show');
  }
}
