// "AI Analysis" panel: shows the diagnosis while the rollouts run, then the verdict
// with the measured repair probability, predicted time, shape and vessel recovery.
import { liveSeconds, ROLLOUTS, ROLLOUT_STEPS, type DiagnosisResult, type Progress } from './diagnosis';
import type { RepairOutcome } from './repair';
import { h } from './ui';

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

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

  /** Forecast vs actual, once the live repair has finished. */
  actual(o: RepairOutcome, f: DiagnosisResult) {
    const chip = this.el.querySelector('.ai-chip');
    if (chip) { chip.className = 'ai-chip ok'; chip.textContent = 'repaired'; }
    this.el.querySelector('.fva')?.remove();
    const row = (k: string, a: string, b: string) => `<tr><th>${k}</th><td>${a}</td><td>${b}</td></tr>`;
    this.el.insertAdjacentHTML('beforeend', `
      <table class="fva"><thead><tr><th></th><th>Forecast</th><th>Actual</th></tr></thead><tbody>
        ${row('Shape', pct(f.shape), pct(o.shape))}
        ${row('Vessels', pct(f.vessels), pct(o.vessels))}
        ${row('Time', `${f.predictedSteps} steps`, `${o.healedAt} steps`)}
      </tbody></table>`);
  }

  progress(p: Progress) {
    if (!p.running) return; // the result view takes over
    const dots = Array.from({ length: ROLLOUTS }, (_, i) => {
      const r = p.rollouts[i];
      const cls = r ? (r.success ? 'ok' : 'bad') : i === p.done && p.running ? 'run' : '';
      return `<span class="roll ${cls}" title="${r ? `run ${i + 1}: shape ${pct(r.final.iou)}, vessels ${pct(r.final.connectivity)}` : `run ${i + 1}`}"></span>`;
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

  result(r: DiagnosisResult) {
    const ok = r.rollouts.filter((x) => x.success).length;
    const dots = r.rollouts.map((x, i) =>
      `<span class="roll ${x.success ? 'ok' : 'bad'}" title="run ${i + 1}: shape ${pct(x.final.iou)}, vessels ${pct(x.final.connectivity)}"></span>`).join('');
    const secs = liveSeconds(r.predictedSteps);
    const verdict = r.repairable
      ? `<div class="verdict ok"><b>REPAIRABLE</b><span>${Math.round(r.confidence * 100)}% confidence</span></div>`
      : `<div class="verdict bad"><b>NOT REPAIRABLE</b><span>${Math.round(r.confidence * 100)}% confidence</span></div>
         <ul class="reasons">${r.reasons.map((x) => `<li>${x}</li>`).join('')}</ul>`;
    const legend = r.repairable
      ? '<div class="legend-row"><span class="sw orange"></span>Orange glow: where the simulated repairs disagree</div>'
      : '<div class="legend-row"><span class="sw red"></span>Red: the area that does not come back</div><div class="legend-row"><span class="sw orange"></span>Orange: where the simulated repairs disagree</div>';
    this.el.innerHTML = `
      <div class="ai-head"><span class="ai-title">AI Analysis</span><span class="ai-chip ${r.repairable ? 'ok' : 'bad'}">done</span></div>
      ${verdict}
      <dl class="ai-stats">
        <dt>Repair probability</dt><dd>${Math.round(r.probability * 100)}%<small>${ok} of ${ROLLOUTS} simulations succeed</small></dd>
        ${r.repairable
          ? `<dt>Predicted time</dt><dd>${r.predictedSteps} steps<small>≈ ${secs.toFixed(1)} s at live speed</small></dd>`
          : `<dt>Repair stalls</dt><dd>after ${r.predictedSteps} steps<small>no further progress</small></dd>`}
        <dt>Shape recovered</dt><dd>${pct(r.shape)}<small>median of ${ROLLOUTS} runs</small></dd>
        <dt>Vessels reconnected</dt><dd>${pct(r.vessels)}<small>vascular tree linked to the hilum</small></dd>
        <dt>Tissue removed</dt><dd>${pct(r.lesionShare)}</dd>
      </dl>
      <div class="rolls">${dots}</div>
      ${legend}
      <p class="ai-note">A run succeeds when it recovers more than 90% of the shape with vessels reconnected.</p>`;
    this.el.classList.add('show');
  }
}
