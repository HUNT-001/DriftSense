import type { JobStatus, Margin } from '../types/contracts';
import { h, setLeftPct, setWidthPct } from '../lib/dom';
import { fmt } from '../lib/format';

export const STATUS_TEXT: Record<JobStatus['status'], string> = {
  queued: 'Queued',
  running: 'Running',
  awaiting_approval: 'Awaiting operator approval',
  succeeded: 'Succeeded',
  failed: 'Failed',
};

const STATUS_ICON: Record<JobStatus['status'], string> = {
  queued: '○', // hollow circle
  running: '◔', // circle quarter
  awaiting_approval: '⚠', // warning
  succeeded: '✔', // check
  failed: '✖', // cross
};

export function statusPill(status: JobStatus['status']): HTMLElement {
  return h(
    'span',
    { class: `pill pill-${status}` },
    h('span', { class: 'pill-icon', 'aria-hidden': 'true' }, STATUS_ICON[status]),
    h('span', null, STATUS_TEXT[status]),
  );
}

export function decisionChip(decision: Margin['decision']): HTMLElement {
  const commit = decision === 'commit';
  return h(
    'span',
    { class: `chip chip-${decision}` },
    h('span', { 'aria-hidden': 'true' }, commit ? '✔ ' : '⚠ '),
    commit ? 'Commit' : 'Escalate to operator',
  );
}

export interface MarginDomain {
  lo: number;
  hi: number;
}

/** Domain for margin bars: includes 0, tau and every value (margins can be negative). */
export function marginDomain(values: (number | null | undefined)[], tau: number): MarginDomain {
  const finite = values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
  const lo = Math.min(0, ...finite.map((v) => v * 1.15));
  const hi = Math.max(0.5, tau * 2, ...finite.map((v) => v * 1.1));
  return { lo, hi };
}

const pct = (v: number, d: MarginDomain) => ((v - d.lo) / (d.hi - d.lo || 1)) * 100;

/** What the agent does with this margin. `final` = the step budget is spent (escalate, not re-acquire). */
export function verdictText(margin: number, tau: number, final = false): string {
  if (margin >= tau) return 'commit allowed';
  return final ? 'escalate to a human' : 're-acquire or escalate';
}

/**
 * Horizontal bar of `margin` with a vertical line at `tau` (and at 0 when margins go negative).
 * Exposed to assistive tech as an image with a full text description; the same comparison and
 * the resulting action are printed as visible text (never colour alone: icon + hatching too).
 */
export function marginBar(
  margin: number,
  tau: number,
  domain: MarginDomain | number,
  compact = false,
  final = false,
): HTMLElement {
  const d: MarginDomain = typeof domain === 'number' ? { lo: 0, hi: domain } : domain;
  const above = margin >= tau;
  const verdict = verdictText(margin, tau, final);
  const desc = `Uniqueness margin ${fmt(margin, 3)} ${above ? 'is at or above' : 'is below'} the threshold tau ${fmt(tau, 3)}: ${verdict}`;
  const zero = pct(0, d);
  const fill = h('span', { class: `mbar-fill ${above ? 'is-above' : 'is-below'}` });
  const from = Math.min(zero, pct(margin, d));
  const to = Math.max(zero, pct(margin, d));
  setLeftPct(fill, from);
  setWidthPct(fill, Math.max(0.8, to - from));
  const tick = setLeftPct(h('span', { class: 'mbar-tau' }), pct(tau, d));
  const tauLabel = setLeftPct(h('span', { class: 'mbar-tau-label', 'aria-hidden': 'true' }, 'τ'), pct(tau, d));
  const zeroTick = d.lo < 0 ? setLeftPct(h('span', { class: 'mbar-zero' }), zero) : null;
  return h(
    'div',
    { class: `mbar${compact ? ' mbar-compact' : ''} ${above ? 'is-commit' : 'is-below'}` },
    h('div', { class: 'mbar-track', role: 'img', 'aria-label': desc }, zeroTick, fill, tick, tauLabel),
    h(
      'p',
      { class: 'mbar-text' },
      h('span', { class: 'mbar-icon', 'aria-hidden': 'true' }, above ? '✔ ' : '✗ '),
      h('span', { class: 'mbar-cmp' }, `margin ${fmt(margin, 3)} ${above ? '≥' : '<'} τ ${fmt(tau, 3)}`),
      h('span', { 'aria-hidden': 'true' }, ' → '),
      h('strong', { class: 'mbar-verdict' }, verdict),
    ),
  );
}

/** Generic labelled bar (timings, dose). */
export function valueBar(label: string, valueText: string, fraction: number, cls = ''): HTMLElement {
  const fill = setWidthPct(h('span', { class: 'vbar-fill' }), fraction * 100);
  return h(
    'div',
    { class: `vbar ${cls}` },
    h('div', { class: 'vbar-label' }, h('span', null, label), h('span', { class: 'vbar-value' }, valueText)),
    h('div', { class: 'vbar-track', 'aria-hidden': 'true' }, fill),
  );
}

export function replayBadge(text = 'OFFLINE REPLAY'): HTMLElement {
  return h('span', { class: 'badge badge-replay' }, h('span', { 'aria-hidden': 'true' }, '⏵ '), text);
}

export function simulatedBadge(): HTMLElement {
  return h('span', { class: 'badge badge-sim' }, 'simulated operator step');
}

export function dl(rows: [string, Node | string][]): HTMLDListElement {
  return h(
    'dl',
    { class: 'kv' },
    rows.flatMap(([k, v]) => [h('dt', null, k), h('dd', null, v)]),
  );
}

export function errorBox(title: string, message: string, action?: HTMLElement): HTMLElement {
  return h(
    'div',
    { class: 'notice notice-error', role: 'alert' },
    h('p', { class: 'notice-title' }, h('span', { 'aria-hidden': 'true' }, '✖ '), title),
    h('p', null, message),
    action ?? null,
  );
}
