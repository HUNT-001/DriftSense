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

/** Domain for margin bars so tau sits comfortably inside the scale. */
export function marginDomain(values: (number | null | undefined)[], tau: number): number {
  const finite = values.filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
  return Math.max(0.5, tau * 2, ...finite.map((v) => v * 1.1));
}

/**
 * Horizontal bar of `margin` with a vertical line at `tau`. Exposed to assistive tech as an image
 * with a full text description; the same comparison is also printed as visible text.
 */
export function marginBar(margin: number, tau: number, max: number, compact = false): HTMLElement {
  const above = margin >= tau;
  const desc = `Margin ${fmt(margin)} ${above ? 'is at or above' : 'is below'} the threshold tau ${fmt(tau)}`;
  const fill = setWidthPct(h('span', { class: `mbar-fill ${above ? 'is-above' : 'is-below'}` }), (margin / max) * 100);
  const tick = setLeftPct(h('span', { class: 'mbar-tau' }), (tau / max) * 100);
  return h(
    'div',
    { class: `mbar${compact ? ' mbar-compact' : ''}` },
    h('div', { class: 'mbar-track', role: 'img', 'aria-label': desc }, fill, tick),
    h(
      'p',
      { class: 'mbar-text' },
      h('span', { 'aria-hidden': 'true' }, above ? '✔ ' : '✗ '),
      `margin ${fmt(margin)} ${above ? '≥' : '<'} τ ${fmt(tau)}`,
      h('span', { class: 'muted' }, above ? ' (unique)' : ' (not proven unique)'),
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

export function syntheticBadge(text = 'SYNTHETIC SAMPLE'): HTMLElement {
  return h('span', { class: 'badge badge-synthetic' }, h('span', { 'aria-hidden': 'true' }, '⚑ '), text);
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
