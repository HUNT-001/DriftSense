import type { TraceFetch } from '../api/types';
import type { TraceEvent } from '../types/contracts';
import { h } from '../lib/dom';
import { fmt, fmtMs } from '../lib/format';
import { KIND_ICON, KIND_LABEL, cumulativeDose, doseBudget, doseUnit, type TraceKind } from '../lib/trace';
import { marginBar, marginDomain, syntheticBadge, valueBar } from './components';

const HIDDEN_SUMMARY_KEYS = new Set(['synthetic', 'dose', 'dose_budget', 'dose_unit']);

function short(v: unknown): string {
  let s: string;
  if (typeof v === 'number') s = Number.isInteger(v) ? String(v) : fmt(v, 3).replace(/0+$/, '').replace(/\.$/, '');
  else if (typeof v === 'string') s = v;
  else s = JSON.stringify(v) ?? String(v);
  return s.length > 90 ? `${s.slice(0, 87)}...` : s;
}

function kvLine(label: string, obj: Record<string, unknown> | undefined, skip?: Set<string>): HTMLElement | null {
  const entries = Object.entries(obj ?? {}).filter(([k]) => !skip?.has(k));
  if (!entries.length) return null;
  return h(
    'p',
    { class: 'tl-kv' },
    h('span', { class: 'muted' }, `${label}: `),
    entries.map(([k, v], i) => [i ? ', ' : '', h('code', null, `${k}=${short(v)}`)]).flat(),
  );
}

function eventItem(ev: TraceEvent, dose: number, budget: number, unit: string, mdomain: number): HTMLElement {
  const kind = ev.kind as TraceKind;
  const body = h(
    'div',
    { class: 'tl-body' },
    h(
      'p',
      { class: 'tl-head' },
      h('span', { class: `kind kind-${kind}` }, h('span', { 'aria-hidden': 'true' }, `${KIND_ICON[kind]} `), KIND_LABEL[kind]),
      ev.tool ? h('code', { class: 'tl-tool' }, ev.tool) : null,
      ev.latency_ms !== null && ev.latency_ms !== undefined
        ? h('span', { class: 'tl-latency muted' }, h('span', { class: 'sr-only' }, 'latency '), fmtMs(ev.latency_ms))
        : null,
      ev.policy ? h('span', { class: 'tl-policy muted' }, `policy: ${ev.policy}`) : null,
    ),
    ev.because ? h('p', { class: 'tl-because' }, h('span', { class: 'muted' }, 'because '), ev.because) : null,
    kvLine('args', ev.args as Record<string, unknown>),
    kvLine('result', ev.result_summary as Record<string, unknown>, HIDDEN_SUMMARY_KEYS),
    typeof ev.margin === 'number' && typeof ev.tau === 'number' ? marginBar(ev.margin, ev.tau, mdomain, true) : null,
    ev.stage_move_nm
      ? h(
          'p',
          { class: 'tl-move' },
          h('span', { 'aria-hidden': 'true' }, '➕ '),
          `Stage move: dx ${fmt(ev.stage_move_nm[0], 1)} nm, dy ${fmt(ev.stage_move_nm[1], 1)} nm`,
        )
      : null,
    budget > 0
      ? valueBar('Cumulative dose', `${fmt(dose, 1)} / ${fmt(budget, 1)} ${unit}`, dose / budget, dose > budget ? 'is-over' : '')
      : null,
  );
  return h(
    'li',
    { class: `tl-item tl-${kind}` },
    h('div', { class: 'tl-step', 'aria-hidden': 'true' }, String(ev.step)),
    h('span', { class: 'sr-only' }, `Step ${ev.step}. `),
    body,
  );
}

/** Agent decision timeline built from TraceEvent JSONL. */
export function renderTrace(t: TraceFetch): HTMLElement {
  const wrap = h('div', { class: 'trace' });
  if (t.synthetic) {
    wrap.append(
      h(
        'p',
        { class: 'notice notice-synthetic' },
        syntheticBadge(),
        ' This trace is a committed synthetic sample that illustrates the agent loop. It is not output from a real run.',
      ),
    );
  }
  if (t.errors.length) {
    wrap.append(
      h(
        'div',
        { class: 'notice notice-warn', role: 'status' },
        h('p', { class: 'notice-title' }, `${t.errors.length} trace line(s) could not be read and were skipped:`),
        h('ul', null, t.errors.slice(0, 5).map((e) => h('li', null, `Line ${e.line}: ${e.message}`))),
      ),
    );
  }
  if (!t.events.length) {
    wrap.append(h('p', { class: 'muted' }, 'No trace events yet.'));
    return wrap;
  }
  const doses = cumulativeDose(t.events);
  const maxDose = Math.max(0, ...doses);
  const budget = doseBudget(t.events) ?? maxDose;
  const unit = doseUnit(t.events);
  const taus = t.events.map((e) => e.tau).filter((x): x is number => typeof x === 'number');
  const mdomain = marginDomain(
    t.events.map((e) => e.margin),
    taus.length ? Math.max(...taus) : 0.25,
  );
  const list = h('ol', { class: 'timeline', 'aria-label': 'Agent decision trace' });
  t.events.forEach((ev, i) => list.append(eventItem(ev, doses[i] ?? 0, budget, unit, mdomain)));
  wrap.append(list);
  return wrap;
}
