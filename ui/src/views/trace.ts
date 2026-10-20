import type { TraceFetch } from '../api/types';
import type { TraceEvent } from '../types/contracts';
import { append, clear, h, setWidthPct } from '../lib/dom';
import { fmt, fmtMs } from '../lib/format';
import {
  KIND_ICON,
  KIND_LABEL,
  cumulativeDose,
  doseBudget,
  fmtDose,
  isSimulatedEvent,
  marginVerdict,
  stepLatency,
  summarizeTrace,
  type TraceKind,
  type TraceSummary,
} from '../lib/trace';
import { marginBar, marginDomain, replayBadge, simulatedBadge, type MarginDomain } from './components';

/** Keys shown elsewhere in the row (dose bar, badges) rather than in the raw key=value line. */
const HIDDEN_SUMMARY_KEYS = new Set(['cumulative_dose', 'dose_this_acq', 'simulated', 'warnings']);
const DECISION_KINDS = new Set<TraceEvent['kind']>(['tool_call', 'decision', 'commit_move', 'escalate', 'episode_end']);

function short(v: unknown): string {
  let s: string;
  if (typeof v === 'number') s = Number.isInteger(v) ? String(v) : fmt(v, 3).replace(/0+$/, '').replace(/\.$/, '');
  else if (typeof v === 'string') s = v;
  else if (Array.isArray(v) && v.every((x) => typeof x === 'number')) s = `[${v.map((x) => short(x)).join(', ')}]`;
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
    entries.map(([k, v], i) => [i ? ' ' : '', h('code', null, `${k}=${short(v)}`)]).flat(),
  );
}

function actionName(ev: TraceEvent): string {
  if (ev.kind === 'tool_call' || ev.kind === 'decision') return ev.tool ?? KIND_LABEL[ev.kind as TraceKind];
  return KIND_LABEL[ev.kind as TraceKind];
}

/** The agent's next action after event i (the first later tool call / decision / terminal event). */
export function nextAction(events: TraceEvent[], i: number): TraceEvent | null {
  for (let j = i + 1; j < events.length; j++) if (DECISION_KINDS.has(events[j]!.kind)) return events[j]!;
  return null;
}

/** One-line plain-language description of an event (used for the live-region announcement). */
export function describeEvent(ev: TraceEvent): string {
  const kind = KIND_LABEL[ev.kind as TraceKind] ?? ev.kind;
  const parts = [`Step ${ev.step}: ${kind}`];
  if (ev.tool) parts.push(ev.tool);
  if (typeof ev.margin === 'number' && typeof ev.tau === 'number') {
    parts.push(`margin ${fmt(ev.margin, 3)} ${ev.margin >= ev.tau ? 'at or above' : 'below'} tau ${fmt(ev.tau, 3)}`);
  }
  if (ev.because) parts.push(`because ${ev.because}`);
  return `${parts.join(', ')}.`;
}

interface RowCtx {
  events: TraceEvent[];
  doses: number[];
  doseMax: number;
  doseIsBudget: boolean;
  domain: MarginDomain;
  summary: TraceSummary;
}

function doseBar(dose: number, ctx: RowCtx): HTMLElement {
  const frac = ctx.doseMax > 0 ? dose / ctx.doseMax : 0;
  const fill = setWidthPct(h('span', { class: 'vbar-fill' }), frac * 100);
  const of = ctx.doseIsBudget ? `of budget ${fmtDose(ctx.doseMax)}` : `of episode total ${fmtDose(ctx.doseMax)}`;
  return h(
    'div',
    { class: `vbar dose${ctx.doseIsBudget && dose > ctx.doseMax ? ' is-over' : ''}` },
    h('div', { class: 'vbar-label' }, h('span', null, 'Cumulative dose'), h('span', { class: 'vbar-value' }, `${fmtDose(dose)} (${of})`)),
    h('div', { class: 'vbar-track', role: 'img', 'aria-label': `Cumulative electron dose ${fmtDose(dose)}, ${of}` }, fill),
  );
}

function eventRow(i: number, ctx: RowCtx): HTMLElement {
  const ev = ctx.events[i]!;
  const kind = ev.kind as TraceKind;
  const lat = stepLatency(ctx.events, i);
  const hasMargin = typeof ev.margin === 'number' && typeof ev.tau === 'number';
  const isLocalizeResult = ev.kind === 'tool_result' && (ev.tool === 'localize' || ev.tool === 'refine') && hasMargin;
  const next = isLocalizeResult ? nextAction(ctx.events, i) : null;
  const isAcquireResult = ev.kind === 'tool_result' && ev.tool === 'acquire';
  const key = ev.kind === 'commit_move' || ev.kind === 'escalate';
  const warnings = Array.isArray(ev.result_summary?.['warnings']) ? (ev.result_summary!['warnings'] as unknown[]) : [];

  const body = h(
    'div',
    { class: 'tl-body' },
    h(
      'p',
      { class: 'tl-head' },
      h('span', { class: `kind kind-${kind}` }, h('span', { 'aria-hidden': 'true' }, `${KIND_ICON[kind] ?? '•'} `), KIND_LABEL[kind] ?? kind),
      ev.tool ? h('code', { class: 'tl-tool' }, ev.tool) : null,
      lat
        ? h(
            'span',
            { class: 'tl-latency muted', title: lat.derived ? 'time since the previous step (from timestamps)' : 'tool latency' },
            h('span', { class: 'sr-only' }, lat.derived ? 'time since previous step ' : 'latency '),
            `${lat.derived ? 'Δt ' : ''}${fmtMs(lat.ms)}`,
          )
        : null,
      ev.policy ? h('span', { class: 'tl-policy muted' }, `policy: ${ev.policy}`) : null,
      isSimulatedEvent(ev) ? simulatedBadge() : null,
    ),
    ev.because
      ? h(
          key ? 'blockquote' : 'p',
          { class: `tl-because${key ? ' is-key' : ''}` },
          h('span', { class: 'tl-because-label' }, 'because '),
          h('q', null, ev.because),
        )
      : null,
    hasMargin ? marginBar(ev.margin!, ev.tau!, ctx.domain, true, ev.kind === 'escalate') : null,
    next && hasMargin
      ? h(
          'p',
          { class: `tl-next is-${marginVerdict(ev.margin!, ev.tau!)}` },
          h('span', { 'aria-hidden': 'true' }, '↪ '),
          'Agent’s next action: ',
          h('strong', null, actionName(next)),
          h('span', { class: 'muted' }, ` (step ${next.step})`),
        )
      : null,
    ev.stage_move_nm
      ? h(
          'p',
          { class: 'tl-move' },
          h('span', { 'aria-hidden': 'true' }, '⇲ '),
          `Stage move committed: dx ${fmt(ev.stage_move_nm[0], 2)} nm, dy ${fmt(ev.stage_move_nm[1], 2)} nm`,
        )
      : null,
    kvLine('args', ev.args as Record<string, unknown>),
    kvLine('result', ev.result_summary as Record<string, unknown>, HIDDEN_SUMMARY_KEYS),
    warnings.length
      ? h('details', { class: 'tl-warn' }, h('summary', null, `${warnings.length} engine warning(s)`), h('ul', null, warnings.map((w) => h('li', null, String(w)))))
      : null,
    isAcquireResult ? doseBar(ctx.doses[i] ?? 0, ctx) : null,
  );
  return h(
    'li',
    { class: `tl-item tl-${kind}${key ? ' tl-key' : ''}`, 'data-step': String(ev.step) },
    h('div', { class: 'tl-step', 'aria-hidden': 'true' }, String(ev.step)),
    h('span', { class: 'sr-only' }, `Step ${ev.step}. `),
    body,
  );
}

/** Headline card: the moment OpenCV's margin decided the physical action. */
export function decisionSummary(s: TraceSummary, events: TraceEvent[]): HTMLElement {
  const d = s.decisive;
  const card = h('div', { class: `decision-card is-${s.outcome}` });
  if (!d || typeof d.margin !== 'number' || typeof d.tau !== 'number') {
    const lastLoc = [...events].reverse().find((e) => typeof e.margin === 'number' && typeof e.tau === 'number');
    card.append(
      h('p', { class: 'decision-title' }, h('span', { 'aria-hidden': 'true' }, '◔ '), s.outcome === 'running' ? 'Agent is working…' : 'Episode ended without a commit'),
      lastLoc
        ? h(
            'p',
            null,
            `Latest OpenCV uniqueness margin ${fmt(lastLoc.margin, 3)} vs τ ${fmt(lastLoc.tau, 3)}: `,
            lastLoc.margin! >= lastLoc.tau! ? 'enough to commit.' : 'not enough to commit yet; the agent keeps looking or asks you.',
          )
        : h('p', { class: 'muted' }, 'Waiting for the first localization margin.'),
    );
    return card;
  }
  const commit = d.kind === 'commit_move';
  append(card, [
    h(
      'p',
      { class: 'decision-title' },
      h('span', { 'aria-hidden': 'true' }, commit ? '✔ ' : '⚠ '),
      commit ? 'The margin cleared τ, so the agent committed the stage move' : 'The margin stayed below τ, so the agent escalated to a human',
    ),
    h(
      'p',
      { class: 'decision-eq' },
      h('span', null, `margin ${fmt(d.margin, 3)}`),
      h('span', { class: 'decision-op' }, commit ? ' ≥ ' : ' < '),
      h('span', null, `τ ${fmt(d.tau, 3)}`),
      h('span', { 'aria-hidden': 'true' }, ' → '),
      h('strong', null, commit ? 'commit_stage_move' : 'escalate'),
    ),
    d.because ? h('p', null, 'Agent’s recorded reason: ', h('q', { class: 'because-key' }, d.because)) : null,
    d.stage_move_nm ? h('p', null, `Corrective move: dx ${fmt(d.stage_move_nm[0], 2)} nm, dy ${fmt(d.stage_move_nm[1], 2)} nm.`) : null,
    h(
      'p',
      { class: 'muted decision-stats' },
      [
        `${events.length} steps`,
        `${s.acquisitions} acquisitions`,
        `dose ${fmtDose(s.dose)}`,
        s.policy ? `policy ${s.policy}` : null,
        s.maxSteps !== null ? `step budget ${s.maxSteps}` : null,
      ]
        .filter(Boolean)
        .join(' · '),
    ),
  ]);
  return card;
}

export interface TraceViewOptions {
  /** Dose budget from episode.json, if the service reports one. */
  doseBudget?: number | null;
}

export interface TraceView {
  el: HTMLElement;
  update(t: TraceFetch, opts?: TraceViewOptions): void;
  /** Number of events rendered. */
  readonly count: number;
}

/** Stateful timeline: new events are appended (not re-rendered) and announced politely. */
export function createTraceView(): TraceView {
  const live = h('div', { class: 'sr-only', 'aria-live': 'polite', 'aria-atomic': 'true', 'data-testid': 'trace-live' });
  const sourceSlot = h('div');
  const summarySlot = h('div');
  const errorSlot = h('div');
  const list = h('ol', { class: 'timeline', 'aria-label': 'Agent steps, oldest first' });
  const empty = h('p', { class: 'muted' }, 'No trace events yet.');
  const el = h('div', { class: 'trace' }, sourceSlot, summarySlot, errorSlot, empty, list, live);
  let rendered: TraceEvent[] = [];
  let domain: MarginDomain | null = null;
  let lastDoseKey = '';

  const inDomain = (events: TraceEvent[], d: MarginDomain) =>
    events.every((e) => typeof e.margin !== 'number' || (e.margin >= d.lo && e.margin <= d.hi));

  return {
    el,
    get count() {
      return rendered.length;
    },
    update(t: TraceFetch, opts: TraceViewOptions = {}) {
      clear(sourceSlot);
      if (t.source === 'replay') {
        sourceSlot.append(
          h(
            'p',
            { class: 'notice notice-replay' },
            replayBadge(),
            ' Offline replay of a recorded trace: these are the real steps the agent logged in a committed episode, paced for viewing. Steps marked “simulated operator step” come from your decision in this tab.',
          ),
        );
      }
      clear(errorSlot);
      if (t.errors.length) {
        errorSlot.append(
          h(
            'div',
            { class: 'notice notice-warn', role: 'status' },
            h('p', { class: 'notice-title' }, `${t.errors.length} trace line(s) could not be read and were skipped:`),
            h('ul', null, t.errors.slice(0, 5).map((e) => h('li', null, `Line ${e.line}: ${e.message}`))),
          ),
        );
      }
      const events = t.events;
      empty.hidden = events.length > 0;
      const summary = summarizeTrace(events);
      clear(summarySlot);
      if (events.length) summarySlot.append(decisionSummary(summary, events));

      const doses = cumulativeDose(events);
      const budget = doseBudget(events, opts.doseBudget);
      const doseMax = budget ?? Math.max(0, ...doses);
      const doseKey = `${doseMax}|${budget !== undefined}`;
      const prefixOk =
        events.length >= rendered.length &&
        rendered.every((e, i) => e.run_id === events[i]!.run_id && e.step === events[i]!.step && e.kind === events[i]!.kind);
      const tau = summary.tau ?? 0.1;
      if (!domain || !prefixOk || !inDomain(events, domain)) {
        // Full render: first time, a different trace, or a scale change.
        domain = marginDomain([...events.map((e) => e.margin), -0.1, 0.6], tau);
        clear(list);
        rendered = [];
      }
      const ctx: RowCtx = { events, doses, doseMax, doseIsBudget: budget !== undefined, domain, summary };
      const start = rendered.length;
      if (doseKey !== lastDoseKey && start > 0) {
        // The dose scale moved (no budget: bars are relative to the episode total so far):
        // refresh only the rows that carry a dose bar, so focus and position stay put.
        for (const bar of list.querySelectorAll('.vbar.dose')) {
          const li = bar.closest<HTMLElement>('li.tl-item');
          if (!li) continue;
          const idx = events.findIndex((e) => String(e.step) === li.dataset['step']);
          if (idx >= 0 && idx < start - 1) li.replaceWith(eventRow(idx, ctx));
        }
      }
      lastDoseKey = doseKey;
      // Re-render the previous last row too: its "next action" may only now be known.
      if (start > 0 && list.lastElementChild) {
        list.lastElementChild.replaceWith(eventRow(start - 1, ctx));
      }
      for (let i = start; i < events.length; i++) {
        const row = eventRow(i, ctx);
        if (start > 0) row.classList.add('tl-new');
        list.append(row);
      }
      if (events.length > start && start > 0) {
        const fresh = events.slice(start);
        const key = fresh.find((e) => e.kind === 'commit_move' || e.kind === 'escalate');
        live.textContent = describeEvent(key ?? fresh.at(-1)!);
      } else if (events.length && start === 0) {
        live.textContent = `Agent trace loaded: ${events.length} steps. ${describeEvent(events.at(-1)!)}`;
      }
      rendered = events.slice();
    },
  };
}

/** Stateless convenience (tests, static rendering). */
export function renderTrace(t: TraceFetch, opts?: TraceViewOptions): HTMLElement {
  const v = createTraceView();
  v.update(t, opts);
  return v.el;
}
