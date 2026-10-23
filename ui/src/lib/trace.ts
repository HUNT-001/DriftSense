import type { TraceEvent } from '../types/contracts';

export const TRACE_KINDS = [
  'observe',
  'tool_call',
  'tool_result',
  'decision',
  'commit_move',
  'escalate',
  'human_approval',
  'error',
  'episode_end',
] as const;

export type TraceKind = (typeof TRACE_KINDS)[number];

export const KIND_LABEL: Record<TraceKind, string> = {
  observe: 'Observe',
  tool_call: 'Tool call',
  tool_result: 'Tool result',
  decision: 'Decision',
  commit_move: 'Commit stage move',
  escalate: 'Escalate to operator',
  human_approval: 'Operator decision',
  error: 'Error',
  episode_end: 'Episode end',
};

/** Glyph shown next to the kind label so the signal never relies on colour alone. */
export const KIND_ICON: Record<TraceKind, string> = {
  observe: '◉', // fisheye
  tool_call: '→', // arrow
  tool_result: '↳', // return arrow
  decision: '◆', // diamond
  commit_move: '✔', // check
  escalate: '⚠', // warning
  human_approval: '☺', // person
  error: '✖', // cross
  episode_end: '■', // square
};

/** Keys allowed on a TraceEvent (contract uses extra="forbid"). */
export const TRACE_EVENT_KEYS = [
  'v',
  'run_id',
  'step',
  'ts',
  'kind',
  'tool',
  'args',
  'result_summary',
  'margin',
  'tau',
  'because',
  'stage_move_nm',
  'latency_ms',
  'policy',
] as const;

const REQUIRED = ['run_id', 'step', 'ts', 'kind'] as const;
const ISO_DATETIME = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$/;

function isObj(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const optNum = (v: unknown) => v === undefined || v === null || isNum(v);
const optStr = (v: unknown) => v === undefined || v === null || typeof v === 'string';

/**
 * Hand-written structural check mirroring contracts/trace_event.schema.json.
 * Returns a list of human-readable problems (empty = valid).
 */
export function validateTraceEvent(o: unknown): string[] {
  if (!isObj(o)) return ['event is not a JSON object'];
  const errs: string[] = [];
  for (const k of REQUIRED) if (!(k in o)) errs.push(`missing required field "${k}"`);
  for (const k of Object.keys(o)) {
    if (!(TRACE_EVENT_KEYS as readonly string[]).includes(k)) errs.push(`unknown field "${k}"`);
  }
  if ('v' in o && o.v !== '1') errs.push('"v" must be "1"');
  if ('run_id' in o && typeof o.run_id !== 'string') errs.push('"run_id" must be a string');
  if ('step' in o && !(Number.isInteger(o.step) && (o.step as number) >= 0)) {
    errs.push('"step" must be an integer >= 0');
  }
  if ('ts' in o && !(typeof o.ts === 'string' && ISO_DATETIME.test(o.ts) && !Number.isNaN(Date.parse(o.ts)))) {
    errs.push('"ts" must be an ISO 8601 date-time string');
  }
  if ('kind' in o && !(TRACE_KINDS as readonly unknown[]).includes(o.kind)) {
    errs.push(`"kind" must be one of ${TRACE_KINDS.join(', ')}`);
  }
  if (!optStr(o.tool)) errs.push('"tool" must be a string or null');
  if (!optStr(o.because)) errs.push('"because" must be a string or null');
  if ('args' in o && !isObj(o.args)) errs.push('"args" must be an object');
  if ('result_summary' in o && !isObj(o.result_summary)) errs.push('"result_summary" must be an object');
  for (const k of ['margin', 'tau', 'latency_ms'] as const) {
    if (!optNum(o[k])) errs.push(`"${k}" must be a number or null`);
  }
  const sm = o.stage_move_nm;
  if (sm !== undefined && sm !== null && !(Array.isArray(sm) && sm.length === 2 && sm.every(isNum))) {
    errs.push('"stage_move_nm" must be [dx, dy] numbers or null');
  }
  if (o.policy !== undefined && o.policy !== null && o.policy !== 'rules' && o.policy !== 'llm') {
    errs.push('"policy" must be "rules", "llm" or null');
  }
  return errs;
}

export interface TraceParseError {
  /** 1-based line number in the JSONL text. */
  line: number;
  message: string;
}

export interface ParsedTrace {
  events: TraceEvent[];
  errors: TraceParseError[];
}

/** Parse JSONL: one TraceEvent per line; blank lines are skipped; bad lines are reported, not fatal. */
export function parseTraceJsonl(text: string): ParsedTrace {
  const events: TraceEvent[] = [];
  const errors: TraceParseError[] = [];
  const lines = text.split(/\r?\n/);
  lines.forEach((raw, i) => {
    const line = raw.trim();
    if (line === '') return;
    let obj: unknown;
    try {
      obj = JSON.parse(line);
    } catch (e) {
      errors.push({ line: i + 1, message: `invalid JSON: ${(e as Error).message}` });
      return;
    }
    const problems = validateTraceEvent(obj);
    if (problems.length) {
      errors.push({ line: i + 1, message: problems.join('; ') });
      return;
    }
    events.push(obj as TraceEvent);
  });
  return { events, errors };
}

function summaryNum(ev: TraceEvent, key: string): number | undefined {
  const v = ev.result_summary?.[key];
  return isNum(v) ? v : undefined;
}

/**
 * Cumulative electron dose after each event. The `acquire` tool_result reports
 * `result_summary.cumulative_dose` (the virtual SEM's running counter); `episode_end` reports the
 * total as `dose`. Events without either carry the last value forward (0 before the first acquire).
 */
export function cumulativeDose(events: TraceEvent[]): number[] {
  let last = 0;
  return events.map((ev) => {
    const d = summaryNum(ev, 'cumulative_dose') ?? (ev.kind === 'episode_end' ? summaryNum(ev, 'dose') : undefined);
    if (d !== undefined) last = Math.max(last, d);
    return last;
  });
}

/** Dose budget if the episode reports one (`max_dose` / `dose_budget` in any result_summary). */
export function doseBudget(events: TraceEvent[], fallback?: number | null): number | undefined {
  for (let i = events.length - 1; i >= 0; i--) {
    const b = summaryNum(events[i]!, 'max_dose') ?? summaryNum(events[i]!, 'dose_budget');
    if (b !== undefined && b > 0) return b;
  }
  return typeof fallback === 'number' && fallback > 0 ? fallback : undefined;
}

/** Engineering-notation dose, e.g. 4.706e7 -> "47.06 M". The simulator's dose is unit-less. */
export function fmtDose(d: number): string {
  if (!Number.isFinite(d)) return '—';
  if (Math.abs(d) >= 1e9) return `${(d / 1e9).toFixed(2)} G`;
  if (Math.abs(d) >= 1e6) return `${(d / 1e6).toFixed(2)} M`;
  if (Math.abs(d) >= 1e3) return `${(d / 1e3).toFixed(2)} k`;
  return d.toFixed(0);
}

/**
 * Step latency in ms: `latency_ms` when the event reports it (tool timings), otherwise the
 * wall-clock gap since the previous event from `ts` (marked `derived`).
 */
export function stepLatency(events: TraceEvent[], i: number): { ms: number; derived: boolean } | null {
  const ev = events[i];
  if (!ev) return null;
  if (isNum(ev.latency_ms)) return { ms: ev.latency_ms, derived: false };
  const prev = events[i - 1];
  if (!prev) return null;
  const dt = Date.parse(ev.ts) - Date.parse(prev.ts);
  return Number.isFinite(dt) && dt >= 0 ? { ms: dt, derived: true } : null;
}

export type MarginVerdict = 'commit' | 'reacquire';

/** The agent's rule: margin >= tau -> commit is allowed; below -> re-acquire, or escalate when out of budget. */
export function marginVerdict(margin: number, tau: number): MarginVerdict {
  return margin >= tau ? 'commit' : 'reacquire';
}

export interface TraceSummary {
  runId: string | null;
  outcome: 'commit' | 'escalate' | 'abort' | 'running';
  /** The commit_move or escalate event that ended the decision loop. */
  decisive: TraceEvent | null;
  tau: number | null;
  /** Margin the decisive action was based on. */
  margin: number | null;
  dose: number;
  acquisitions: number;
  policy: string | null;
  maxSteps: number | null;
}

export function summarizeTrace(events: TraceEvent[]): TraceSummary {
  const decisive = [...events].reverse().find((e) => e.kind === 'commit_move' || e.kind === 'escalate') ?? null;
  const end = [...events].reverse().find((e) => e.kind === 'episode_end') ?? null;
  const start = events.find((e) => e.result_summary?.['episode_start'] === true);
  const rawOutcome = end?.result_summary?.['outcome'];
  let outcome: TraceSummary['outcome'] = 'running';
  if (rawOutcome === 'commit' || rawOutcome === 'escalate' || rawOutcome === 'abort') outcome = rawOutcome;
  else if (end) outcome = decisive?.kind === 'commit_move' ? 'commit' : decisive ? 'escalate' : 'abort';
  const doses = cumulativeDose(events);
  const startTau = start ? summaryNum(start, 'tau') : undefined;
  const lastMargined = [...events].reverse().find((e) => isNum(e.margin));
  const policy = decisive?.policy ?? (typeof start?.result_summary?.['policy'] === 'string' ? (start.result_summary['policy'] as string) : null);
  return {
    runId: events[0]?.run_id ?? null,
    outcome,
    decisive,
    tau: isNum(decisive?.tau) ? decisive!.tau! : (startTau ?? (isNum(lastMargined?.tau) ? lastMargined!.tau! : null)),
    margin: isNum(decisive?.margin) ? decisive!.margin! : isNum(lastMargined?.margin) ? lastMargined!.margin! : null,
    dose: (end ? summaryNum(end, 'dose') : undefined) ?? doses.at(-1) ?? 0,
    acquisitions: (end ? summaryNum(end, 'n_acq') : undefined) ?? events.filter((e) => e.kind === 'tool_result' && e.tool === 'acquire').length,
    policy,
    maxSteps: start ? (summaryNum(start, 'max_steps') ?? null) : null,
  };
}

/** nm per pixel of the latest search acquisition (fov_nm / width_px from an acquire tool_result). */
export function pixelSizeNm(events: TraceEvent[]): number | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i]!;
    if (ev.kind !== 'tool_result' || ev.tool !== 'acquire') continue;
    const fov = summaryNum(ev, 'fov_nm');
    const w = summaryNum(ev, 'width_px');
    if (fov && w) return fov / w;
  }
  return null;
}

/** Events the offline replay synthesised (operator decision); they carry result_summary.simulated. */
export function isSimulatedEvent(ev: TraceEvent): boolean {
  return ev.result_summary?.['simulated'] === true;
}
