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
 * Cumulative electron dose after each event. `result_summary.dose` is the virtual SEM's cumulative
 * dose counter; events without it carry the last known value forward (0 before the first reading).
 */
export function cumulativeDose(events: TraceEvent[]): number[] {
  let last = 0;
  return events.map((ev) => {
    const d = summaryNum(ev, 'dose');
    if (d !== undefined) last = Math.max(last, d);
    return last;
  });
}

/** Dose budget if any event reports one (`result_summary.dose_budget`). */
export function doseBudget(events: TraceEvent[]): number | undefined {
  for (let i = events.length - 1; i >= 0; i--) {
    const b = summaryNum(events[i]!, 'dose_budget');
    if (b !== undefined && b > 0) return b;
  }
  return undefined;
}

export function doseUnit(events: TraceEvent[]): string {
  for (const ev of events) {
    const u = ev.result_summary?.['dose_unit'];
    if (typeof u === 'string' && u) return u;
  }
  return 'a.u.';
}

/** True when any event marks itself as synthetic (`result_summary.synthetic === true`). */
export function isSyntheticTrace(events: TraceEvent[]): boolean {
  return events.some((e) => e.result_summary?.['synthetic'] === true);
}
