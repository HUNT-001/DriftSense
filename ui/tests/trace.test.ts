import { describe, expect, it } from 'vitest';
import commitText from '../src/fixtures/sample_trace.jsonl?raw';
import escalateText from '../src/fixtures/sample_trace_escalate.jsonl?raw';
import {
  TRACE_EVENT_KEYS,
  cumulativeDose,
  doseBudget,
  isSyntheticTrace,
  parseTraceJsonl,
  validateTraceEvent,
} from '../src/lib/trace';
import { loadSchema, schemaDir, validate } from './helpers/schema';

const ok = { run_id: 'r', step: 0, ts: '2026-10-11T09:00:00Z', kind: 'observe' };

describe('parseTraceJsonl', () => {
  it('skips blank lines and handles CRLF', () => {
    const text = `\n${JSON.stringify(ok)}\r\n   \r\n${JSON.stringify({ ...ok, step: 1 })}\n\n`;
    const r = parseTraceJsonl(text);
    expect(r.errors).toEqual([]);
    expect(r.events.map((e) => e.step)).toEqual([0, 1]);
  });

  it('reports malformed JSON with 1-based line numbers and keeps going', () => {
    const text = [JSON.stringify(ok), '', '{not json', JSON.stringify({ ...ok, step: 2 })].join('\n');
    const r = parseTraceJsonl(text);
    expect(r.events).toHaveLength(2);
    expect(r.errors).toHaveLength(1);
    expect(r.errors[0]!.line).toBe(3);
    expect(r.errors[0]!.message).toMatch(/invalid JSON/);
  });

  it('reports structurally invalid events with their line number', () => {
    const text = [JSON.stringify({ ...ok, kind: 'teleport' }), JSON.stringify({ ...ok, extra: 1 }), JSON.stringify(ok)].join('\n');
    const r = parseTraceJsonl(text);
    expect(r.events).toHaveLength(1);
    expect(r.errors.map((e) => e.line)).toEqual([1, 2]);
    expect(r.errors[0]!.message).toMatch(/kind/);
    expect(r.errors[1]!.message).toMatch(/unknown field "extra"/);
  });
});

describe('validateTraceEvent', () => {
  it('accepts a minimal event', () => expect(validateTraceEvent(ok)).toEqual([]));
  it.each([
    [{ ...ok, step: -1 }, /step/],
    [{ ...ok, step: 1.5 }, /step/],
    [{ ...ok, ts: 'yesterday' }, /ts/],
    [{ ...ok, v: '2' }, /"v"/],
    [{ ...ok, stage_move_nm: [1] }, /stage_move_nm/],
    [{ ...ok, policy: 'magic' }, /policy/],
    [{ ...ok, margin: 'high' }, /margin/],
    [{ run_id: 'r' }, /missing required/],
    [[1, 2], /not a JSON object/],
  ])('rejects %j', (ev, re) => {
    expect(validateTraceEvent(ev).join(' ')).toMatch(re);
  });
});

describe('dose helpers', () => {
  it('carries cumulative dose forward and finds the budget', () => {
    const evs = parseTraceJsonl(commitText).events;
    const d = cumulativeDose(evs);
    expect(d[0]).toBe(10);
    for (let i = 1; i < d.length; i++) expect(d[i]!).toBeGreaterThanOrEqual(d[i - 1]!);
    expect(doseBudget(evs)).toBe(60);
  });
});

describe('committed sample traces', () => {
  const commit = parseTraceJsonl(commitText);
  const escalate = parseTraceJsonl(escalateText);

  it('parse without errors and are marked synthetic', () => {
    expect(commit.errors).toEqual([]);
    expect(escalate.errors).toEqual([]);
    expect(isSyntheticTrace(commit.events)).toBe(true);
    expect(isSyntheticTrace(escalate.events)).toBe(true);
  });

  it('commit sample tells the canonical story', () => {
    const ev = commit.events;
    const firstLocalize = ev.find((e) => e.kind === 'tool_result' && e.tool === 'driftsense.localize')!;
    expect(firstLocalize.margin).toBe(0.08);
    expect(firstLocalize.tau).toBe(0.22);
    expect(firstLocalize.because).toMatch(/margin below tau/);
    expect(ev.some((e) => e.tool === 'driftsense.localize' && e.args?.['mode'] === 'contour')).toBe(true);
    expect(ev.some((e) => e.tool === 'sem.acquire' && e.args?.['fov_nm'] === 2048)).toBe(true);
    const commitMove = ev.find((e) => e.kind === 'commit_move')!;
    expect(commitMove.margin).toBe(0.31);
    expect(commitMove.margin!).toBeGreaterThanOrEqual(commitMove.tau!);
    expect(commitMove.stage_move_nm).toHaveLength(2);
    expect(ev.at(-1)!.kind).toBe('episode_end');
    const kinds = ev.map((e) => e.kind);
    expect(kinds.indexOf('commit_move')).toBeGreaterThan(kinds.indexOf('decision'));
    expect(Math.max(...ev.map((e) => e.step))).toBeLessThanOrEqual(6);
  });

  it('escalate sample ends with escalate -> human_approval and never commits below tau', () => {
    const ev = escalate.events;
    expect(ev.slice(-2).map((e) => e.kind)).toEqual(['escalate', 'human_approval']);
    expect(ev.some((e) => e.kind === 'commit_move')).toBe(false);
    const esc = ev.at(-2)!;
    expect(esc.margin!).toBeLessThan(esc.tau!);
  });

  it('every event key is a contract key', () => {
    for (const e of [...commit.events, ...escalate.events]) {
      for (const k of Object.keys(e)) expect(TRACE_EVENT_KEYS).toContain(k);
    }
  });

  const schema = loadSchema('trace_event');
  it.skipIf(!schema)(`validate against trace_event.schema.json (${schemaDir()})`, () => {
    for (const [name, text] of [
      ['sample_trace.jsonl', commitText],
      ['sample_trace_escalate.jsonl', escalateText],
    ] as const) {
      text
        .split(/\r?\n/)
        .filter((l) => l.trim())
        .forEach((line, i) => {
          expect(validate(schema!, JSON.parse(line)), `${name}:${i + 1}`).toEqual([]);
        });
    }
    // The subset validator does reject bad events (guards against a vacuous pass).
    expect(validate(schema!, { ...ok, extra: 1 })).not.toEqual([]);
    expect(validate(schema!, { ...ok, kind: 'teleport' })).not.toEqual([]);
    expect(validate(schema!, { ...ok, stage_move_nm: [1, 'a'] })).not.toEqual([]);
    // The hand-written key list matches the schema's properties exactly.
    expect([...TRACE_EVENT_KEYS].sort()).toEqual(Object.keys(schema!['properties'] as object).sort());
  });
});
