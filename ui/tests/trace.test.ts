import { describe, expect, it } from 'vitest';
import {
  TRACE_EVENT_KEYS,
  cumulativeDose,
  doseBudget,
  marginVerdict,
  parseTraceJsonl,
  pixelSizeNm,
  stepLatency,
  summarizeTrace,
  validateTraceEvent,
} from '../src/lib/trace';
import { loadSchema, schemaDir, validate } from './helpers/schema';
import { readSample, repoTrace } from './helpers/samples';

const ok = { run_id: 'r', step: 0, ts: '2026-10-11T09:00:00Z', kind: 'observe' };
const commitText = readSample('demo_commit.jsonl');
const escalateText = readSample('demo_escalate.jsonl');

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

describe('real committed sample traces (public/samples = traces/)', () => {
  const commit = parseTraceJsonl(commitText);
  const escalate = parseTraceJsonl(escalateText);

  it('are byte-identical (modulo line endings) to the repo traces when those exist', () => {
    for (const [name, text] of [
      ['demo_commit.jsonl', commitText],
      ['demo_escalate.jsonl', escalateText],
    ] as const) {
      const repo = repoTrace(name);
      if (repo !== null) expect(text.replace(/\r\n/g, '\n'), name).toBe(repo.replace(/\r\n/g, '\n'));
    }
  });

  it('parse with zero errors', () => {
    expect(commit.errors).toEqual([]);
    expect(escalate.errors).toEqual([]);
    expect(commit.events.length).toBeGreaterThan(5);
    expect(escalate.events.length).toBeGreaterThan(5);
  });

  it('commit trace: the OpenCV margin clears tau and drives the commit_move', () => {
    const ev = commit.events;
    const cm = ev.find((e) => e.kind === 'commit_move')!;
    expect(cm).toBeTruthy();
    expect(cm.margin!).toBeGreaterThanOrEqual(cm.tau!);
    expect(cm.because).toMatch(/^margin [\d.]+ >= tau [\d.]+$/);
    expect(cm.stage_move_nm).toHaveLength(2);
    // The localize result that preceded the commit carries the same margin.
    const loc = ev.filter((e) => e.kind === 'tool_result' && e.tool === 'localize' && e.step < cm.step).at(-1)!;
    expect(loc.margin).toBe(cm.margin);
    expect(marginVerdict(loc.margin!, loc.tau!)).toBe('commit');
    expect(ev.at(-1)!.kind).toBe('episode_end');
    const s = summarizeTrace(ev);
    expect(s.outcome).toBe('commit');
    expect(s.decisive?.kind).toBe('commit_move');
    expect(s.acquisitions).toBe(3);
  });

  it('escalate trace: margin below tau, never commits, escalates to a human', () => {
    const ev = escalate.events;
    expect(ev.some((e) => e.kind === 'commit_move')).toBe(false);
    const esc = ev.find((e) => e.kind === 'escalate')!;
    expect(esc.margin!).toBeLessThan(esc.tau!);
    expect(esc.because).toMatch(/< tau/);
    expect(ev.some((e) => e.kind === 'tool_call' && e.tool === 'request_human_approval')).toBe(true);
    const s = summarizeTrace(ev);
    expect(s.outcome).toBe('escalate');
    expect(s.margin!).toBeLessThan(s.tau!);
  });

  it('dose: cumulative_dose from acquire results, monotone, ends at the episode total', () => {
    for (const ev of [commit.events, escalate.events]) {
      const d = cumulativeDose(ev);
      expect(d[0]).toBe(0);
      for (let i = 1; i < d.length; i++) expect(d[i]!).toBeGreaterThanOrEqual(d[i - 1]!);
      const end = ev.at(-1)!;
      expect(d.at(-1)).toBe(end.result_summary!['dose']);
      expect(doseBudget(ev)).toBeUndefined();
      expect(doseBudget(ev, 1e8)).toBe(1e8);
    }
  });

  it('step latency uses latency_ms, else the timestamp gap', () => {
    const ev = commit.events;
    const iLoc = ev.findIndex((e) => e.tool === 'localize' && e.kind === 'tool_result');
    expect(stepLatency(ev, iLoc)).toEqual({ ms: ev[iLoc]!.latency_ms, derived: false });
    const d = stepLatency(ev, 1)!;
    expect(d.derived).toBe(true);
    expect(d.ms).toBeCloseTo(Date.parse(ev[1]!.ts) - Date.parse(ev[0]!.ts));
    expect(stepLatency(ev, 0)).toBeNull();
  });

  it('pixel size comes from the last acquire (fov_nm / width_px)', () => {
    expect(pixelSizeNm(commit.events)).toBe(1);
  });

  it('every event key is a contract key', () => {
    for (const e of [...commit.events, ...escalate.events]) {
      for (const k of Object.keys(e)) expect(TRACE_EVENT_KEYS).toContain(k);
    }
  });

  const schema = loadSchema('trace_event');
  it.skipIf(!schema)(`every line validates against trace_event.schema.json (${schemaDir()})`, () => {
    for (const [name, text] of [
      ['demo_commit.jsonl', commitText],
      ['demo_escalate.jsonl', escalateText],
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
