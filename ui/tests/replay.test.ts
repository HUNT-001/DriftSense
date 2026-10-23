import { describe, expect, it } from 'vitest';
import { REPLAY_QUEUED_MS, REPLAY_STEP_MS, ReplayApi, loadRecordings, revealTimes } from '../src/api/replay';
import { isSimulatedEvent, summarizeTrace } from '../src/lib/trace';
import type { JobStatus } from '../src/types/contracts';
import { fsLoader } from './helpers/samples';
import { loadSchema, validate } from './helpers/schema';

const jobSchema = loadSchema('job_status');
const resultSchema = loadSchema('localize_result');
const traceSchema = loadSchema('trace_event');

function setup() {
  let now = 1_000_000;
  const api = new ReplayApi({ loader: fsLoader, latencyMs: 0, now: () => now });
  return { api, advance: (ms: number) => (now += ms) };
}

const check = (j: JobStatus) => {
  if (jobSchema) expect(validate(jobSchema, j)).toEqual([]);
  if (j.result) expect(j.result.margin.decision === 'commit').toBe(j.result.margin.margin >= j.result.margin.tau);
};

describe('recorded samples', () => {
  it('load: trace + capture whose decisive margin reproduces the trace', async () => {
    const recs = await loadRecordings(fsLoader);
    expect(recs.map((r) => r.name)).toEqual(['demo_commit', 'demo_escalate']);
    for (const r of recs) {
      const entry = r.capture.localize[0]!;
      const line = r.events.find((e) => e.step === entry.trace_step)!;
      expect(line.kind).toBe('tool_result');
      expect(line.tool).toBe('localize');
      expect(entry.result.margin.margin).toBe(line.margin);
      if (resultSchema) expect(validate(resultSchema, entry.result)).toEqual([]);
      expect(Object.keys(entry.artifact_files).sort()).toEqual(['ncc_heatmap.png', 'overlay.png', 'roughness_heatmap.png', 'top5_gallery.png']);
    }
    const esc = recs[1]!.capture.approval_request!;
    expect(esc.options).toHaveLength(3);
    expect(esc.margin.margin).toBeLessThan(esc.margin.tau);
    expect(recs[0]!.capture.approval_request).toBeNull();
  });

  it('pacing: a full replay finishes well inside the 60 s judge budget', async () => {
    const recs = await loadRecordings(fsLoader);
    for (const r of recs) {
      const t = REPLAY_QUEUED_MS + revealTimes(r.events).at(-1)! + 4 * REPLAY_STEP_MS;
      expect(t).toBeLessThan(20_000);
      expect(t).toBeGreaterThan(3_000); // slow enough to follow
    }
  });
});

describe('ReplayApi', () => {
  it('lists the two recorded episodes as agent demos', async () => {
    const { api } = setup();
    const demos = await api.demos();
    expect(demos.map((d) => d.demo_id)).toEqual(['vsem-dram-0', 'vsem-dram-1']);
    expect(demos.every((d) => d.agent)).toBe(true);
    expect((await api.version()).platform).toMatch(/offline replay of a recorded trace/);
  });

  it('commit episode: queued -> running with a growing trace -> succeeded with the recorded result', async () => {
    const { api, advance } = setup();
    const j = await api.createEpisode({ family: 'dram', seed: 0 });
    expect(j.status).toBe('queued');
    check(j);
    expect(await api.getTrace(j.job_id)).toBeNull();
    advance(REPLAY_QUEUED_MS);
    const running = await api.getJob(j.job_id);
    expect(running.status).toBe('running');
    expect(running.trace_key).toBe('traces/dram-0.jsonl');
    const t1 = await api.getTrace(j.job_id);
    expect(t1?.source).toBe('replay');
    expect(t1?.events).toHaveLength(1);
    advance(3000);
    const t2 = await api.getTrace(j.job_id);
    expect(t2!.events.length).toBeGreaterThan(1);
    expect(t2!.events.length).toBeLessThan(17);
    advance(60_000);
    const done = await api.getJob(j.job_id);
    expect(done.status).toBe('succeeded');
    check(done);
    expect(done.result?.candidates[0]?.score).toBeCloseTo(0.814, 3);
    const t3 = await api.getTrace(j.job_id);
    expect(t3!.events).toHaveLength(17);
    expect(t3!.events.some(isSimulatedEvent)).toBe(false);
    const ep = (await api.getArtifactJson(j.job_id, 'episode.json')) as Record<string, unknown>;
    expect(ep['outcome']).toBe('commit');
    expect(api.artifactUrl(j.job_id, { kind: 'heatmap', uri: 'jobs/x/ncc_heatmap.png' })).toBe('samples/demo_commit/step7_ncc_heatmap.webp');
    expect(api.artifactAltText(j.job_id, { kind: 'heatmap', uri: 'ncc_heatmap.png' })?.length).toBeGreaterThan(10);
  });

  it('escalate episode pauses at request_human_approval with the recorded top-3', async () => {
    const { api, advance } = setup();
    const j = await api.createEpisode({ family: 'dram', seed: 1 });
    advance(60_000);
    const w = await api.getJob(j.job_id);
    expect(w.status).toBe('awaiting_approval');
    check(w);
    expect(w.approval?.options.map((o) => o.rank)).toEqual([1, 2, 3]);
    const t = await api.getTrace(j.job_id);
    expect(t!.events.at(-1)).toMatchObject({ kind: 'tool_call', tool: 'request_human_approval' });
    // Recorded auto:none answer is NOT shown; the operator answers it.
    expect(t!.events.some((e) => e.result_summary?.['approver'] === 'auto:none')).toBe(false);
  });

  it('approve rank k -> simulated operator events, then succeeded with offset = candidate k', async () => {
    const { api, advance } = setup();
    const j = await api.createEpisode({ family: 'dram', seed: 1 });
    advance(60_000);
    await expect(api.approve(j.job_id, { chosen_rank: 1, approver: ' ' })).rejects.toMatchObject({ status: 422 });
    const after = await api.approve(j.job_id, { chosen_rank: 2, approver: 'Ada', note: 'second looks right' });
    expect(after.status).toBe('running');
    advance(10_000);
    const done = await api.getJob(j.job_id);
    expect(done.status).toBe('succeeded');
    check(done);
    const opt2 = (await api.recordings())[1]!.capture.approval_request!.options[1]!;
    expect(done.result?.offset).toMatchObject({ x_px: opt2.x_px, y_px: opt2.y_px });
    expect(done.result?.warnings?.some((w) => w.includes('"operator_choice"'))).toBe(true);
    const ev = (await api.getTrace(j.job_id))!.events;
    const sim = ev.filter(isSimulatedEvent);
    expect(sim.map((e) => e.kind)).toEqual(['tool_result', 'escalate', 'human_approval', 'episode_end']);
    expect(sim[0]!.result_summary).toMatchObject({ chosen_rank: 2, approver: 'Ada', note: 'second looks right' });
    expect(ev.map((e) => e.step)).toEqual(ev.map((_, i) => i)); // contiguous steps
    if (traceSchema) for (const e of ev) expect(validate(traceSchema, e)).toEqual([]);
    expect(summarizeTrace(ev).outcome).toBe('escalate');
    await expect(api.approve(j.job_id, { chosen_rank: 1, approver: 'Ada' })).rejects.toMatchObject({ status: 409 });
  });

  it('reject all -> failed with rejected_by_operator', async () => {
    const { api, advance } = setup();
    const j = await api.createEpisode({ family: 'dram', seed: 1 });
    advance(60_000);
    await api.approve(j.job_id, { chosen_rank: null, approver: 'Ada' });
    advance(10_000);
    const r = await api.getJob(j.job_id);
    expect(r.status).toBe('failed');
    expect(r.error?.code).toBe('rejected_by_operator');
    check(r);
  });

  it('unrecorded family/seed plays the closest recording; single localization has no trace', async () => {
    const { api, advance } = setup();
    const e = await api.createEpisode({ family: 'finfet', seed: 99 });
    expect(api.describeJob(e.job_id)).toMatchObject({ recording: 'demo_escalate', requested: { family: 'finfet', seed: 99 } });
    const j = await api.createJob({ demo_id: 'vsem-dram-0', agent: false });
    advance(10_000);
    const done = await api.getJob(j.job_id);
    expect(done.status).toBe('succeeded');
    expect(done.trace_key).toBeNull();
    expect(await api.getTrace(j.job_id)).toBeNull();
    await expect(api.createJob({ demo_id: 'nope' })).rejects.toMatchObject({ status: 422 });
    await expect(api.createUpload({ filename: 'a.png', content_type: 'image/png', size_bytes: 1 })).rejects.toMatchObject({ code: 'offline' });
    await expect(api.getJob('missing')).rejects.toMatchObject({ status: 404 });
  });

  it('is idempotent on client_token', async () => {
    const { api } = setup();
    const a = await api.createEpisode({ family: 'dram', seed: 0, client_token: 't' });
    const b = await api.createEpisode({ family: 'dram', seed: 0, client_token: 't' });
    expect(b.job_id).toBe(a.job_id);
  });
});
