import { describe, expect, it, vi } from 'vitest';
import { HttpApi } from '../src/api/http';
import { MOCK_EVENT_MS, MOCK_QUEUED_MS, MOCK_RUN_MS, MockApi } from '../src/api/mock';
import { ApiError, newClientToken } from '../src/api/types';
import { createApi, loadConfig } from '../src/config';
import type { JobStatus } from '../src/types/contracts';
import { loadSchema, validate } from './helpers/schema';

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

const jobBody: JobStatus = {
  job_id: 'j1',
  status: 'queued',
  created_at: '2026-10-11T00:00:00Z',
  updated_at: '2026-10-11T00:00:00Z',
};

describe('HttpApi (stubbed fetch)', () => {
  it('builds URLs, methods and JSON bodies', async () => {
    const fetchFn = vi.fn(async () => json(jobBody, 202));
    const api = new HttpApi('https://api.example.com/', fetchFn as unknown as typeof fetch);
    const res = await api.createJob({ demo_id: 'demo_grid', mode: 'auto', agent: true, client_token: 't' });
    expect(res.job_id).toBe('j1');
    const [url, init] = fetchFn.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('https://api.example.com/v1/jobs');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({ demo_id: 'demo_grid', mode: 'auto', agent: true, client_token: 't' });
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('application/json');

    await api.getJob('a/b');
    expect((fetchFn.mock.calls[1] as unknown as [string])[0]).toBe('https://api.example.com/v1/jobs/a%2Fb');
  });

  it('maps error bodies to ApiError (FastAPI detail string)', async () => {
    const api = new HttpApi('', (async () => json({ detail: 'job not found' }, 404)) as unknown as typeof fetch);
    await expect(api.getJob('x')).rejects.toMatchObject({ name: 'ApiError', status: 404, message: 'job not found' });
  });

  it('wraps network failures', async () => {
    const api = new HttpApi('', (async () => {
      throw new TypeError('Failed to fetch');
    }) as unknown as typeof fetch);
    await expect(api.demos()).rejects.toMatchObject({ status: 0, code: 'network' });
    expect(await api.healthz()).toBe(false);
  });

  it('PUTs uploads to the presigned URL with the given headers', async () => {
    const fetchFn = vi.fn(async () => new Response(null, { status: 200 }));
    const api = new HttpApi('https://api', fetchFn as unknown as typeof fetch);
    const blob = new Blob(['x'], { type: 'image/png' });
    await api.putUpload({ upload_id: 'u', key: 'k', url: 'https://s3/put', headers: { 'Content-Type': 'image/png' }, expires_in: 60 }, blob);
    const [url, init] = fetchFn.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('https://s3/put');
    expect(init.method).toBe('PUT');
    expect(init.headers).toEqual({ 'Content-Type': 'image/png' });
  });

  it('fetches and parses the JSONL trace; 404 means no trace', async () => {
    const line = JSON.stringify({ run_id: 'r', step: 0, ts: '2026-10-11T00:00:00Z', kind: 'observe' });
    const ok = new HttpApi('https://api', (async () => new Response(`${line}\n\nbad\n`, { status: 200 })) as unknown as typeof fetch);
    const t = await ok.getTrace('j');
    expect(t?.events).toHaveLength(1);
    expect(t?.errors[0]?.line).toBe(3);
    expect(t?.synthetic).toBe(false);
    const missing = new HttpApi('https://api', (async () => new Response('', { status: 404 })) as unknown as typeof fetch);
    expect(await missing.getTrace('j')).toBeNull();
  });

  it('approve falls back to GET when the response has no JobStatus body', async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 204 }))
      .mockResolvedValueOnce(json({ ...jobBody, status: 'running' }));
    const api = new HttpApi('https://api', fetchFn as unknown as typeof fetch);
    const r = await api.approve('j1', { chosen_rank: 1, approver: 'Ada' });
    expect(r.status).toBe('running');
    const [url, init] = fetchFn.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('https://api/v1/jobs/j1/approval');
    expect(JSON.parse(init.body as string)).toEqual({ chosen_rank: 1, approver: 'Ada' });
  });

  it('resolves artifact URLs against the API base', () => {
    const api = new HttpApi('https://api/');
    expect(api.artifactUrl('j1', { kind: 'heatmap', uri: 'heatmap.png' })).toBe('https://api/v1/jobs/j1/artifacts/heatmap.png');
  });
});

describe('config', () => {
  it('uses the mock when apiBase is empty, the live client otherwise', async () => {
    const empty = await loadConfig((async () => json({ apiBase: '' })) as unknown as typeof fetch, { search: '', hash: '' });
    expect(empty).toMatchObject({ mock: true, mockReason: 'no-api-base' });
    expect(createApi(empty).kind).toBe('mock');
    const live = await loadConfig((async () => json({ apiBase: 'https://api/' })) as unknown as typeof fetch, { search: '', hash: '#/' });
    expect(live).toEqual({ apiBase: 'https://api', mock: false });
    expect(createApi(live).kind).toBe('live');
    const forced = await loadConfig((async () => json({ apiBase: 'https://api' })) as unknown as typeof fetch, { search: '?mock=1', hash: '' });
    expect(forced).toMatchObject({ mock: true, mockReason: 'query' });
    const broken = await loadConfig((async () => {
      throw new Error('offline');
    }) as unknown as typeof fetch, { search: '', hash: '' });
    expect(broken).toMatchObject({ mock: true, mockReason: 'config-error' });
  });

  it('generates RFC 4122 v4 client tokens', () => {
    expect(newClientToken()).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  });
});

describe('MockApi', () => {
  const setup = () => {
    let now = 1_000_000;
    const api = new MockApi({ latencyMs: 0, now: () => now });
    return { api, advance: (ms: number) => (now += ms) };
  };
  const jobSchema = loadSchema('job_status');
  const check = (j: JobStatus) => {
    if (jobSchema) expect(validate(jobSchema, j)).toEqual([]);
    if (j.result) expect(j.result.margin.decision === 'commit').toBe(j.result.margin.margin >= j.result.margin.tau);
  };

  it('serves version and the four demo sites', async () => {
    const { api } = setup();
    expect((await api.demos()).map((d) => d.id)).toEqual(['demo_finfet_easy', 'demo_dram_hard', 'demo_grid', 'demo_staggered']);
    expect((await api.version()).contract_version).toBe('1.0.0');
  });

  it('runs a localization job queued -> running -> succeeded deterministically', async () => {
    const { api, advance } = setup();
    const j = await api.createJob({ demo_id: 'demo_finfet_easy', mode: 'fast' });
    expect(j.status).toBe('queued');
    check(j);
    advance(MOCK_QUEUED_MS);
    expect((await api.getJob(j.job_id)).status).toBe('running');
    advance(MOCK_RUN_MS);
    const done = await api.getJob(j.job_id);
    expect(done.status).toBe('succeeded');
    expect(done.result?.engine).toBe('fake');
    expect(done.result?.mode).toBe('fast');
    check(done);
    expect(await api.getTrace(j.job_id)).toBeNull();
  });

  it('is idempotent on client_token and validates the body', async () => {
    const { api } = setup();
    const a = await api.createJob({ demo_id: 'demo_grid', client_token: 'tok' });
    const b = await api.createJob({ demo_id: 'demo_grid', client_token: 'tok' });
    expect(b.job_id).toBe(a.job_id);
    await expect(api.createJob({})).rejects.toBeInstanceOf(ApiError);
    await expect(api.createJob({ demo_id: 'nope' })).rejects.toMatchObject({ status: 422 });
    await expect(api.getJob('missing')).rejects.toMatchObject({ status: 404 });
  });

  it('agent episode reveals the synthetic trace progressively and commits', async () => {
    const { api, advance } = setup();
    const j = await api.createJob({ demo_id: 'demo_grid', agent: true });
    advance(MOCK_QUEUED_MS);
    const first = await api.getTrace(j.job_id);
    expect(first?.synthetic).toBe(true);
    expect(first?.events).toHaveLength(1);
    advance(MOCK_EVENT_MS * 20);
    const t = await api.getTrace(j.job_id);
    expect(t?.events.at(-1)?.kind).toBe('episode_end');
    const done = await api.getJob(j.job_id);
    expect(done.status).toBe('succeeded');
    expect(done.trace_key).toBeTruthy();
    check(done);
  });

  it('hard DRAM agent episode escalates, then resumes after approval', async () => {
    const { api, advance } = setup();
    const j = await api.createJob({ demo_id: 'demo_dram_hard', agent: true });
    advance(MOCK_QUEUED_MS + MOCK_EVENT_MS * 20);
    const waiting = await api.getJob(j.job_id);
    expect(waiting.status).toBe('awaiting_approval');
    expect(waiting.approval?.options).toHaveLength(3);
    expect(waiting.approval?.margin.decision).toBe('escalate');
    check(waiting);
    expect((await api.getTrace(j.job_id))?.events.at(-1)?.kind).toBe('escalate');

    await expect(api.approve(j.job_id, { chosen_rank: 1, approver: ' ' })).rejects.toMatchObject({ status: 422 });
    const after = await api.approve(j.job_id, { chosen_rank: 2, approver: 'Ada' });
    expect(after.status).toBe('succeeded');
    expect(after.result?.offset.x_px).toBe(after.approval?.options[1]?.x_px ?? after.result?.candidates[1]?.x_px);
    check(after);
    const trace = await api.getTrace(j.job_id);
    const last = trace!.events.at(-1)!;
    expect(last.kind).toBe('human_approval');
    expect(last.result_summary?.['chosen_rank']).toBe(2);
    await expect(api.approve(j.job_id, { chosen_rank: 1, approver: 'Ada' })).rejects.toMatchObject({ status: 409 });
  });

  it('reject-all ends the job as failed with rejected_by_operator', async () => {
    const { api, advance } = setup();
    const j = await api.createJob({ demo_id: 'demo_dram_hard', agent: true });
    advance(MOCK_QUEUED_MS + MOCK_EVENT_MS * 20);
    const r = await api.approve(j.job_id, { chosen_rank: null, approver: 'Ada' });
    expect(r.status).toBe('failed');
    expect(r.error?.code).toBe('rejected_by_operator');
    check(r);
  });

  it('accepts uploads only for png/tiff <= 20 MB and runs upload jobs', async () => {
    const { api } = setup();
    await expect(api.createUpload({ filename: 'a.jpg', content_type: 'image/jpeg', size_bytes: 1 })).rejects.toMatchObject({ status: 415 });
    await expect(api.createUpload({ filename: 'a.png', content_type: 'image/png', size_bytes: 21 * 1024 * 1024 })).rejects.toMatchObject({ status: 413 });
    const t1 = await api.createUpload({ filename: 't.png', content_type: 'image/png', size_bytes: 10 });
    const t2 = await api.createUpload({ filename: 's.tif', content_type: 'image/tiff', size_bytes: 10 });
    await api.putUpload(t1, new Blob(['x']));
    const j = await api.createJob({ template_key: t1.key, search_key: t2.key });
    expect(j.job_id).toMatch(/^mock-job-/);
  });

  it('serves synthetic SVG artifacts as data URIs', async () => {
    const { api } = setup();
    const j = await api.createJob({ demo_id: 'demo_grid' });
    const url = api.artifactUrl(j.job_id, { kind: 'heatmap', uri: 'heatmap.svg' });
    expect(url.startsWith('data:image/svg+xml')).toBe(true);
    expect(decodeURIComponent(url)).toContain('SYNTHETIC');
  });
});
