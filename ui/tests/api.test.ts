import { describe, expect, it, vi } from 'vitest';
import { HttpApi } from '../src/api/http';
import { episodeParams, isEpisodeDemo, newClientToken, type DemoInfo } from '../src/api/types';
import { apiOverride, createApi, loadConfig, resolveConfig, wantsOffline } from '../src/config';
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

type Call = [string, RequestInit];

describe('HttpApi (stubbed fetch)', () => {
  it('POST /v1/jobs with the JobCreate body', async () => {
    const fetchFn = vi.fn(async () => json(jobBody, 202));
    const api = new HttpApi('https://api.example.com/', fetchFn as unknown as typeof fetch);
    const res = await api.createJob({ demo_id: 'demo_grid', mode: 'auto', agent: false, client_token: 't' });
    expect(res.job_id).toBe('j1');
    const [url, init] = fetchFn.mock.calls[0] as unknown as Call;
    expect(url).toBe('https://api.example.com/v1/jobs');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({ demo_id: 'demo_grid', mode: 'auto', agent: false, client_token: 't' });
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('application/json');
    expect(init.cache).toBe('no-store');
    const schema = loadSchema('job_create');
    if (schema) expect(validate(schema, JSON.parse(init.body as string))).toEqual([]);

    await api.getJob('a/b');
    expect((fetchFn.mock.calls[1] as unknown as [string])[0]).toBe('https://api.example.com/v1/jobs/a%2Fb');
  });

  it('POST /v1/agent/episodes {family, seed, client_token}', async () => {
    const fetchFn = vi.fn(async () => json({ ...jobBody, job_id: 'e1' }, 202));
    const api = new HttpApi('https://api', fetchFn as unknown as typeof fetch);
    const r = await api.createEpisode({ family: 'dram', seed: 7, client_token: 'tok' });
    expect(r.job_id).toBe('e1');
    const [url, init] = fetchFn.mock.calls[0] as unknown as Call;
    expect(url).toBe('https://api/v1/agent/episodes');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({ family: 'dram', seed: 7, client_token: 'tok' });
  });

  it('maps the service ErrorInfo body {code, message} to ApiError', async () => {
    const api = new HttpApi('', (async () => json({ code: 'not_found', message: 'job not found' }, 404)) as unknown as typeof fetch);
    await expect(api.getJob('x')).rejects.toMatchObject({ name: 'ApiError', status: 404, code: 'not_found', message: 'job not found' });
  });

  it('wraps network failures; healthz is false on failure or timeout', async () => {
    const api = new HttpApi('', (async () => {
      throw new TypeError('Failed to fetch');
    }) as unknown as typeof fetch);
    await expect(api.demos()).rejects.toMatchObject({ status: 0, code: 'network' });
    expect(await api.healthz()).toBe(false);
    const hang = new HttpApi('https://api', ((_u: string, init: RequestInit) =>
      new Promise((_r, rej) => init.signal?.addEventListener('abort', () => rej(new Error('aborted'))))) as unknown as typeof fetch);
    expect(await hang.healthz(20)).toBe(false);
    const up = new HttpApi('https://api', (async () => json({ ok: true })) as unknown as typeof fetch);
    expect(await up.healthz()).toBe(true);
  });

  it('PUTs uploads to the presigned URL with exactly the returned headers', async () => {
    const fetchFn = vi.fn(async () => new Response(null, { status: 200 }));
    const api = new HttpApi('https://api', fetchFn as unknown as typeof fetch);
    const blob = new Blob(['x'], { type: 'image/png' });
    const headers = { 'Content-Type': 'image/png', 'Content-Length': '1' };
    await api.putUpload({ upload_id: 'u', key: 'k', url: 'https://s3/put?sig', method: 'PUT', headers, expires_in: 900 }, blob);
    const [url, init] = fetchFn.mock.calls[0] as unknown as Call;
    expect(url).toBe('https://s3/put?sig');
    expect(init.method).toBe('PUT');
    expect(init.headers).toEqual(headers);
    expect(init.body).toBe(blob);
  });

  it('fetches and parses the NDJSON trace; 404 not_ready means no trace yet', async () => {
    const line = JSON.stringify({ run_id: 'r', step: 0, ts: '2026-10-11T00:00:00Z', kind: 'observe' });
    const ok = new HttpApi('https://api', (async () => new Response(`${line}\n\nbad\n`, { status: 200 })) as unknown as typeof fetch);
    const t = await ok.getTrace('j');
    expect(t?.events).toHaveLength(1);
    expect(t?.errors[0]?.line).toBe(3);
    expect(t?.source).toBe('live');
    const missing = new HttpApi('https://api', (async () => json({ code: 'not_ready', message: 'no trace yet' }, 404)) as unknown as typeof fetch);
    expect(await missing.getTrace('j')).toBeNull();
  });

  it('getArtifactJson reads /v1/jobs/{id}/artifacts/{name}; 404 -> null', async () => {
    const fetchFn = vi.fn().mockResolvedValueOnce(json({ outcome: 'commit' })).mockResolvedValueOnce(json({ code: 'not_found', message: 'x' }, 404));
    const api = new HttpApi('https://api', fetchFn as unknown as typeof fetch);
    expect(await api.getArtifactJson('j1', 'episode.json')).toEqual({ outcome: 'commit' });
    expect((fetchFn.mock.calls[0] as [string])[0]).toBe('https://api/v1/jobs/j1/artifacts/episode.json');
    expect(await api.getArtifactJson('j1', 'episode.json')).toBeNull();
  });

  it('approve POSTs the ApprovalDecision and falls back to GET when the body is not a JobStatus', async () => {
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(new Response(null, { status: 204 }))
      .mockResolvedValueOnce(json({ ...jobBody, status: 'running' }));
    const api = new HttpApi('https://api', fetchFn as unknown as typeof fetch);
    const r = await api.approve('j1', { chosen_rank: 1, approver: 'Ada' });
    expect(r.status).toBe('running');
    const [url, init] = fetchFn.mock.calls[0] as Call;
    expect(url).toBe('https://api/v1/jobs/j1/approval');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({ chosen_rank: 1, approver: 'Ada' });
  });

  it('resolves artifact URIs (S3 keys jobs/<id>/<name>) by basename', () => {
    const api = new HttpApi('https://api/');
    expect(api.artifactUrl('j1', { kind: 'heatmap', uri: 'jobs/j1/ncc_heatmap.png' })).toBe('https://api/v1/jobs/j1/artifacts/ncc_heatmap.png');
  });
});

describe('demos', () => {
  const d = (x: Partial<DemoInfo>): DemoInfo => ({ demo_id: 'demo_x', title: 't', lattice: 'dram', difficulty: 'easy', ...x });
  it('detects agent-episode demos by flag or the vsem- prefix', () => {
    expect(isEpisodeDemo(d({ demo_id: 'vsem-dram-7' }))).toBe(true);
    expect(isEpisodeDemo(d({ agent: true }))).toBe(true);
    expect(isEpisodeDemo(d({ kind: 'agent_episode' }))).toBe(true);
    expect(isEpisodeDemo(d({}))).toBe(false);
  });
  it('derives family + seed from the id or explicit fields', () => {
    expect(episodeParams(d({ demo_id: 'vsem-dram-7' }))).toEqual({ family: 'dram', seed: 7 });
    expect(episodeParams(d({ demo_id: 'vsem-finfet-12' }))).toEqual({ family: 'finfet', seed: 12 });
    expect(episodeParams(d({ demo_id: 'ep', agent: true, family: 'finfet', seed: 3 }))).toEqual({ family: 'finfet', seed: 3 });
    expect(episodeParams(d({ demo_id: 'ep', agent: true, lattice: 'grid' }))).toEqual({ family: 'random', seed: 0 });
  });
});

describe('config', () => {
  const cfgFetch = (apiBase: string) => (async () => json({ apiBase })) as unknown as typeof fetch;

  it('offline when apiBase is empty or config.json is missing; live otherwise', async () => {
    const empty = await loadConfig(cfgFetch(''), { search: '', hash: '' });
    expect(empty).toMatchObject({ offline: true, offlineReason: 'no-api-base' });
    expect(createApi(empty).kind).toBe('replay');
    const live = await loadConfig(cfgFetch('https://api/'), { search: '', hash: '#/' });
    expect(live).toEqual({ apiBase: 'https://api', offline: false });
    expect(createApi(live).kind).toBe('live');
    const broken = await loadConfig((async () => {
      throw new Error('offline');
    }) as unknown as typeof fetch, { search: '', hash: '' });
    expect(broken).toMatchObject({ offline: true, offlineReason: 'config-error' });
  });

  it('?api= overrides config.json; ?demo=offline forces offline replay', async () => {
    const fetchFn = vi.fn(cfgFetch('https://from-config'));
    const o = await loadConfig(fetchFn, { search: '?api=https://override.example/', hash: '' });
    expect(o).toEqual({ apiBase: 'https://override.example', offline: false });
    expect(fetchFn).not.toHaveBeenCalled();
    expect(apiOverride({ search: '?api=javascript:alert(1)', hash: '' })).toBeNull();
    expect(apiOverride({ search: '', hash: '#/jobs/x?api=http://localhost:8000' })).toBe('http://localhost:8000');
    const forced = await loadConfig(cfgFetch('https://api'), { search: '?demo=offline', hash: '' });
    expect(forced).toMatchObject({ offline: true, offlineReason: 'query' });
    expect(wantsOffline({ search: '', hash: '#/?demo=offline' })).toBe(true);
    expect(wantsOffline({ search: '?mock=1', hash: '' })).toBe(true);
    expect(wantsOffline({ search: '?demo=live', hash: '' })).toBe(false);
  });

  it('falls back to offline replay when /healthz is unreachable', async () => {
    const live = { apiBase: 'https://api', offline: false };
    const down = await resolveConfig(live, (async () => {
      throw new TypeError('Failed to fetch');
    }) as unknown as typeof fetch);
    expect(down).toEqual({ apiBase: 'https://api', offline: true, offlineReason: 'unreachable' });
    const err500 = await resolveConfig(live, (async () => json({}, 503)) as unknown as typeof fetch);
    expect(err500.offline).toBe(true);
    const up = await resolveConfig(live, (async () => json({ ok: true })) as unknown as typeof fetch);
    expect(up).toBe(live);
  });

  it('generates RFC 4122 v4 client tokens', () => {
    expect(newClientToken()).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  });
});
