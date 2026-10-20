import { describe, expect, it } from 'vitest';
import { artifactAlt, resolveArtifactUrl } from '../src/lib/artifacts';
import { jobHref, parseRoute } from '../src/router';
import { wantsMock } from '../src/config';

describe('resolveArtifactUrl', () => {
  const base = 'https://api.example.com/';
  it('passes absolute http(s) URLs through', () => {
    expect(resolveArtifactUrl(base, 'j', 'https://s3.example/x.png?sig=1')).toBe('https://s3.example/x.png?sig=1');
  });
  it('resolves relative paths under /v1/jobs/{id}/artifacts/', () => {
    expect(resolveArtifactUrl(base, 'j 1', './heatmap.png')).toBe('https://api.example.com/v1/jobs/j%201/artifacts/heatmap.png');
    expect(resolveArtifactUrl('', 'j', 'sub/over lay.png')).toBe('/v1/jobs/j/artifacts/sub/over%20lay.png');
  });
  it('maps s3:// keys to their file name and drops traversal segments', () => {
    expect(resolveArtifactUrl(base, 'j', 's3://bucket/artifacts/j/heatmap.png')).toBe('https://api.example.com/v1/jobs/j/artifacts/heatmap.png');
    expect(resolveArtifactUrl(base, 'j', '../../etc/passwd')).toBe('https://api.example.com/v1/jobs/j/artifacts/etc/passwd');
  });
  it('generates descriptive alt text', () => {
    const job = {
      job_id: 'j9',
      result: {
        engine: 'fake' as const,
        engine_version: '0',
        mode: 'auto' as const,
        lattice_type: 'finfet' as const,
        offset: { x_px: 1, y_px: 2 },
        candidates: [{ rank: 1, x_px: 10, y_px: 20, score: 0.9 }] as [{ rank: number; x_px: number; y_px: number; score: number }],
        margin: { margin: 0.3, tau: 0.2, calibrated: true, decision: 'commit' as const },
        total_ms: 1,
      },
    };
    const alt = artifactAlt({ kind: 'heatmap', uri: 'h.png' }, job);
    expect(alt).toMatch(/heatmap for job j9 on a finfet lattice/);
    expect(alt).toMatch(/\(10\.0, 20\.0\) px/);
  });
});

describe('router', () => {
  it('parses routes', () => {
    expect(parseRoute('')).toEqual({ name: 'home' });
    expect(parseRoute('#/')).toEqual({ name: 'home' });
    expect(parseRoute('#/?mock=1')).toEqual({ name: 'home' });
    expect(parseRoute('#/about')).toEqual({ name: 'about' });
    expect(parseRoute('#/responsible/')).toEqual({ name: 'responsible' });
    expect(parseRoute(jobHref('a b'))).toEqual({ name: 'job', id: 'a b' });
    expect(parseRoute('#/nope')).toEqual({ name: 'notfound', path: '/nope' });
  });
  it('detects mock mode from query or hash', () => {
    expect(wantsMock('?mock=1', '')).toBe(true);
    expect(wantsMock('', '#/?mock=1')).toBe(true);
    expect(wantsMock('?mock=0', '#/')).toBe(false);
  });
});
