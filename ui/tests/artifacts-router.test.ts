import { describe, expect, it } from 'vitest';
import { artifactAlt, artifactBasename, artifactOrder, artifactTitle, resolveArtifactUrl } from '../src/lib/artifacts';
import { jobHref, parseRoute } from '../src/router';

describe('artifact URI resolution', () => {
  const base = 'https://api.example.com/';
  it('passes absolute http(s) URLs through (presigned GET)', () => {
    expect(resolveArtifactUrl(base, 'j', 'https://s3.example/x.png?sig=1')).toBe('https://s3.example/x.png?sig=1');
  });
  it('resolves the service S3 key jobs/<id>/<name> by basename', () => {
    expect(resolveArtifactUrl(base, 'abc', 'jobs/abc/ncc_heatmap.png')).toBe('https://api.example.com/v1/jobs/abc/artifacts/ncc_heatmap.png');
    expect(resolveArtifactUrl('', 'abc', 'jobs/abc/result.json')).toBe('/v1/jobs/abc/artifacts/result.json');
  });
  it('resolves local relative names and s3:// URIs by basename', () => {
    expect(resolveArtifactUrl(base, 'j 1', './overlay.png')).toBe('https://api.example.com/v1/jobs/j%201/artifacts/overlay.png');
    expect(resolveArtifactUrl(base, 'j', 's3://bucket/jobs/j/top5_gallery.png')).toBe('https://api.example.com/v1/jobs/j/artifacts/top5_gallery.png');
    expect(artifactBasename('jobs\\j\\x.png')).toBe('x.png');
  });
  it('refuses names the service would reject (spaces, traversal)', () => {
    expect(resolveArtifactUrl(base, 'j', 'heat map.png')).toBe('');
    expect(resolveArtifactUrl(base, 'j', '../..')).toBe('');
    expect(resolveArtifactUrl(base, 'j', '../../etc/passwd')).toBe('https://api.example.com/v1/jobs/j/artifacts/passwd');
  });
  it('titles and orders NCC before roughness before overlay', () => {
    const a = (uri: string) => ({ kind: 'heatmap' as const, uri });
    expect(artifactTitle(a('jobs/j/ncc_heatmap.png'))).toMatch(/NCC/);
    expect(artifactTitle(a('jobs/j/roughness_heatmap.png'))).toMatch(/Roughness/);
    const names = ['overlay.png', 'roughness_heatmap.png', 'x.png', 'ncc_heatmap.png'];
    expect([...names].sort((p, q) => artifactOrder(a(p)) - artifactOrder(a(q)))).toEqual(['ncc_heatmap.png', 'roughness_heatmap.png', 'overlay.png', 'x.png']);
  });
  it('alt text: prefers the engine-provided alt, else generates a description', () => {
    const job = {
      job_id: 'j9',
      result: {
        engine: 'v2' as const,
        engine_version: '0',
        mode: 'auto' as const,
        lattice_type: 'finfet' as const,
        offset: { x_px: 1, y_px: 2 },
        candidates: [{ rank: 1, x_px: 10, y_px: 20, score: 0.9 }] as [{ rank: number; x_px: number; y_px: number; score: number }],
        margin: { margin: 0.3, tau: 0.2, calibrated: true, decision: 'commit' as const },
        total_ms: 1,
      },
    };
    const gen = artifactAlt({ kind: 'heatmap', uri: 'jobs/j9/roughness_heatmap.png' }, job);
    expect(gen).toMatch(/roughness residual heatmap for job j9 on a finfet lattice/);
    expect(gen).toMatch(/\(10\.0, 20\.0\) px/);
    expect(artifactAlt({ kind: 'heatmap', uri: 'h.png' }, job, 'NCC heatmap: bright peaks at every copy')).toBe('NCC heatmap: bright peaks at every copy');
  });
});

describe('router', () => {
  it('parses routes', () => {
    expect(parseRoute('')).toEqual({ name: 'home' });
    expect(parseRoute('#/')).toEqual({ name: 'home' });
    expect(parseRoute('#/?demo=offline')).toEqual({ name: 'home' });
    expect(parseRoute('#/about')).toEqual({ name: 'about' });
    expect(parseRoute('#/responsible/')).toEqual({ name: 'responsible' });
    expect(parseRoute(jobHref('a b'))).toEqual({ name: 'job', id: 'a b' });
    expect(parseRoute('#/nope')).toEqual({ name: 'notfound', path: '/nope' });
  });
});
