import type { Artifact, JobStatus } from '../types/contracts';
import { fmt } from './format';

/** Artifact names the service accepts (service/api.py: ^[A-Za-z0-9._-]{1,128}$, no ".."). */
const NAME_RE = /^[A-Za-z0-9._-]{1,128}$/;

/** Last path segment of an Artifact.uri (`jobs/<id>/heatmap.png`, `s3://b/jobs/<id>/x.png`, `./x.png`). */
export function artifactBasename(uri: string): string {
  const noQuery = uri.split(/[?#]/)[0] ?? '';
  const seg = noQuery.replace(/^s3:\/\/[^/]+\//i, '').split(/[\\/]/).filter(Boolean).pop() ?? '';
  return seg;
}

/**
 * Resolve an Artifact.uri to a browser URL.
 *  - `http://` / `https://` (e.g. a presigned S3 GET): used as-is.
 *  - anything else (the service stores the S3 key `jobs/<id>/<name>`; locally a relative file
 *    name; or `s3://bucket/key`): resolved BY BASENAME to `${apiBase}/v1/jobs/{jobId}/artifacts/<name>`
 *    (service README "Artifact URI convention"). Returns '' when the name is not servable.
 */
export function resolveArtifactUrl(apiBase: string, jobId: string, uri: string): string {
  if (/^https?:\/\//i.test(uri)) return uri;
  const name = artifactBasename(uri);
  if (!NAME_RE.test(name) || name.includes('..')) return '';
  const base = apiBase.replace(/\/+$/, '');
  return `${base}/v1/jobs/${encodeURIComponent(jobId)}/artifacts/${encodeURIComponent(name)}`;
}

const KIND_TITLE: Record<Artifact['kind'], string> = {
  heatmap: 'Score heatmap',
  overlay: 'Match overlay on the search frame',
  contour: 'Contour fingerprint view',
  candidates: 'Top candidates gallery',
  json: 'Raw result data',
};

/** Titles for the S9 explain artifacts the v2 engine emits (driftsense/v2/explain.py). */
const NAME_TITLE: Record<string, string> = {
  ncc_heatmap: 'NCC heatmap (where a copy could be)',
  roughness_heatmap: 'Roughness heatmap (which copy is real)',
  top5_gallery: 'Top-5 candidate crops',
  overlay: 'Match overlay on the search frame',
};

function stem(a: Artifact): string {
  return artifactBasename(a.uri).replace(/\.[a-z0-9]+$/i, '');
}

export function artifactTitle(a: Artifact): string {
  return NAME_TITLE[stem(a)] ?? KIND_TITLE[a.kind] ?? a.kind;
}

/** Display order: NCC vs roughness side by side first, then overlay, gallery, rest. */
export function artifactOrder(a: Artifact): number {
  const i = ['ncc_heatmap', 'roughness_heatmap', 'overlay', 'top5_gallery'].indexOf(stem(a));
  return i >= 0 ? i : 10;
}

/**
 * Artifact has no alt-text field in contract 1.0.0 (the engine computes one but the contract drops
 * it). Prefer an alt supplied by the source; otherwise describe the kind plus the job's result.
 */
export function artifactAlt(a: Artifact, job: Pick<JobStatus, 'job_id' | 'result'>, provided?: string): string {
  if (provided && provided.trim().length > 10) return provided.trim();
  const r = job.result;
  const lattice = r?.lattice_type && r.lattice_type !== 'unknown' ? ` on a ${r.lattice_type} lattice` : '';
  const top = r?.candidates?.[0];
  const topText = top ? ` Rank-1 candidate at (${fmt(top.x_px, 1)}, ${fmt(top.y_px, 1)}) px.` : '';
  const n = r?.candidates?.length ?? 0;
  switch (stem(a)) {
    case 'ncc_heatmap':
      return `Normalised cross-correlation heatmap for job ${job.job_id}${lattice}: every periodic copy lights up, brighter means a stronger template match.${topText}`;
    case 'roughness_heatmap':
      return `Line-edge roughness residual heatmap for job ${job.job_id}${lattice}: the roughness fingerprint that separates the real copy from its look-alikes.${topText}`;
    default:
  }
  switch (a.kind) {
    case 'heatmap':
      return `Candidate score heatmap for job ${job.job_id}${lattice}: brighter means a higher score.${topText}`;
    case 'overlay':
      return `Search image for job ${job.job_id}${lattice} with the matched template location outlined.${topText}`;
    case 'contour':
      return `Contour fingerprint view for job ${job.job_id}${lattice}: detected cell outlines used for shape matching.${topText}`;
    case 'candidates':
      return `Crops of the top ${n} candidate locations for job ${job.job_id}, ranked with score captions.${topText}`;
    default:
      return `${artifactTitle(a)} for job ${job.job_id}.`;
  }
}

export function isImageArtifact(a: Artifact): boolean {
  return a.kind !== 'json' && !/\.(json|jsonl|csv|txt)$/i.test(a.uri);
}
