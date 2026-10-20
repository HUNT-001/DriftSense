import type { Artifact, JobStatus } from '../types/contracts';
import { fmt } from './format';

/**
 * Resolve an Artifact.uri to a browser URL.
 *  - `http://` / `https://` (e.g. presigned S3 GET): used as-is.
 *  - `s3://bucket/key`: the service must proxy it; we use the key's last segment (file name).
 *  - anything else (relative path such as `heatmap.png`): resolved against
 *    `${apiBase}/v1/jobs/{jobId}/artifacts/`.
 * ASSUMPTION (handoff to service): GET /v1/jobs/{id}/artifacts/{name} serves (or 302-redirects to)
 * the artifact bytes. See ui/README.md.
 */
export function resolveArtifactUrl(apiBase: string, jobId: string, uri: string): string {
  if (/^https?:\/\//i.test(uri)) return uri;
  let path = uri;
  const s3 = /^s3:\/\/[^/]+\/(.+)$/i.exec(uri);
  if (s3) path = s3[1]!.split('/').pop() ?? s3[1]!;
  path = path.replace(/^(\.\/)+/, '').replace(/^\/+/, '');
  const encoded = path
    .split('/')
    .filter((seg) => seg && seg !== '..' && seg !== '.')
    .map(encodeURIComponent)
    .join('/');
  const base = apiBase.replace(/\/+$/, '');
  return `${base}/v1/jobs/${encodeURIComponent(jobId)}/artifacts/${encoded}`;
}

const KIND_TITLE: Record<Artifact['kind'], string> = {
  heatmap: 'Candidate score heatmap',
  overlay: 'Detection overlay',
  contour: 'Contour fingerprint view',
  candidates: 'Top candidates',
  json: 'Raw result data',
};

export function artifactTitle(a: Artifact): string {
  return KIND_TITLE[a.kind] ?? a.kind;
}

/**
 * Artifact has no alt-text field in contract 1.0.0 (handoff: consider `alt?: string` from S9).
 * Until then we generate a descriptive alt from the kind plus the job's result.
 */
export function artifactAlt(a: Artifact, job: Pick<JobStatus, 'job_id' | 'result'>): string {
  const r = job.result;
  const lattice = r?.lattice_type && r.lattice_type !== 'unknown' ? ` on a ${r.lattice_type} lattice` : '';
  const top = r?.candidates?.[0];
  const topText = top ? ` Rank-1 candidate at (${fmt(top.x_px, 1)}, ${fmt(top.y_px, 1)}) px.` : '';
  const n = r?.candidates?.length ?? 0;
  switch (a.kind) {
    case 'heatmap':
      return `Candidate score heatmap for job ${job.job_id}${lattice}, viridis scale: brighter yellow means a higher fused score.${topText}`;
    case 'overlay':
      return `Search image for job ${job.job_id}${lattice} with the matched template location outlined.${topText}`;
    case 'contour':
      return `Contour fingerprint view for job ${job.job_id}${lattice}: detected cell outlines used for shape matching.${topText}`;
    case 'candidates':
      return `Search image for job ${job.job_id} with the top ${n} candidate locations marked by rank number.${topText}`;
    default:
      return `${artifactTitle(a)} for job ${job.job_id}.`;
  }
}

export function isImageArtifact(a: Artifact): boolean {
  return a.kind !== 'json';
}
