import type { ApprovalDecision, Artifact, JobCreate, JobStatus } from '../types/contracts';
import { resolveArtifactUrl } from '../lib/artifacts';
import { isSyntheticTrace, parseTraceJsonl } from '../lib/trace';
import { ApiError, type Api, type DemoSite, type TraceFetch, type UploadRequest, type UploadTicket, type VersionInfo } from './types';

export type FetchFn = typeof fetch;

/** Real job API client (FastAPI service behind API Gateway). */
export class HttpApi implements Api {
  readonly kind = 'live' as const;
  readonly apiBase: string;

  constructor(
    apiBase: string,
    private readonly fetchFn: FetchFn = (...a) => globalThis.fetch(...a),
  ) {
    this.apiBase = apiBase.replace(/\/+$/, '');
  }

  private async send(method: string, path: string, body?: unknown): Promise<Response> {
    const init: RequestInit = { method, headers: { Accept: 'application/json' } };
    if (body !== undefined) {
      init.headers = { ...init.headers, 'Content-Type': 'application/json' };
      init.body = JSON.stringify(body);
    }
    let res: Response;
    try {
      res = await this.fetchFn(`${this.apiBase}${path}`, init);
    } catch (e) {
      throw new ApiError(`Network error contacting the API: ${(e as Error).message}`, 0, 'network');
    }
    if (!res.ok) throw await toApiError(res);
    return res;
  }

  private async json<T>(method: string, path: string, body?: unknown): Promise<T> {
    const res = await this.send(method, path, body);
    try {
      return (await res.json()) as T;
    } catch {
      throw new ApiError(`Malformed JSON from ${path}`, res.status, 'bad_json');
    }
  }

  async healthz(): Promise<boolean> {
    try {
      await this.send('GET', '/healthz');
      return true;
    } catch {
      return false;
    }
  }

  version(): Promise<VersionInfo> {
    return this.json('GET', '/v1/version');
  }

  demos(): Promise<DemoSite[]> {
    return this.json('GET', '/v1/demos');
  }

  createUpload(req: UploadRequest): Promise<UploadTicket> {
    return this.json('POST', '/v1/uploads', req);
  }

  async putUpload(ticket: UploadTicket, file: Blob): Promise<void> {
    let res: Response;
    try {
      res = await this.fetchFn(ticket.url, { method: 'PUT', headers: ticket.headers ?? {}, body: file });
    } catch (e) {
      throw new ApiError(`Upload failed: ${(e as Error).message}`, 0, 'network');
    }
    if (!res.ok) throw new ApiError(`Upload failed (HTTP ${res.status})`, res.status, 'upload_failed');
  }

  createJob(body: JobCreate): Promise<JobStatus> {
    return this.json('POST', '/v1/jobs', body);
  }

  getJob(jobId: string): Promise<JobStatus> {
    return this.json('GET', `/v1/jobs/${encodeURIComponent(jobId)}`);
  }

  async approve(jobId: string, decision: ApprovalDecision): Promise<JobStatus> {
    const res = await this.send('POST', `/v1/jobs/${encodeURIComponent(jobId)}/approval`, decision);
    try {
      const body = (await res.json()) as JobStatus;
      if (body && typeof body === 'object' && 'status' in body) return body;
    } catch {
      /* fall through: service may answer 202/204 without a JobStatus body */
    }
    return this.getJob(jobId);
  }

  async getTrace(jobId: string): Promise<TraceFetch | null> {
    let res: Response;
    try {
      res = await this.fetchFn(`${this.apiBase}/v1/jobs/${encodeURIComponent(jobId)}/trace`, {
        headers: { Accept: 'application/x-ndjson, application/jsonl, text/plain' },
      });
    } catch (e) {
      throw new ApiError(`Network error fetching trace: ${(e as Error).message}`, 0, 'network');
    }
    if (res.status === 404) return null;
    if (!res.ok) throw await toApiError(res);
    const parsed = parseTraceJsonl(await res.text());
    return { ...parsed, synthetic: isSyntheticTrace(parsed.events) };
  }

  artifactUrl(jobId: string, artifact: Artifact): string {
    return resolveArtifactUrl(this.apiBase, jobId, artifact.uri);
  }
}

async function toApiError(res: Response): Promise<ApiError> {
  let msg = `HTTP ${res.status}`;
  let code: string | undefined;
  try {
    const body = (await res.json()) as Record<string, unknown>;
    const detail = body['detail'] ?? body['message'] ?? body['error'];
    if (typeof detail === 'string') msg = detail;
    else if (detail && typeof detail === 'object') {
      const d = detail as Record<string, unknown>;
      if (typeof d['message'] === 'string') msg = d['message'];
      if (typeof d['code'] === 'string') code = d['code'];
      if (Array.isArray(detail)) msg = `Request rejected by the API (${detail.length} validation error(s))`;
    }
    if (typeof body['code'] === 'string') code = body['code'];
  } catch {
    /* non-JSON error body */
  }
  return new ApiError(msg, res.status, code);
}
