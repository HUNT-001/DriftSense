import type { ApprovalDecision, Artifact, JobCreate, JobStatus } from '../types/contracts';
import { resolveArtifactUrl } from '../lib/artifacts';
import { parseTraceJsonl } from '../lib/trace';
import {
  ApiError,
  type Api,
  type DemoInfo,
  type EpisodeCreate,
  type TraceFetch,
  type UploadRequest,
  type UploadTicket,
  type VersionInfo,
} from './types';

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
    // no-store: job status changes between polls; never serve it from the HTTP cache.
    const init: RequestInit = { method, headers: { Accept: 'application/json' }, cache: 'no-store' };
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

  /** GET /healthz with a hard timeout (used at boot to decide on offline replay). */
  async healthz(timeoutMs = 4000): Promise<boolean> {
    const ctrl = typeof AbortController === 'function' ? new AbortController() : null;
    const t = ctrl ? setTimeout(() => ctrl.abort(), timeoutMs) : undefined;
    try {
      const res = await this.fetchFn(`${this.apiBase}/healthz`, {
        headers: { Accept: 'application/json' },
        ...(ctrl ? { signal: ctrl.signal } : {}),
      });
      return res.ok;
    } catch {
      return false;
    } finally {
      clearTimeout(t);
    }
  }

  version(): Promise<VersionInfo> {
    return this.json('GET', '/v1/version');
  }

  demos(): Promise<DemoInfo[]> {
    return this.json('GET', '/v1/demos');
  }

  createUpload(req: UploadRequest): Promise<UploadTicket> {
    return this.json('POST', '/v1/uploads', req);
  }

  async putUpload(ticket: UploadTicket, file: Blob): Promise<void> {
    let res: Response;
    try {
      // Send exactly the signed headers the service returned (Content-Type + Content-Length).
      res = await this.fetchFn(ticket.url, { method: ticket.method ?? 'PUT', headers: { ...(ticket.headers ?? {}) }, body: file });
    } catch (e) {
      throw new ApiError(`Upload failed: ${(e as Error).message}`, 0, 'network');
    }
    if (!res.ok) throw new ApiError(`Upload failed (HTTP ${res.status})`, res.status, 'upload_failed');
  }

  createJob(body: JobCreate): Promise<JobStatus> {
    return this.json('POST', '/v1/jobs', body);
  }

  /** POST /v1/agent/episodes -> 202 JobStatus. */
  createEpisode(body: EpisodeCreate): Promise<JobStatus> {
    return this.json('POST', '/v1/agent/episodes', body);
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
        cache: 'no-store',
      });
    } catch (e) {
      throw new ApiError(`Network error fetching trace: ${(e as Error).message}`, 0, 'network');
    }
    // 404 not_ready: the agent has not written its first event yet (the trace grows while running).
    if (res.status === 404) return null;
    if (!res.ok) throw await toApiError(res);
    const parsed = parseTraceJsonl(await res.text());
    return { ...parsed, source: 'live' };
  }

  async getArtifactJson(jobId: string, name: string): Promise<unknown | null> {
    let res: Response;
    try {
      res = await this.fetchFn(
        `${this.apiBase}/v1/jobs/${encodeURIComponent(jobId)}/artifacts/${encodeURIComponent(name)}`,
        { headers: { Accept: 'application/json' } },
      );
    } catch (e) {
      throw new ApiError(`Network error fetching ${name}: ${(e as Error).message}`, 0, 'network');
    }
    if (res.status === 404) return null;
    if (!res.ok) throw await toApiError(res);
    try {
      return (await res.json()) as unknown;
    } catch {
      throw new ApiError(`Malformed JSON in ${name}`, res.status, 'bad_json');
    }
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
