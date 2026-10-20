import type { ApprovalDecision, Artifact, JobCreate, JobStatus, TraceEvent } from '../types/contracts';
import type { TraceParseError } from '../lib/trace';

/** GET /v1/version */
export interface VersionInfo {
  git_sha: string;
  contract_version: string;
  service_version: string;
  cv2_version?: string | null;
  cv2_file?: string | null;
  python: string;
  arch: string;
}

/** GET /v1/demos item */
export interface DemoSite {
  id: string;
  title: string;
  family: string;
  difficulty: string;
  description: string;
}

/** POST /v1/uploads body */
export interface UploadRequest {
  filename: string;
  content_type: string;
  size_bytes: number;
}

/** POST /v1/uploads response */
export interface UploadTicket {
  upload_id: string;
  key: string;
  url: string;
  headers: Record<string, string>;
  expires_in: number;
}

export interface TraceFetch {
  events: TraceEvent[];
  errors: TraceParseError[];
  /** True when the events are a committed synthetic sample (demo mode), not a real agent run. */
  synthetic: boolean;
}

export interface Api {
  readonly kind: 'live' | 'mock';
  readonly apiBase: string;
  healthz(): Promise<boolean>;
  version(): Promise<VersionInfo>;
  demos(): Promise<DemoSite[]>;
  createUpload(req: UploadRequest): Promise<UploadTicket>;
  putUpload(ticket: UploadTicket, file: Blob): Promise<void>;
  createJob(body: JobCreate): Promise<JobStatus>;
  getJob(jobId: string): Promise<JobStatus>;
  approve(jobId: string, decision: ApprovalDecision): Promise<JobStatus>;
  /** null when the job has no trace (yet). */
  getTrace(jobId: string): Promise<TraceFetch | null>;
  artifactUrl(jobId: string, artifact: Artifact): string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export function newClientToken(): string {
  const c = globalThis.crypto;
  if (c && typeof c.randomUUID === 'function') return c.randomUUID();
  const b = new Uint8Array(16);
  if (c && typeof c.getRandomValues === 'function') c.getRandomValues(b);
  else for (let i = 0; i < 16; i++) b[i] = Math.floor(Math.random() * 256);
  b[6] = (b[6]! & 0x0f) | 0x40;
  b[8] = (b[8]! & 0x3f) | 0x80;
  const h = [...b].map((x) => x.toString(16).padStart(2, '0')).join('');
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}
