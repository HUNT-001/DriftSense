import type { ApprovalDecision, Artifact, JobCreate, JobStatus, TraceEvent } from '../types/contracts';
import type { TraceParseError } from '../lib/trace';

/** GET /v1/version (service/models.py VersionInfo). Every field is optional on our side so an
 * older or newer service never breaks the footer. */
export interface VersionInfo {
  service_version?: string;
  git_sha?: string;
  contract_version?: string;
  engine?: string | null;
  opencv_version?: string | null;
  opencv_file?: string | null;
  numpy_version?: string | null;
  python_version?: string;
  arch?: string;
  platform?: string;
}

/** GET /v1/demos item (service/models.py DemoInfo). Agent-episode demos carry an extra flag whose
 * exact name the service had not fixed yet; see `isEpisodeDemo`. */
export interface DemoInfo {
  demo_id: string;
  title: string;
  lattice: string;
  difficulty: 'easy' | 'medium' | 'hard' | string;
  description?: string;
  /** Optional extras an episode demo may carry. */
  agent?: boolean;
  episode?: boolean;
  kind?: string;
  family?: string;
  seed?: number;
}

export type EpisodeFamily = 'dram' | 'finfet' | 'random';

/** POST /v1/agent/episodes body. */
export interface EpisodeCreate {
  family: EpisodeFamily;
  seed: number;
  client_token?: string | null;
}

/** POST /v1/uploads body (service/models.py UploadRequest). */
export interface UploadRequest {
  filename: string;
  content_type: string;
  size_bytes: number;
  role?: 'template' | 'search';
  /** Reuse the first upload's id so template and search share one prefix. */
  upload_id?: string;
}

/** POST /v1/uploads response (service/models.py UploadResponse). */
export interface UploadTicket {
  upload_id: string;
  key: string;
  url: string;
  method?: 'PUT';
  /** The PUT must send exactly these (Content-Type and Content-Length are signed). */
  headers: Record<string, string>;
  expires_in: number;
  max_bytes?: number;
}

export type TraceSource = 'live' | 'replay';

export interface TraceFetch {
  events: TraceEvent[];
  errors: TraceParseError[];
  /** `replay` = offline replay of a recorded trace (no API involved). */
  source: TraceSource;
}

export interface Api {
  readonly kind: 'live' | 'replay';
  readonly apiBase: string;
  healthz(): Promise<boolean>;
  version(): Promise<VersionInfo>;
  demos(): Promise<DemoInfo[]>;
  createUpload(req: UploadRequest): Promise<UploadTicket>;
  putUpload(ticket: UploadTicket, file: Blob): Promise<void>;
  createJob(body: JobCreate): Promise<JobStatus>;
  createEpisode(body: EpisodeCreate): Promise<JobStatus>;
  getJob(jobId: string): Promise<JobStatus>;
  approve(jobId: string, decision: ApprovalDecision): Promise<JobStatus>;
  /** null while the trace does not exist yet (404 not_ready). */
  getTrace(jobId: string): Promise<TraceFetch | null>;
  /** GET /v1/jobs/{id}/artifacts/{name} parsed as JSON; null when absent (404). */
  getArtifactJson(jobId: string, name: string): Promise<unknown | null>;
  artifactUrl(jobId: string, artifact: Artifact): string;
  /** Engine-provided alt text for an artifact, when the source has it (offline samples do). */
  artifactAltText?(jobId: string, artifact: Artifact): string | undefined;
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

/** Agent-episode demos: an explicit flag if the service sends one, else the `vsem-` id prefix. */
export function isEpisodeDemo(d: DemoInfo): boolean {
  if (d.agent === true || d.episode === true) return true;
  if (typeof d.kind === 'string' && /episode|agent/i.test(d.kind)) return true;
  return d.demo_id.startsWith('vsem-');
}

/** Family + seed for an episode demo: explicit fields, else parsed from `vsem-<family>-<seed>`. */
export function episodeParams(d: DemoInfo): EpisodeCreate {
  const m = /^vsem-([a-z]+)-(\d+)$/i.exec(d.demo_id);
  const fam = (d.family ?? m?.[1] ?? d.lattice ?? 'random').toLowerCase();
  const family: EpisodeFamily = fam === 'dram' || fam === 'finfet' ? fam : 'random';
  const seed = Number.isInteger(d.seed) ? (d.seed as number) : m ? Number(m[2]) : 0;
  return { family, seed };
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

/** Job ids the user started as agent episodes in this tab (JobStatus does not echo `agent`). */
const EPISODE_KEY = 'driftsense.episodes';
const episodeIds = new Set<string>();

export function rememberEpisode(jobId: string): void {
  episodeIds.add(jobId);
  try {
    const prev = JSON.parse(sessionStorage.getItem(EPISODE_KEY) ?? '[]') as string[];
    sessionStorage.setItem(EPISODE_KEY, JSON.stringify([...new Set([...prev, jobId])].slice(-50)));
  } catch {
    /* storage unavailable */
  }
}

export function isKnownEpisode(jobId: string): boolean {
  if (episodeIds.has(jobId)) return true;
  try {
    return (JSON.parse(sessionStorage.getItem(EPISODE_KEY) ?? '[]') as string[]).includes(jobId);
  } catch {
    return false;
  }
}
