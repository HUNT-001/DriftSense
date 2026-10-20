/**
 * In-browser mock of the job API ("demo mode"). Same interface as HttpApi, deterministic,
 * simulated latency. Every number it returns is SYNTHETIC (engine "fake") and the UI labels it so.
 */
import type {
  ApprovalDecision,
  Artifact,
  Candidate,
  JobCreate,
  JobStatus,
  LocalizeResult,
  TraceEvent,
} from '../types/contracts';
import { parseTraceJsonl } from '../lib/trace';
import { ALLOWED_CONTENT_TYPES, MAX_UPLOAD_BYTES } from '../lib/upload';
import sampleCommit from '../fixtures/sample_trace.jsonl?raw';
import sampleEscalate from '../fixtures/sample_trace_escalate.jsonl?raw';
import { candidatesSvg, heatmapSvg, overlaySvg } from './mockImages';
import { ApiError, type Api, type DemoSite, type TraceFetch, type UploadRequest, type UploadTicket, type VersionInfo } from './types';

export const MOCK_QUEUED_MS = 700;
export const MOCK_RUN_MS = 1500;
export const MOCK_EVENT_MS = 450;
const TAU = 0.22;
const PIXEL_NM = 1.25;

export const MOCK_DEMOS: DemoSite[] = [
  {
    id: 'demo_finfet_easy',
    title: 'FinFET lines (clean)',
    family: 'finfet',
    difficulty: 'easy',
    description: 'Periodic fin lines with visible edge roughness.',
  },
  {
    id: 'demo_dram_hard',
    title: 'DRAM contacts (noisy)',
    family: 'dram',
    difficulty: 'hard',
    description: 'Dense contact holes at low signal; the agent may need to ask an operator.',
  },
  {
    id: 'demo_grid',
    title: 'Square grid',
    family: 'grid',
    difficulty: 'medium',
    description: 'Two-dimensional square lattice of identical features.',
  },
  {
    id: 'demo_staggered',
    title: 'Staggered array',
    family: 'staggered',
    difficulty: 'medium',
    description: 'Offset rows: neighbouring copies sit half a period apart.',
  },
];

type TraceName = 'commit' | 'escalate';

const TRACES: Record<TraceName, TraceEvent[]> = {
  commit: parseTraceJsonl(sampleCommit).events,
  escalate: parseTraceJsonl(sampleEscalate).events,
};

interface Scenario {
  candidates: Candidate[];
  margin: number;
  lattice: LocalizeResult['lattice_type'];
  trace: TraceName | null;
}

function cands(list: [number, number, number][]): Candidate[] {
  return list.map(([x, y, s], i) => ({ rank: i + 1, x_px: x, y_px: y, score: s }));
}

function scenarioFor(body: JobCreate): Scenario {
  const agent = !!body.agent;
  switch (body.demo_id) {
    case 'demo_dram_hard':
      return agent
        ? {
            candidates: cands([[150.4, 88.1, 0.71], [214.0, 88.3, 0.69], [182.2, 120.0, 0.64]]),
            margin: 0.15,
            lattice: 'dram',
            trace: 'escalate',
          }
        : {
            candidates: cands([[150.3, 88.0, 0.62], [182.1, 104.0, 0.61], [214.0, 88.2, 0.6]]),
            margin: 0.08,
            lattice: 'dram',
            trace: null,
          };
    case 'demo_grid':
      return {
        candidates: cands([[96.0, 64.2, 0.8], [128.1, 64.0, 0.5], [96.2, 96.1, 0.47]]),
        margin: 0.36,
        lattice: 'grid',
        trace: agent ? 'commit' : null,
      };
    case 'demo_staggered':
      return {
        candidates: cands([[260.4, 120.3, 0.77], [276.5, 152.2, 0.55], [244.2, 152.0, 0.52]]),
        margin: 0.27,
        lattice: 'staggered',
        trace: agent ? 'commit' : null,
      };
    case 'demo_finfet_easy':
    default:
      return agent
        ? {
            candidates: cands([[212.5, 97.2, 0.81], [372.9, 97.4, 0.55], [180.5, 97.2, 0.53]]),
            margin: 0.31,
            lattice: body.demo_id ? 'finfet' : 'unknown',
            trace: 'commit',
          }
        : {
            candidates: cands([[212.5, 97.2, 0.83], [244.4, 97.0, 0.52], [180.5, 97.1, 0.49]]),
            margin: 0.41,
            lattice: body.demo_id ? 'finfet' : 'unknown',
            trace: null,
          };
  }
}

const MOCK_ARTIFACTS: Artifact[] = [
  { kind: 'heatmap', uri: 'heatmap.svg' },
  { kind: 'overlay', uri: 'overlay.svg' },
  { kind: 'candidates', uri: 'candidates.svg' },
];

function buildResult(body: JobCreate, s: Scenario, chosenRank: number | null): LocalizeResult {
  const pick = s.candidates.find((c) => c.rank === chosenRank) ?? s.candidates[0]!;
  const timings = [
    ['ingest', 2.0],
    ['lattice', 6.0],
    ['rectify', 4.0],
    ['roughness', 9.0],
    ['score', 15.0],
    ['confidence', 1.0],
    ['subpixel', 5.0],
    ['explain', 8.0],
  ].map(([stage, ms]) => ({ stage: stage as string, ms: ms as number }));
  return {
    contract_version: '1.0.0',
    engine: 'fake',
    engine_version: 'mock-0.0.0',
    mode: body.mode ?? 'auto',
    lattice_type: s.lattice,
    offset: {
      x_px: pick.x_px,
      y_px: pick.y_px,
      x_nm: +(pick.x_px * PIXEL_NM).toFixed(2),
      y_nm: +(pick.y_px * PIXEL_NM).toFixed(2),
      pixel_size_nm: PIXEL_NM,
    },
    candidates: s.candidates as LocalizeResult['candidates'],
    margin: {
      margin: s.margin,
      tau: TAU,
      calibrated: true,
      p_correct: null,
      decision: s.margin >= TAU ? 'commit' : 'escalate',
    },
    timings,
    total_ms: timings.reduce((a, t) => a + t.ms, 0),
    artifacts: MOCK_ARTIFACTS,
    warnings: ['Demo mode: synthetic result from the in-browser mock API (engine "fake").'],
  };
}

interface MockJob {
  id: string;
  created: number;
  body: JobCreate;
  scenario: Scenario;
  decision?: ApprovalDecision;
  decidedAt?: number;
}

export interface MockOptions {
  /** Simulated network latency per call (ms). */
  latencyMs?: number;
  now?: () => number;
  sleep?: (ms: number) => Promise<void>;
}

export class MockApi implements Api {
  readonly kind = 'mock' as const;
  readonly apiBase = '';
  private readonly jobs = new Map<string, MockJob>();
  private readonly tokens = new Map<string, string>();
  private seq = 0;
  private uploads = 0;
  private readonly latency: number;
  private readonly now: () => number;
  private readonly sleep: (ms: number) => Promise<void>;

  constructor(opts: MockOptions = {}) {
    this.latency = opts.latencyMs ?? 250;
    this.now = opts.now ?? (() => Date.now());
    this.sleep = opts.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)));
  }

  private async lag<T>(v: T): Promise<T> {
    if (this.latency > 0) await this.sleep(this.latency);
    return structuredClone(v);
  }

  async healthz(): Promise<boolean> {
    return this.lag(true);
  }

  version(): Promise<VersionInfo> {
    return this.lag({
      git_sha: 'demo-mode',
      contract_version: '1.0.0',
      service_version: 'mock',
      cv2_version: null,
      cv2_file: null,
      python: 'n/a (browser)',
      arch: 'browser',
    });
  }

  demos(): Promise<DemoSite[]> {
    return this.lag(MOCK_DEMOS);
  }

  async createUpload(req: UploadRequest): Promise<UploadTicket> {
    if (!(ALLOWED_CONTENT_TYPES as readonly string[]).includes(req.content_type)) {
      throw new ApiError('Only image/png and image/tiff are accepted', 415, 'unsupported_media_type');
    }
    if (req.size_bytes <= 0 || req.size_bytes > MAX_UPLOAD_BYTES) {
      throw new ApiError('File must be between 1 byte and 20 MB', 413, 'too_large');
    }
    const n = ++this.uploads;
    const id = `mock-upload-${String(n).padStart(4, '0')}`;
    return this.lag({
      upload_id: id,
      key: `uploads/${id}/${req.filename}`,
      url: `mock://uploads/${id}`,
      headers: { 'Content-Type': req.content_type },
      expires_in: 900,
    });
  }

  async putUpload(_ticket: UploadTicket, _file: Blob): Promise<void> {
    await this.lag(null);
  }

  async createJob(body: JobCreate): Promise<JobStatus> {
    const hasKeys = !!(body.template_key && body.search_key);
    if (!body.demo_id && !hasKeys) {
      throw new ApiError('provide demo_id or both template_key and search_key', 422, 'validation');
    }
    if (body.demo_id && !MOCK_DEMOS.some((d) => d.id === body.demo_id)) {
      throw new ApiError(`unknown demo_id "${body.demo_id}"`, 422, 'validation');
    }
    if (body.client_token && this.tokens.has(body.client_token)) {
      return this.getJob(this.tokens.get(body.client_token)!);
    }
    const id = `mock-job-${String(++this.seq).padStart(4, '0')}`;
    const job: MockJob = { id, created: this.now(), body, scenario: scenarioFor(body) };
    this.jobs.set(id, job);
    if (body.client_token) this.tokens.set(body.client_token, id);
    return this.lag(this.snapshot(job));
  }

  async getJob(jobId: string): Promise<JobStatus> {
    const job = this.jobs.get(jobId);
    if (!job) {
      await this.lag(null);
      throw new ApiError(`job ${jobId} not found (demo-mode jobs live only in this browser tab)`, 404, 'not_found');
    }
    return this.lag(this.snapshot(job));
  }

  async approve(jobId: string, decision: ApprovalDecision): Promise<JobStatus> {
    const job = this.jobs.get(jobId);
    if (!job) throw new ApiError(`job ${jobId} not found`, 404, 'not_found');
    if (this.snapshot(job).status !== 'awaiting_approval') {
      throw new ApiError('job is not awaiting approval', 409, 'conflict');
    }
    if (!decision.approver?.trim()) throw new ApiError('approver is required', 422, 'validation');
    const rank = decision.chosen_rank ?? null;
    if (rank !== null && !job.scenario.candidates.some((c) => c.rank === rank)) {
      throw new ApiError(`chosen_rank ${rank} is not one of the offered options`, 422, 'validation');
    }
    job.decision = { ...decision, chosen_rank: rank };
    job.decidedAt = this.now();
    return this.lag(this.snapshot(job));
  }

  async getTrace(jobId: string): Promise<TraceFetch | null> {
    const job = this.jobs.get(jobId);
    if (!job || !job.scenario.trace) return this.lag(null);
    const events = this.visibleEvents(job);
    if (events === null) return this.lag(null);
    return this.lag({ events, errors: [], synthetic: true });
  }

  artifactUrl(jobId: string, artifact: Artifact): string {
    const job = this.jobs.get(jobId);
    const c = job?.scenario.candidates ?? [];
    if (artifact.kind === 'heatmap') return heatmapSvg(c);
    if (artifact.kind === 'overlay') return overlaySvg(c);
    return candidatesSvg(c);
  }

  // ---------- internals ----------

  /** Events in the agent's trace up to (and including) the escalation, if any. */
  private preApprovalEvents(name: TraceName): TraceEvent[] {
    const all = TRACES[name];
    const esc = all.findIndex((e) => e.kind === 'escalate');
    return esc >= 0 ? all.slice(0, esc + 1) : all;
  }

  private visibleEvents(job: MockJob): TraceEvent[] | null {
    const name = job.scenario.trace;
    if (!name) return null;
    const t = this.now() - job.created;
    if (t < MOCK_QUEUED_MS) return null;
    const pre = this.preApprovalEvents(name);
    const shown = Math.min(pre.length, Math.floor((t - MOCK_QUEUED_MS) / MOCK_EVENT_MS) + 1);
    const events = pre.slice(0, shown);
    if (job.decision && shown === pre.length) events.push(this.approvalEvent(name, job.decision));
    return events;
  }

  private approvalEvent(name: TraceName, d: ApprovalDecision): TraceEvent {
    const template = TRACES[name].find((e) => e.kind === 'human_approval');
    const last = this.preApprovalEvents(name).at(-1)!;
    const rank = d.chosen_rank ?? null;
    return {
      ...(template ?? last),
      kind: 'human_approval',
      step: last.step,
      tool: rank === null ? null : 'stage.move',
      args: { chosen_rank: rank },
      result_summary: {
        synthetic: true,
        chosen_rank: rank,
        approver: d.approver,
        ...(d.note ? { note: d.note } : {}),
        dose: last.result_summary?.['dose'] ?? 0,
      },
      because: rank === null ? 'operator rejected all candidates: no stage move' : `operator approved candidate #${rank}`,
      stage_move_nm: rank === null ? null : (template?.stage_move_nm ?? null),
      latency_ms: null,
    };
  }

  private snapshot(job: MockJob): JobStatus {
    const t = this.now() - job.created;
    const iso = (ms: number) => new Date(job.created + ms).toISOString();
    const base: JobStatus = {
      job_id: job.id,
      status: 'queued',
      created_at: iso(0),
      updated_at: iso(0),
      engine: 'fake',
      result: null,
      trace_key: null,
      approval: null,
      error: null,
      worker: null,
    };
    if (t < MOCK_QUEUED_MS) return base;
    const worker = {
      instance_type: 'in-browser mock',
      arch: 'unknown' as const,
      opencv_version: 'n/a (mock)',
      numpy_version: 'n/a (mock)',
      cool: false,
    };
    const s = job.scenario;
    const running: JobStatus = { ...base, status: 'running', updated_at: iso(MOCK_QUEUED_MS), worker };
    if (s.trace) running.trace_key = `traces/${job.id}.jsonl`;
    let doneAt: number;
    if (s.trace) {
      doneAt = MOCK_QUEUED_MS + (this.preApprovalEvents(s.trace).length - 1) * MOCK_EVENT_MS;
    } else doneAt = MOCK_QUEUED_MS + MOCK_RUN_MS;
    if (t < doneAt) return running;

    const result = buildResult(job.body, s, job.decision?.chosen_rank ?? null);
    if (s.trace === 'escalate') {
      if (!job.decision) {
        return {
          ...running,
          status: 'awaiting_approval',
          updated_at: iso(doneAt),
          result,
          approval: {
            options: s.candidates.slice(0, 3) as NonNullable<JobStatus['approval']>['options'],
            margin: result.margin,
          },
        };
      }
      const decidedMs = (job.decidedAt ?? this.now()) - job.created;
      if (job.decision.chosen_rank === null || job.decision.chosen_rank === undefined) {
        return {
          ...running,
          status: 'failed',
          updated_at: iso(decidedMs),
          result,
          error: { code: 'rejected_by_operator', message: 'The operator rejected all candidates; no stage move was made.' },
        };
      }
      return { ...running, status: 'succeeded', updated_at: iso(decidedMs), result };
    }
    return { ...running, status: 'succeeded', updated_at: iso(doneAt), result };
  }
}
