/**
 * Offline replay ("judge mode"): an in-browser stand-in for the job API that replays the REAL
 * committed agent traces (traces/demo_commit.jsonl, traces/demo_escalate.jsonl, copied to
 * public/samples/) with realistic pacing, plus the engine outputs re-captured for those exact
 * episodes (public/samples/*.capture.json: LocalizeResult, ApprovalRequest, explain images).
 *
 * Nothing here is invented except the operator step: in the recording the escalation was answered
 * by `auto:none`; in replay the person at the keyboard answers it. Those events are marked
 * `result_summary.simulated = true` and the UI labels them.
 */
import type {
  ApprovalDecision,
  ApprovalRequest,
  Artifact,
  JobCreate,
  JobStatus,
  LocalizeResult,
  TraceEvent,
} from '../types/contracts';
import { parseTraceJsonl, summarizeTrace } from '../lib/trace';
import { artifactBasename } from '../lib/artifacts';
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

export const REPLAY_NAMES = ['demo_commit', 'demo_escalate'] as const;
export type ReplayName = (typeof REPLAY_NAMES)[number];

/** Time from "submit" to the first event (the job sits in `queued`). */
export const REPLAY_QUEUED_MS = 800;
/** Base pause per trace step so a person can follow along; tool latency is added on top. */
export const REPLAY_STEP_MS = 550;
export const REPLAY_STEP_MIN_MS = 400;
export const REPLAY_STEP_MAX_MS = 1600;

export interface CaptureEntry {
  trace_step: number;
  result: LocalizeResult;
  /** Artifact.uri (e.g. `ncc_heatmap.png`) -> path under samples/. */
  artifact_files: Record<string, string>;
  artifact_alt: Record<string, string>;
}

export interface Capture {
  trace: string;
  seed: number;
  family: string;
  localize: CaptureEntry[];
  approval_request: ApprovalRequest | null;
}

export interface Recording {
  name: ReplayName;
  events: TraceEvent[];
  capture: Capture;
}

/** Loads a file from public/samples/ (text). Tests inject a filesystem loader. */
export type SampleLoader = (path: string) => Promise<string>;

export const fetchSampleLoader =
  (base = 'samples/'): SampleLoader =>
  async (path) => {
    const res = await fetch(`${base}${path}`, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`could not load ${base}${path} (HTTP ${res.status})`);
    return res.text();
  };

export async function loadRecordings(load: SampleLoader): Promise<Recording[]> {
  return Promise.all(
    REPLAY_NAMES.map(async (name) => {
      const [jsonl, cap] = await Promise.all([load(`${name}.jsonl`), load(`${name}.capture.json`)]);
      const parsed = parseTraceJsonl(jsonl);
      if (parsed.errors.length) throw new Error(`${name}.jsonl: ${parsed.errors[0]!.message}`);
      return { name, events: parsed.events, capture: JSON.parse(cap) as Capture };
    }),
  );
}

/** Pause before event i is revealed. */
export function stepPause(ev: TraceEvent): number {
  const lat = typeof ev.latency_ms === 'number' ? ev.latency_ms : 0;
  return Math.max(REPLAY_STEP_MIN_MS, Math.min(REPLAY_STEP_MAX_MS, REPLAY_STEP_MS + lat * 1.5));
}

/** Cumulative reveal times (ms after the job left `queued`) for a list of events. */
export function revealTimes(events: TraceEvent[]): number[] {
  let t = 0;
  return events.map((ev, i) => (i === 0 ? 0 : (t += stepPause(ev))));
}

export function demoIdFor(r: Recording): string {
  return `vsem-${r.capture.family}-${r.capture.seed}`;
}

interface ReplayJob {
  id: string;
  rec: Recording;
  agent: boolean;
  created: number;
  requested?: EpisodeCreate;
  decision?: ApprovalDecision;
  decidedAt?: number;
}

export interface ReplayOptions {
  loader?: SampleLoader;
  now?: () => number;
  /** Simulated network latency per call (ms). */
  latencyMs?: number;
  /** Base URL for sample images (relative to the page). */
  sampleBase?: string;
}

export class ReplayApi implements Api {
  readonly kind = 'replay' as const;
  readonly apiBase = '';
  private readonly jobs = new Map<string, ReplayJob>();
  private readonly tokens = new Map<string, string>();
  private seq = 0;
  private readonly now: () => number;
  private readonly latency: number;
  private readonly sampleBase: string;
  private recs: Promise<Recording[]> | null = null;
  private readonly loader: SampleLoader;

  constructor(opts: ReplayOptions = {}) {
    this.sampleBase = opts.sampleBase ?? 'samples/';
    this.loader = opts.loader ?? fetchSampleLoader(this.sampleBase);
    this.now = opts.now ?? (() => Date.now());
    this.latency = opts.latencyMs ?? 120;
  }

  recordings(): Promise<Recording[]> {
    this.recs ??= loadRecordings(this.loader);
    return this.recs;
  }

  private async lag<T>(v: T): Promise<T> {
    if (this.latency > 0) await new Promise((r) => setTimeout(r, this.latency));
    return structuredClone(v);
  }

  async healthz(): Promise<boolean> {
    return true;
  }

  async version(): Promise<VersionInfo> {
    const recs = await this.recordings();
    const r = recs[0]?.capture.localize[0]?.result;
    return this.lag({
      service_version: 'offline-replay',
      git_sha: 'recorded',
      contract_version: r?.contract_version ?? '1.0.0',
      engine: r ? `${r.engine} ${r.engine_version}` : null,
      opencv_version: null,
      opencv_file: null,
      numpy_version: null,
      python_version: 'n/a (browser)',
      arch: 'browser',
      platform: 'offline replay of a recorded trace',
    });
  }

  async demos(): Promise<DemoInfo[]> {
    const recs = await this.recordings();
    return this.lag(
      recs.map((r) => {
        const s = summarizeTrace(r.events);
        const commit = s.outcome === 'commit';
        return {
          demo_id: demoIdFor(r),
          title: `Recorded episode: ${r.capture.family.toUpperCase()} seed ${r.capture.seed} (${commit ? 'commits' : 'escalates'})`,
          lattice: r.capture.family,
          difficulty: commit ? 'medium' : 'hard',
          description: commit
            ? 'The uniqueness margin clears τ, so the agent commits the corrective stage move by itself.'
            : 'The margin stays below τ when the step budget runs out, so the agent asks you to choose.',
          agent: true,
          family: r.capture.family,
          seed: r.capture.seed,
        };
      }),
    );
  }

  async createUpload(_req: UploadRequest): Promise<UploadTicket> {
    throw new ApiError('Uploads need the live API; offline replay only plays recorded episodes.', 503, 'offline');
  }

  async putUpload(): Promise<void> {
    throw new ApiError('Uploads need the live API.', 503, 'offline');
  }

  private async pick(family?: string, seed?: number): Promise<Recording> {
    const recs = await this.recordings();
    return (
      recs.find((r) => r.capture.family === family && r.capture.seed === seed) ??
      recs.find((r) => r.name === 'demo_escalate') ??
      recs[0]!
    );
  }

  private register(rec: Recording, agent: boolean, token?: string | null, requested?: EpisodeCreate): ReplayJob {
    if (token && this.tokens.has(token)) return this.jobs.get(this.tokens.get(token)!)!;
    const id = `replay-${rec.name.replace('demo_', '')}-${String(++this.seq).padStart(3, '0')}`;
    const job: ReplayJob = { id, rec, agent, created: this.now(), ...(requested ? { requested } : {}) };
    this.jobs.set(id, job);
    if (token) this.tokens.set(token, id);
    return job;
  }

  async createJob(body: JobCreate): Promise<JobStatus> {
    if (!body.demo_id) {
      throw new ApiError('Offline replay can only run the recorded demo episodes.', 422, 'validation_error');
    }
    const recs = await this.recordings();
    const rec = recs.find((r) => demoIdFor(r) === body.demo_id);
    if (!rec) throw new ApiError(`unknown demo_id "${body.demo_id}"`, 422, 'validation_error');
    return this.lag(this.snapshot(this.register(rec, !!body.agent, body.client_token)));
  }

  async createEpisode(body: EpisodeCreate): Promise<JobStatus> {
    const rec = await this.pick(body.family, body.seed);
    return this.lag(this.snapshot(this.register(rec, true, body.client_token, body)));
  }

  /** Which recording a replay job plays and what was asked for (for the "offline" notice). */
  describeJob(jobId: string): { recording: ReplayName; demoId: string; requested?: EpisodeCreate } | null {
    const j = this.jobs.get(jobId);
    if (!j) return null;
    return { recording: j.rec.name, demoId: demoIdFor(j.rec), ...(j.requested ? { requested: j.requested } : {}) };
  }

  async getJob(jobId: string): Promise<JobStatus> {
    const job = this.jobs.get(jobId);
    if (!job) {
      throw new ApiError(`job ${jobId} not found (offline replay jobs live only in this browser tab)`, 404, 'not_found');
    }
    return this.lag(this.snapshot(job));
  }

  async approve(jobId: string, decision: ApprovalDecision): Promise<JobStatus> {
    const job = this.jobs.get(jobId);
    if (!job) throw new ApiError(`job ${jobId} not found`, 404, 'not_found');
    if (this.snapshot(job).status !== 'awaiting_approval') {
      throw new ApiError('job is not awaiting approval', 409, 'invalid_state');
    }
    if (!decision.approver?.trim()) throw new ApiError('approver is required', 422, 'validation_error');
    const rank = decision.chosen_rank ?? null;
    const opts = job.rec.capture.approval_request?.options ?? [];
    if (rank !== null && !opts.some((c) => c.rank === rank)) {
      throw new ApiError(`chosen_rank ${rank} is not one of the offered options`, 422, 'validation_error');
    }
    job.decision = { ...decision, chosen_rank: rank };
    job.decidedAt = this.now();
    return this.lag(this.snapshot(job));
  }

  async getTrace(jobId: string): Promise<TraceFetch | null> {
    const job = this.jobs.get(jobId);
    if (!job || !job.agent) return this.lag(null);
    const events = this.visibleEvents(job);
    if (!events) return this.lag(null);
    return this.lag({ events, errors: [], source: 'replay' as const });
  }

  async getArtifactJson(jobId: string, name: string): Promise<unknown | null> {
    const job = this.jobs.get(jobId);
    if (!job) return null;
    const snap = this.snapshot(job);
    if (name === 'result.json') return snap.result ?? null;
    if (name === 'episode.json' && job.agent && (snap.status === 'succeeded' || snap.status === 'failed')) {
      const events = this.visibleEvents(job) ?? [];
      const s = summarizeTrace(events);
      return this.lag({
        run_id: s.runId,
        outcome: s.outcome,
        steps: events.length,
        tau: s.tau,
        final_margin: s.margin,
        dose: s.dose,
        n_acq: s.acquisitions,
        policy: s.policy,
        operator_choice: job.decision ? (job.decision.chosen_rank ?? null) : undefined,
        source: 'derived in the browser from the recorded trace (offline replay)',
      });
    }
    return null;
  }

  private entry(job: ReplayJob): CaptureEntry | undefined {
    const c = job.rec.capture.localize;
    const decisive = summarizeTrace(job.rec.events).decisive;
    // The localize whose margin drove the decision (the last captured one before it).
    return [...c].reverse().find((e) => !decisive || e.trace_step < decisive.step) ?? c[0];
  }

  artifactUrl(jobId: string, artifact: Artifact): string {
    const job = this.jobs.get(jobId);
    const file = job ? this.entry(job)?.artifact_files[artifactBasename(artifact.uri)] : undefined;
    return file ? `${this.sampleBase}${file}` : '';
  }

  artifactAltText(jobId: string, artifact: Artifact): string | undefined {
    const job = this.jobs.get(jobId);
    return job ? this.entry(job)?.artifact_alt[artifactBasename(artifact.uri)] : undefined;
  }

  // ---------- timeline ----------

  /** Index of the event after which the recording waits for the operator (request_human_approval call). */
  private pauseIndex(rec: Recording): number {
    if (!rec.capture.approval_request) return -1;
    return rec.events.findIndex((e) => e.kind === 'tool_call' && e.tool === 'request_human_approval');
  }

  private preEvents(rec: Recording): TraceEvent[] {
    const p = this.pauseIndex(rec);
    return p >= 0 ? rec.events.slice(0, p + 1) : rec.events;
  }

  /** Events the replay writes after the operator decides (simulated; marked as such). */
  private postEvents(job: ReplayJob): TraceEvent[] {
    const d = job.decision!;
    const rec = job.rec;
    const pre = this.preEvents(rec);
    const last = pre.at(-1)!;
    const recordedEsc = rec.events.find((e) => e.kind === 'escalate');
    const recordedEnd = rec.events.find((e) => e.kind === 'episode_end');
    const rank = d.chosen_rank ?? null;
    const ts = new Date(Date.parse(last.ts) + Math.max(0, (job.decidedAt ?? 0) - job.created - this.pausedAt(job))).toISOString();
    const who = { chosen_rank: rank, approver: d.approver, ...(d.note ? { note: d.note } : {}), simulated: true };
    let step = last.step;
    const base = (kind: TraceEvent['kind']): TraceEvent => ({
      v: '1',
      run_id: last.run_id,
      step: ++step,
      ts,
      kind,
      tool: null,
      args: {},
      result_summary: {},
      margin: null,
      tau: null,
      because: null,
      stage_move_nm: null,
      latency_ms: null,
      policy: null,
    });
    const out: TraceEvent[] = [{ ...base('tool_result'), tool: 'request_human_approval', result_summary: who }];
    out.push({
      ...base('escalate'),
      margin: recordedEsc?.margin ?? last.margin ?? null,
      tau: recordedEsc?.tau ?? last.tau ?? null,
      because: recordedEsc?.because ?? null,
      policy: recordedEsc?.policy ?? null,
      result_summary: who,
    });
    const chosen = rank === null ? undefined : rec.capture.approval_request?.options.find((o) => o.rank === rank);
    out.push({
      ...base('human_approval'),
      tool: rank === null ? null : 'operator_choice',
      args: { chosen_rank: rank },
      result_summary: { ...who, ...(chosen ? { x_px: chosen.x_px, y_px: chosen.y_px, score: chosen.score } : {}) },
      because:
        rank === null
          ? `${d.approver} rejected all candidates: no stage move`
          : `${d.approver} approved candidate #${rank}; the agent did not commit on its own (margin below tau)`,
    });
    out.push({
      ...base('episode_end'),
      result_summary: {
        ...(recordedEnd?.result_summary ?? {}),
        outcome: 'escalate',
        operator_choice: rank,
        simulated: true,
      },
    });
    return out;
  }

  /** ms after `created` at which the pre-approval part finishes. */
  private pausedAt(job: ReplayJob): number {
    const pre = this.preEvents(job.rec);
    return REPLAY_QUEUED_MS + (revealTimes(pre).at(-1) ?? 0);
  }

  private visibleEvents(job: ReplayJob): TraceEvent[] | null {
    const t = this.now() - job.created - REPLAY_QUEUED_MS;
    if (t < 0) return null;
    const pre = this.preEvents(job.rec);
    const times = revealTimes(pre);
    const shown = pre.filter((_, i) => times[i]! <= t);
    if (!job.decision || shown.length < pre.length) return shown;
    const post = this.postEvents(job);
    const since = this.now() - (job.decidedAt ?? this.now());
    const postTimes = post.map((_, i) => (i + 1) * REPLAY_STEP_MS);
    return [...shown, ...post.filter((_, i) => postTimes[i]! <= since)];
  }

  private postDoneAt(job: ReplayJob): number {
    return (job.decidedAt ?? 0) + 4 * REPLAY_STEP_MS;
  }

  private snapshot(job: ReplayJob): JobStatus {
    const t = this.now() - job.created;
    const iso = (ms: number) => new Date(job.created + ms).toISOString();
    const entry = this.entry(job);
    const recorded = entry ? (structuredClone(entry.result) as LocalizeResult) : null;
    const base: JobStatus = {
      job_id: job.id,
      status: 'queued',
      created_at: iso(0),
      updated_at: iso(0),
      engine: recorded?.engine ?? null,
      result: null,
      trace_key: null,
      approval: null,
      error: null,
      worker: null,
    };
    if (t < REPLAY_QUEUED_MS) return base;
    const running: JobStatus = { ...base, status: 'running', updated_at: iso(REPLAY_QUEUED_MS) };
    if (job.agent) running.trace_key = `traces/${job.rec.events[0]?.run_id ?? job.id}.jsonl`;

    if (!job.agent) {
      // Single localization: the recorded decisive LocalizeResult, no agent loop.
      const doneAt = REPLAY_QUEUED_MS + 1200;
      if (t < doneAt || !recorded) return running;
      return { ...running, status: 'succeeded', updated_at: iso(doneAt), result: recorded };
    }

    const pausedAt = this.pausedAt(job);
    if (t < pausedAt) return running;
    const req = job.rec.capture.approval_request;
    if (!req) return { ...running, status: 'succeeded', updated_at: iso(pausedAt), result: recorded };

    if (!job.decision) {
      return { ...running, status: 'awaiting_approval', updated_at: iso(pausedAt), result: recorded, approval: req };
    }
    const decidedMs = (job.decidedAt ?? this.now()) - job.created;
    if (this.now() < this.postDoneAt(job)) {
      return { ...running, status: 'running', updated_at: iso(decidedMs), result: recorded };
    }
    const doneMs = this.postDoneAt(job) - job.created;
    const rank = job.decision.chosen_rank ?? null;
    if (rank === null) {
      return {
        ...running,
        status: 'failed',
        updated_at: iso(doneMs),
        result: recorded,
        error: { code: 'rejected_by_operator', message: 'The operator rejected all candidates; no stage move was made.' },
      };
    }
    const chosen = req.options.find((o) => o.rank === rank)!;
    const result = recorded
      ? {
          ...recorded,
          offset: { ...recorded.offset, x_px: chosen.x_px, y_px: chosen.y_px },
          warnings: [
            ...(recorded.warnings ?? []),
            JSON.stringify({ operator_choice: { rank, approver: job.decision.approver, x_px: chosen.x_px, y_px: chosen.y_px, simulated: true } }),
          ],
        }
      : null;
    return { ...running, status: 'succeeded', updated_at: iso(doneMs), result };
  }
}
