import type { JobStatus } from '../types/contracts';

export const POLL_MIN_MS = 500;
export const POLL_MAX_MS = 2000;
export const POLL_FACTOR = 1.5;
export const MAX_CONSECUTIVE_ERRORS = 5;

/** Backoff: 500 -> 750 -> 1125 -> 1688 -> 2000 (cap). `null` = first wait. */
export function nextDelay(prev: number | null): number {
  if (prev === null) return POLL_MIN_MS;
  return Math.min(POLL_MAX_MS, Math.round(prev * POLL_FACTOR));
}

/**
 * idle      not started
 * polling   fetching / waiting between fetches
 * paused    job is awaiting_approval (resume() after the decision is posted)
 * done      job reached succeeded / failed
 * error     too many consecutive fetch errors (resume() retries)
 * stopped   disposed by the view
 */
export type PollPhase = 'idle' | 'polling' | 'paused' | 'done' | 'error' | 'stopped';

export function phaseForStatus(status: JobStatus['status']): 'polling' | 'paused' | 'done' {
  if (status === 'succeeded' || status === 'failed') return 'done';
  if (status === 'awaiting_approval') return 'paused';
  return 'polling';
}

export type Sleep = (ms: number, signal: AbortSignal) => Promise<void>;

export const defaultSleep: Sleep = (ms, signal) =>
  new Promise((resolve) => {
    if (signal.aborted) return resolve();
    const t = setTimeout(resolve, ms);
    signal.addEventListener(
      'abort',
      () => {
        clearTimeout(t);
        resolve();
      },
      { once: true },
    );
  });

export interface PollerOptions {
  fetchJob: () => Promise<JobStatus>;
  onUpdate: (job: JobStatus) => void;
  onPhase?: (phase: PollPhase, error?: unknown) => void;
  sleep?: Sleep;
}

/** Poll GET /v1/jobs/{id} with backoff; stop on terminal states, pause on awaiting_approval. */
export class JobPoller {
  private _phase: PollPhase = 'idle';
  private ctrl: AbortController | null = null;
  private lastStatus: JobStatus['status'] | null = null;
  /** Delays actually waited, for diagnostics and tests. */
  readonly waits: number[] = [];
  private readonly sleep: Sleep;

  constructor(private readonly opts: PollerOptions) {
    this.sleep = opts.sleep ?? defaultSleep;
  }

  get phase(): PollPhase {
    return this._phase;
  }

  /** Starts (or restarts after paused/error) the loop. Resolves when the loop exits. */
  start(): Promise<void> {
    if (this._phase === 'polling' || this._phase === 'stopped' || this._phase === 'done') {
      return Promise.resolve();
    }
    this.ctrl = new AbortController();
    this.setPhase('polling');
    return this.loop(this.ctrl.signal);
  }

  resume(): Promise<void> {
    return this.start();
  }

  stop(): void {
    this.ctrl?.abort();
    this.setPhase('stopped');
  }

  private setPhase(p: PollPhase, err?: unknown) {
    if (this._phase === 'stopped') return;
    this._phase = p;
    this.opts.onPhase?.(p, err);
  }

  private async loop(signal: AbortSignal): Promise<void> {
    let delay: number | null = null;
    let errors = 0;
    while (!signal.aborted) {
      try {
        const job = await this.opts.fetchJob();
        if (signal.aborted) return;
        errors = 0;
        if (job.status !== this.lastStatus) delay = null; // react quickly after a transition
        this.lastStatus = job.status;
        this.opts.onUpdate(job);
        const next = phaseForStatus(job.status);
        if (next !== 'polling') {
          this.setPhase(next);
          return;
        }
      } catch (e) {
        if (signal.aborted) return;
        errors++;
        if (errors >= MAX_CONSECUTIVE_ERRORS) {
          this.setPhase('error', e);
          return;
        }
      }
      delay = nextDelay(delay);
      this.waits.push(delay);
      await this.sleep(delay, signal);
    }
  }
}
