import { describe, expect, it } from 'vitest';
import { JobPoller, MAX_CONSECUTIVE_ERRORS, POLL_MAX_MS, POLL_MIN_MS, nextDelay, phaseForStatus, type PollPhase } from '../src/lib/poll';
import type { JobStatus } from '../src/types/contracts';

const job = (status: JobStatus['status']): JobStatus => ({
  job_id: 'j1',
  status,
  created_at: '2026-10-11T00:00:00Z',
  updated_at: '2026-10-11T00:00:00Z',
});

/** Scripted fetcher: each call returns (or throws) the next item. */
function script(items: (JobStatus['status'] | Error)[]) {
  let i = 0;
  return {
    calls: () => i,
    fetchJob: async () => {
      const it = items[Math.min(i++, items.length - 1)]!;
      if (it instanceof Error) throw it;
      return job(it);
    },
  };
}

const instantSleep = async () => {};

describe('nextDelay', () => {
  it('backs off from 0.5 s to a 2 s cap', () => {
    const seq: number[] = [];
    let d: number | null = null;
    for (let i = 0; i < 7; i++) seq.push((d = nextDelay(d)));
    expect(seq[0]).toBe(POLL_MIN_MS);
    expect(seq).toEqual([500, 750, 1125, 1688, 2000, 2000, 2000]);
    expect(Math.max(...seq)).toBe(POLL_MAX_MS);
  });
});

describe('phaseForStatus', () => {
  it('maps statuses', () => {
    expect(phaseForStatus('queued')).toBe('polling');
    expect(phaseForStatus('running')).toBe('polling');
    expect(phaseForStatus('awaiting_approval')).toBe('paused');
    expect(phaseForStatus('succeeded')).toBe('done');
    expect(phaseForStatus('failed')).toBe('done');
  });
});

describe('JobPoller', () => {
  it('polls until succeeded, backing off while the status is unchanged', async () => {
    const s = script(['queued', 'running', 'running', 'running', 'succeeded']);
    const seen: string[] = [];
    const phases: PollPhase[] = [];
    const p = new JobPoller({ fetchJob: s.fetchJob, onUpdate: (j) => seen.push(j.status), onPhase: (ph) => phases.push(ph), sleep: instantSleep });
    await p.start();
    expect(seen).toEqual(['queued', 'running', 'running', 'running', 'succeeded']);
    expect(p.phase).toBe('done');
    expect(phases).toEqual(['polling', 'done']);
    // reset to 500 on each status transition, then back off
    expect(p.waits).toEqual([500, 500, 750, 1125]);
  });

  it('stops on failed', async () => {
    const s = script(['running', 'failed', 'running']);
    const p = new JobPoller({ fetchJob: s.fetchJob, onUpdate: () => {}, sleep: instantSleep });
    await p.start();
    expect(p.phase).toBe('done');
    expect(s.calls()).toBe(2);
  });

  it('pauses on awaiting_approval and resumes on demand', async () => {
    const s = script(['running', 'awaiting_approval', 'running', 'succeeded']);
    const p = new JobPoller({ fetchJob: s.fetchJob, onUpdate: () => {}, sleep: instantSleep });
    await p.start();
    expect(p.phase).toBe('paused');
    expect(s.calls()).toBe(2);
    await p.resume();
    expect(p.phase).toBe('done');
    expect(s.calls()).toBe(4);
  });

  it('tolerates transient errors, gives up after MAX_CONSECUTIVE_ERRORS', async () => {
    const boom = new Error('net');
    const s1 = script(['running', boom, boom, 'succeeded']);
    const p1 = new JobPoller({ fetchJob: s1.fetchJob, onUpdate: () => {}, sleep: instantSleep });
    await p1.start();
    expect(p1.phase).toBe('done');

    const s2 = script([boom]);
    let lastErr: unknown;
    const p2 = new JobPoller({ fetchJob: s2.fetchJob, onUpdate: () => {}, onPhase: (_p, e) => (lastErr = e), sleep: instantSleep });
    await p2.start();
    expect(p2.phase).toBe('error');
    expect(s2.calls()).toBe(MAX_CONSECUTIVE_ERRORS);
    expect(lastErr).toBe(boom);
  });

  it('stop() aborts a pending wait and prevents further fetches', async () => {
    const s = script(['running']);
    let release: () => void = () => {};
    const p = new JobPoller({
      fetchJob: s.fetchJob,
      onUpdate: () => {},
      sleep: (_ms, signal) =>
        new Promise<void>((r) => {
          release = r;
          signal.addEventListener('abort', () => r());
        }),
    });
    const done = p.start();
    await Promise.resolve();
    await Promise.resolve();
    p.stop();
    await done;
    release();
    expect(p.phase).toBe('stopped');
    expect(s.calls()).toBe(1);
  });
});
