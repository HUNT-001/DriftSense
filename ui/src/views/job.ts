import type { TraceFetch } from '../api/types';
import { ApiError } from '../api/types';
import type { JobStatus } from '../types/contracts';
import { clear, h } from '../lib/dom';
import { fmtTime } from '../lib/format';
import { JobPoller, type PollPhase } from '../lib/poll';
import { renderApprovalCard } from './approval';
import { STATUS_TEXT, errorBox, statusPill } from './components';
import type { AppContext, View } from './context';
import { renderResult } from './result';
import { renderTrace } from './trace';

export function jobView(ctx: AppContext, jobId: string): View {
  const heading = h('h1', { tabindex: '-1' }, 'Run ', h('code', null, jobId));
  const pillSlot = h('div', { class: 'status-line' });
  const meta = h('p', { class: 'muted meta' });
  const pollNote = h('p', { class: 'muted poll-note' });
  const errorSlot = h('div');
  const approvalSlot = h('div');
  const traceSection = h('section', { class: 'card', 'aria-labelledby': 'trace-h', hidden: true });
  const traceBody = h('div');
  traceSection.append(h('h2', { id: 'trace-h' }, 'Agent decision trace'), traceBody);
  const resultSection = h('section', { 'aria-labelledby': 'result-h', hidden: true });
  const resultBody = h('div');
  resultSection.append(h('h2', { id: 'result-h' }, 'Result'), resultBody);

  const el = h(
    'div',
    { class: 'view view-job' },
    h('p', null, h('a', { href: '#/' }, '← Try another site')),
    heading,
    pillSlot,
    meta,
    pollNote,
    errorSlot,
    approvalSlot,
    traceSection,
    resultSection,
  );

  let last: JobStatus | null = null;
  let lastResultKey = '';
  let traceCount = -1;
  let traceBusy = false;
  let tracePending: JobStatus | null = null;
  let decisionMsg = '';
  let disposed = false;

  const refreshTrace = async (job: JobStatus): Promise<void> => {
    if (disposed) return;
    if (traceBusy) {
      tracePending = job; // re-fetch once the in-flight request finishes
      return;
    }
    traceBusy = true;
    try {
      const t: TraceFetch | null = await ctx.api.getTrace(job.job_id);
      if (disposed) return;
      if (!t) {
        if (job.trace_key) {
          traceSection.hidden = false;
          if (traceCount !== 0) {
            clear(traceBody);
            traceBody.append(h('p', { class: 'muted' }, 'Trace not available yet.'));
            traceCount = 0;
          }
        }
        return;
      }
      traceSection.hidden = false;
      const key = t.events.length + t.errors.length * 1000;
      if (key !== traceCount) {
        traceCount = key;
        clear(traceBody);
        traceBody.append(renderTrace(t));
      }
    } catch (e) {
      if (disposed) return;
      traceSection.hidden = false;
      clear(traceBody);
      traceBody.append(errorBox('Could not load the agent trace', (e as Error).message));
      traceCount = -1;
    } finally {
      traceBusy = false;
      const next = tracePending;
      tracePending = null;
      if (next) void refreshTrace(next);
    }
  };

  const update = (job: JobStatus) => {
    const changed = !last || last.status !== job.status;
    last = job;
    clear(pillSlot);
    pillSlot.append(statusPill(job.status));
    meta.textContent = `Created ${fmtTime(job.created_at)} · updated ${fmtTime(job.updated_at)}${job.engine ? ` · engine ${job.engine}` : ''}`;
    if (changed) ctx.announce(`Job status: ${STATUS_TEXT[job.status]}.`);
    ctx.setWorker(job.worker);

    clear(errorSlot);
    if (job.status === 'failed') {
      errorSlot.append(
        errorBox(
          job.error?.code === 'rejected_by_operator' ? 'Rejected by operator' : 'The job failed',
          job.error ? `${job.error.message} (code: ${job.error.code})` : 'No error details were returned.',
        ),
      );
    }

    if (job.status === 'awaiting_approval' && job.approval) {
      if (!approvalSlot.firstChild) {
        approvalSlot.append(
          renderApprovalCard(ctx, job, (updated, message) => {
            decisionMsg = message;
            update(updated);
            void poller.resume();
          }),
        );
      }
    } else if (approvalSlot.firstChild && job.status !== 'awaiting_approval') {
      // Replace the card with a short, persistent summary of the operator's decision.
      clear(approvalSlot);
      if (decisionMsg) {
        approvalSlot.append(
          h('p', { class: 'notice notice-ok' }, h('span', { 'aria-hidden': 'true' }, '✔ '), decisionMsg),
        );
      }
    }

    if (job.trace_key || ctx.api.kind === 'mock') void refreshTrace(job);

    const resultKey = job.result ? `${job.status}|${job.updated_at}` : '';
    if (resultKey !== lastResultKey) {
      lastResultKey = resultKey;
      clear(resultBody);
      if (job.result) {
        resultSection.hidden = false;
        resultBody.append(renderResult(ctx.api, job));
      } else resultSection.hidden = true;
    }
  };

  const onPhase = (phase: PollPhase, err?: unknown) => {
    if (phase === 'stopped') return;
    pollNote.textContent = '';
    if (phase === 'polling') pollNote.textContent = 'Checking for updates…';
    if (phase === 'paused') pollNote.textContent = 'Waiting for an operator decision. Updates resume after you decide.';
    if (phase === 'error') {
      clear(pollNote);
      const retry = h('button', { type: 'button', class: 'btn' }, 'Retry');
      retry.addEventListener('click', () => void poller.resume());
      const notFound = err instanceof ApiError && err.status === 404;
      pollNote.append(
        errorBox(
          notFound ? 'Job not found' : 'Lost contact with the API',
          (err as Error)?.message ?? 'Unknown error',
          notFound ? h('a', { href: '#/' }, 'Start a new run') : retry,
        ),
      );
      ctx.announce(notFound ? 'Job not found.' : 'Lost contact with the API.');
    }
  };

  const poller = new JobPoller({
    fetchJob: async () => {
      try {
        return await ctx.api.getJob(jobId);
      } catch (e) {
        // A 404 will not fix itself: surface it immediately.
        if (e instanceof ApiError && e.status === 404) {
          onPhase('error', e);
          poller.stop();
        }
        throw e;
      }
    },
    onUpdate: update,
    onPhase,
  });
  void poller.start();

  return {
    el,
    heading,
    dispose() {
      disposed = true;
      poller.stop();
    },
  };
}
