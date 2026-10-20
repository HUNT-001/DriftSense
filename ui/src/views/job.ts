import { ReplayApi } from '../api/replay';
import { ApiError, isKnownEpisode, rememberEpisode } from '../api/types';
import type { JobStatus } from '../types/contracts';
import { clear, h } from '../lib/dom';
import { fmtTime } from '../lib/format';
import { JobPoller, phaseForStatus, type PollPhase } from '../lib/poll';
import { pixelSizeNm } from '../lib/trace';
import { jobHref } from '../router';
import { renderApprovalCard } from './approval';
import { STATUS_TEXT, errorBox, replayBadge, statusPill } from './components';
import type { AppContext, View } from './context';
import { episodeBlock, outcomeSummary, renderResult, responsiblePanel } from './result';
import { createTraceView } from './trace';

/** Trace refresh cadence while the job runs (ms). The trace grows independently of the status. */
export const TRACE_POLL_MS = { live: 1000, replay: 350 } as const;

export function jobView(ctx: AppContext, jobId: string): View {
  const episode = isKnownEpisode(jobId);
  const heading = h('h1', { tabindex: '-1' }, episode ? 'Agent episode ' : 'Run ', h('code', null, jobId));
  const pillSlot = h('div', { class: 'status-line', role: 'status' });
  const meta = h('p', { class: 'muted meta' });
  const replayNote = h('div');
  const pollNote = h('p', { class: 'muted poll-note' });
  const outcomeSlot = h('div');
  const approvalSlot = h('div');
  const traceView = createTraceView();
  const traceSection = h(
    'section',
    { class: 'card', 'aria-labelledby': 'trace-h', hidden: true },
    h('h2', { id: 'trace-h' }, 'Agent timeline'),
    h(
      'p',
      { class: 'hint' },
      'Each row is one logged step. The OpenCV uniqueness margin is compared with τ: at or above τ the agent may commit the stage move; below τ it re-acquires, or escalates to you when its budget is spent.',
    ),
    traceView.el,
  );
  const traceError = h('div');
  traceSection.append(traceError);
  const episodeSection = h('section', { class: 'card', 'aria-labelledby': 'ep-h', hidden: true });
  const resultSection = h('section', { 'aria-labelledby': 'result-h', hidden: true });
  const resultBody = h('div');
  resultSection.append(h('h2', { id: 'result-h' }, 'Localization result'), resultBody);

  const el = h(
    'div',
    { class: 'view view-job' },
    h('p', null, h('a', { href: '#/' }, '← Start another run')),
    heading,
    pillSlot,
    meta,
    replayNote,
    pollNote,
    outcomeSlot,
    approvalSlot,
    traceSection,
    episodeSection,
    resultSection,
    responsiblePanel(),
  );

  if (ctx.api instanceof ReplayApi) {
    const info = ctx.api.describeJob(jobId);
    if (info) {
      const other = info.recording === 'demo_commit' ? 'escalate' : 'commit';
      const req = info.requested;
      const mismatch = req && `vsem-${req.family}-${req.seed}` !== info.demoId;
      replayNote.append(
        h(
          'p',
          { class: 'notice notice-replay' },
          replayBadge(),
          ` Offline replay of a recorded trace: playing ${info.demoId} (traces/${info.recording}.jsonl).`,
          mismatch ? ` Family "${req!.family}" seed ${req!.seed} was not recorded, so the closest recorded episode is shown.` : '',
          ' ',
          h('a', { href: '#/', 'data-replay-other': other }, `Replay the ${other} episode next`),
        ),
      );
    }
  }

  let last: JobStatus | null = null;
  let lastResultKey = '';
  let decisionMsg = '';
  let decidedHere = false;
  let disposed = false;
  let traceTimer: ReturnType<typeof setTimeout> | undefined;
  let traceBusy = false;
  let wantsTrace = episode || ctx.api.kind === 'replay';
  let pixelNm: number | null = null;
  let episodeLoaded = false;

  const refreshTrace = async (): Promise<void> => {
    if (disposed || traceBusy || !wantsTrace) return;
    traceBusy = true;
    try {
      const t = await ctx.api.getTrace(jobId);
      if (disposed) return;
      clear(traceError);
      if (!t) {
        if (last?.trace_key || episode) {
          traceSection.hidden = false;
        }
        return;
      }
      traceSection.hidden = false;
      traceView.update(t);
      const px = pixelSizeNm(t.events);
      if (px !== pixelNm) {
        pixelNm = px;
        lastResultKey = ''; // re-render the offset with nm
        if (last) renderResultIfChanged(last);
      }
    } catch (e) {
      if (disposed) return;
      traceSection.hidden = false;
      clear(traceError);
      traceError.append(errorBox('Could not load the agent trace', (e as Error).message));
    } finally {
      traceBusy = false;
    }
  };

  /** Fast trace ticker while the job is queued/running; one last fetch after it stops. */
  const scheduleTrace = () => {
    clearTimeout(traceTimer);
    if (disposed || !wantsTrace) return;
    const active = !last || phaseForStatus(last.status) === 'polling';
    void refreshTrace().then(() => {
      if (disposed) return;
      if (active) traceTimer = setTimeout(scheduleTrace, TRACE_POLL_MS[ctx.api.kind]);
    });
  };

  const loadEpisode = async () => {
    if (episodeLoaded || !wantsTrace) return;
    episodeLoaded = true;
    try {
      const ep = await ctx.api.getArtifactJson(jobId, 'episode.json');
      const block = episodeBlock(ep);
      if (disposed || !block) return;
      clear(episodeSection);
      episodeSection.append(h('h2', { id: 'ep-h' }, 'Episode summary (episode.json)'), block);
      episodeSection.hidden = false;
    } catch {
      /* optional artifact */
    }
  };

  function renderResultIfChanged(job: JobStatus) {
    const resultKey = job.result ? `${job.status}|${job.updated_at}|${pixelNm ?? ''}` : '';
    if (resultKey === lastResultKey) return;
    lastResultKey = resultKey;
    clear(resultBody);
    if (job.result) {
      resultSection.hidden = false;
      resultBody.append(renderResult(ctx.api, job, pixelNm));
    } else resultSection.hidden = true;
  }

  const update = (job: JobStatus) => {
    const prev = last;
    const changed = !prev || prev.status !== job.status;
    last = job;
    clear(pillSlot);
    pillSlot.append(statusPill(job.status));
    meta.textContent = `Created ${fmtTime(job.created_at)} · updated ${fmtTime(job.updated_at)}${job.engine ? ` · engine ${job.engine}` : ''}`;
    if (changed && prev) ctx.announce(`Job status: ${STATUS_TEXT[job.status]}.`);
    ctx.setWorker(job.worker);
    if (job.trace_key) wantsTrace = true;

    clear(outcomeSlot);
    const outcome = outcomeSummary(job);
    if (outcome) {
      outcomeSlot.append(outcome);
      if (changed && decidedHere) outcome.focus();
    }

    if (job.status === 'awaiting_approval' && job.approval) {
      if (!approvalSlot.firstChild) {
        const card = renderApprovalCard(
          ctx,
          job,
          (updated, message) => {
            decisionMsg = message;
            decidedHere = true;
            update(updated);
            void poller.resume();
            scheduleTrace();
          },
          pixelNm,
        );
        approvalSlot.append(card);
        ctx.announce('Operator approval needed: the agent could not prove a unique match.');
        // Move focus to the card so keyboard and screen-reader users land on the decision.
        card.querySelector<HTMLElement>('#approval-h')?.focus();
      }
    } else if (approvalSlot.firstChild && job.status !== 'awaiting_approval') {
      // Replace the card with a short, persistent confirmation of the operator's decision.
      clear(approvalSlot);
      if (decisionMsg) {
        approvalSlot.append(h('p', { class: 'notice notice-ok', role: 'status' }, h('span', { 'aria-hidden': 'true' }, '✔ '), decisionMsg));
      }
    }

    if (changed) scheduleTrace();
    if (job.status === 'succeeded' || job.status === 'failed') {
      // The final lines may land just after the status flips: fetch once more shortly after.
      setTimeout(() => void refreshTrace(), 1200);
      void loadEpisode();
    }
    renderResultIfChanged(job);
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

  // "Replay the other episode" link: start it directly.
  el.addEventListener('click', (ev) => {
    const a = (ev.target as Element | null)?.closest?.('a[data-replay-other]');
    if (!a || !(ctx.api instanceof ReplayApi)) return;
    ev.preventDefault();
    const api = ctx.api;
    void api.recordings().then(async (recs) => {
      const want = a.getAttribute('data-replay-other') === 'commit' ? 'demo_commit' : 'demo_escalate';
      const rec = recs.find((r) => r.name === want);
      if (!rec) return;
      const job = await api.createEpisode({ family: rec.capture.family as 'dram', seed: rec.capture.seed });
      rememberEpisode(job.job_id);
      ctx.navigate(jobHref(job.job_id));
    });
  });

  return {
    el,
    heading,
    dispose() {
      disposed = true;
      clearTimeout(traceTimer);
      poller.stop();
    },
  };
}
