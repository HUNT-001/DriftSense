import type { ApprovalRequest, JobStatus } from '../types/contracts';
import { ApiError } from '../api/types';
import {
  APPROVER_MAX,
  NOTE_MAX,
  buildApprovalDecision,
  describeDecision,
  explainEscalation,
  type ApprovalChoice,
} from '../lib/approval';
import { artifactAlt } from '../lib/artifacts';
import { clear, h } from '../lib/dom';
import { fmt, viridis } from '../lib/format';
import { marginBar, marginDomain, replayBadge } from './components';
import type { AppContext } from './context';

const APPROVER_KEY = 'driftsense.approver';

function rememberedApprover(): string {
  try {
    return localStorage.getItem(APPROVER_KEY) ?? '';
  } catch {
    return '';
  }
}

function rememberApprover(v: string) {
  try {
    localStorage.setItem(APPROVER_KEY, v);
  } catch {
    /* storage unavailable: ignore */
  }
}

/**
 * Operator approval card for a job in `awaiting_approval`.
 * Flow: choose (Approve #n / Reject all) -> validate name -> confirm step -> POST ApprovalDecision.
 */
export function renderApprovalCard(
  ctx: AppContext,
  job: JobStatus,
  onDecided: (job: JobStatus, message: string) => void,
  pixelNm: number | null = null,
): HTMLElement {
  const req = job.approval as ApprovalRequest;
  const options = req.options;
  const candArtifact = job.result?.artifacts?.find((a) => a.kind === 'candidates');

  const nameErr = h('p', { id: 'approver-err', class: 'field-error', hidden: true });
  const approver = h('input', {
    id: 'approver',
    name: 'approver',
    type: 'text',
    autocomplete: 'name',
    required: true,
    maxlength: APPROVER_MAX,
    'aria-required': 'true',
    'aria-describedby': 'approver-hint',
  });
  approver.value = rememberedApprover();
  const note = h('textarea', { id: 'approval-note', name: 'note', rows: 2, maxlength: NOTE_MAX });
  const confirmArea = h('div', { class: 'confirm', hidden: true });
  const status = h('div', { class: 'approval-status', role: 'status' });
  const choiceButtons: HTMLButtonElement[] = [];

  const scores = options.map((o) => o.score);
  const lo = Math.min(...scores);
  const hi = Math.max(...scores);

  const setBusy = (busy: boolean) => {
    for (const b of choiceButtons) b.disabled = busy;
    approver.disabled = busy;
    note.disabled = busy;
  };

  const showNameError = (msg: string | null) => {
    nameErr.hidden = !msg;
    nameErr.textContent = msg ?? '';
    if (msg) {
      approver.setAttribute('aria-invalid', 'true');
      approver.setAttribute('aria-describedby', 'approver-hint approver-err');
    } else {
      approver.removeAttribute('aria-invalid');
      approver.setAttribute('aria-describedby', 'approver-hint');
    }
  };

  const submit = async (choice: ApprovalChoice) => {
    const built = buildApprovalDecision({ choice, approver: approver.value, note: note.value, optionCount: options.length });
    if (!built.ok) return;
    setBusy(true);
    clear(confirmArea);
    confirmArea.hidden = true;
    status.textContent = 'Sending decision…';
    try {
      const updated = await ctx.api.approve(job.job_id, built.decision);
      rememberApprover(built.decision.approver);
      const msg =
        choice === 'reject'
          ? `Decision recorded: all candidates rejected by ${built.decision.approver}. No stage move.`
          : `Decision recorded: rank ${choice} approved by ${built.decision.approver}.`;
      status.textContent = msg;
      ctx.announce(msg);
      onDecided(updated, msg);
    } catch (e) {
      setBusy(false);
      const msg = e instanceof ApiError ? e.message : (e as Error).message;
      status.textContent = '';
      status.append(h('span', { class: 'field-error' }, `Could not send the decision: ${msg}. Please try again.`));
    }
  };

  const ask = (choice: ApprovalChoice, trigger: HTMLButtonElement) => {
    const built = buildApprovalDecision({ choice, approver: approver.value, note: note.value, optionCount: options.length });
    if (!built.ok) {
      const ne = built.errors.find((e) => e.field === 'approver');
      showNameError(ne?.message ?? null);
      status.textContent = built.errors.map((e) => e.message).join(' ');
      approver.focus();
      return;
    }
    showNameError(null);
    status.textContent = '';
    clear(confirmArea);
    const confirmBtn = h('button', { type: 'button', class: choice === 'reject' ? 'btn btn-danger' : 'btn btn-primary' }, 'Confirm');
    const cancelBtn = h('button', { type: 'button', class: 'btn' }, 'Cancel');
    const heading = h('h3', { id: 'confirm-h', tabindex: '-1' }, 'Confirm your decision');
    confirmArea.append(
      heading,
      h('p', null, describeDecision(choice, options)),
      h('p', { class: 'muted' }, `Approver: ${built.decision.approver}${built.decision.note ? ` — note: ${built.decision.note}` : ''}`),
      h('div', { class: 'actions' }, confirmBtn, cancelBtn),
    );
    confirmArea.setAttribute('role', 'group');
    confirmArea.setAttribute('aria-labelledby', 'confirm-h');
    confirmArea.hidden = false;
    confirmBtn.addEventListener('click', () => void submit(choice));
    cancelBtn.addEventListener('click', () => {
      confirmArea.hidden = true;
      clear(confirmArea);
      trigger.focus();
    });
    confirmArea.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape') cancelBtn.click();
    });
    heading.focus();
  };

  const optionList = h(
    'ol',
    { class: 'options', 'aria-label': 'Candidates offered by the agent' },
    options.map((o) => {
      const btn = h(
        'button',
        { type: 'button', class: 'btn btn-primary', 'aria-describedby': `opt-${o.rank}-desc` },
        `Approve rank ${o.rank}`,
      );
      btn.addEventListener('click', () => ask(o.rank as ApprovalChoice, btn));
      choiceButtons.push(btn);
      const sw = h('span', { class: 'swatch', 'aria-hidden': 'true' });
      sw.style.setProperty('background-color', viridis(hi > lo ? (o.score - lo) / (hi - lo) : 1));
      const nm =
        pixelNm !== null
          ? h('p', { class: 'muted' }, `≈ x ${fmt(o.x_px * pixelNm, 1)} nm, y ${fmt(o.y_px * pixelNm, 1)} nm`)
          : null;
      return h(
        'li',
        { class: 'option' },
        h('p', { class: 'option-rank' }, `Rank ${o.rank}`),
        h(
          'div',
          { id: `opt-${o.rank}-desc` },
          h('p', null, `x ${fmt(o.x_px, 1)} px, y ${fmt(o.y_px, 1)} px`),
          nm,
          h('p', null, sw, `fused score ${fmt(o.score, 3)}`),
          typeof o.roughness_score === 'number' ? h('p', { class: 'muted' }, `roughness ${fmt(o.roughness_score, 3)}`) : null,
          typeof o.lattice_score === 'number' ? h('p', { class: 'muted' }, `lattice ${fmt(o.lattice_score, 3)}`) : null,
        ),
        btn,
      );
    }),
  );

  const rejectBtn = h('button', { type: 'button', class: 'btn btn-danger' }, 'Reject all (no stage move)');
  rejectBtn.addEventListener('click', () => ask('reject', rejectBtn));
  choiceButtons.push(rejectBtn);

  const heading = h('h2', { id: 'approval-h', tabindex: '-1' }, h('span', { 'aria-hidden': 'true' }, '⚠ '), 'Operator approval needed');
  const { margin, tau } = req.margin;
  const card = h(
    'section',
    { class: 'card approval', 'aria-labelledby': 'approval-h' },
    heading,
    ctx.api.kind === 'replay'
      ? h(
          'p',
          { class: 'notice notice-replay' },
          replayBadge('SIMULATED APPROVAL'),
          ' Offline replay: the candidates and margin are the recorded engine output; your decision is applied in this browser tab only.',
        )
      : null,
    h('p', { class: 'lead' }, explainEscalation(req)),
    h(
      'dl',
      { class: 'kv approval-kv' },
      h('dt', null, 'Uniqueness margin'),
      h('dd', null, fmt(margin, 3)),
      h('dt', null, 'Commit threshold τ'),
      h('dd', null, `${fmt(tau, 3)}${req.margin.calibrated ? ' (calibrated)' : ''}`),
      typeof req.margin.p_correct === 'number' ? [h('dt', null, 'P(correct)'), h('dd', null, fmt(req.margin.p_correct, 3))] : null,
    ),
    marginBar(margin, tau, marginDomain([margin], tau), false, true),
    candArtifact
      ? h(
          'figure',
          { class: 'approval-thumb' },
          h('img', {
            src: ctx.api.artifactUrl(job.job_id, candArtifact),
            alt: artifactAlt(candArtifact, job, ctx.api.artifactAltText?.(job.job_id, candArtifact)),
            loading: 'lazy',
          }),
          h('figcaption', null, 'Top candidate crops, ranked left to right'),
        )
      : null,
    h('h3', null, `Top ${options.length} candidates`),
    optionList,
    h(
      'div',
      { class: 'field' },
      h('label', { for: 'approver' }, 'Your name (required)'),
      h('p', { id: 'approver-hint', class: 'hint' }, 'Recorded with the decision for the audit trail.'),
      approver,
      nameErr,
    ),
    h('div', { class: 'field' }, h('label', { for: 'approval-note' }, 'Note (optional)'), note),
    h('div', { class: 'actions' }, rejectBtn),
    confirmArea,
    status,
  );
  return card;
}
