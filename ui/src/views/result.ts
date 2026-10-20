import type { Api } from '../api/types';
import type { JobStatus, LocalizeResult, WorkerInfo } from '../types/contracts';
import { artifactAlt, artifactOrder, artifactTitle, isImageArtifact } from '../lib/artifacts';
import { h } from '../lib/dom';
import { fmt, fmtMs, fmtTime, viridis } from '../lib/format';
import { decisionChip, dl, marginBar, marginDomain, valueBar } from './components';

function offsetBlock(r: LocalizeResult, pixelNm: number | null): HTMLElement {
  const o = r.offset;
  const rows: [string, string][] = [['Offset (px)', `x ${fmt(o.x_px, 2)}, y ${fmt(o.y_px, 2)}`]];
  if (typeof o.x_nm === 'number' && typeof o.y_nm === 'number') {
    rows.push(['Offset (nm)', `x ${fmt(o.x_nm, 2)}, y ${fmt(o.y_nm, 2)}`]);
  } else {
    const px = typeof o.pixel_size_nm === 'number' ? o.pixel_size_nm : pixelNm;
    if (px !== null) {
      rows.push(['Offset (nm)', `≈ x ${fmt(o.x_px * px, 2)}, y ${fmt(o.y_px * px, 2)} (${fmt(px, 3)} nm/px from the acquisition)`]);
    }
  }
  if (typeof o.pixel_size_nm === 'number') rows.push(['Pixel size', `${fmt(o.pixel_size_nm, 3)} nm`]);
  rows.push(['Lattice', r.lattice_type ?? 'unknown']);
  rows.push(['Mode', r.mode]);
  rows.push(['Engine', `${r.engine} ${r.engine_version}`]);
  return dl(rows);
}

function marginBlock(r: LocalizeResult): HTMLElement {
  const m = r.margin;
  const sentence =
    m.decision === 'commit'
      ? `Margin ${fmt(m.margin, 3)} ≥ τ ${fmt(m.tau, 3)}: the best candidate is unique enough to commit the stage move.`
      : `Margin ${fmt(m.margin, 3)} < τ ${fmt(m.tau, 3)}: uniqueness is not proven, so the stage move needs an operator.`;
  return h(
    'div',
    { class: 'margin-block' },
    h('p', { class: 'margin-decision' }, 'Decision: ', decisionChip(m.decision)),
    marginBar(m.margin, m.tau, marginDomain([m.margin], m.tau)),
    h('p', null, sentence),
    h(
      'p',
      { class: 'muted' },
      m.calibrated ? 'τ is calibrated.' : 'τ is not calibrated (default threshold).',
      typeof m.p_correct === 'number' ? ` Estimated probability correct: ${fmt(m.p_correct, 3)}.` : '',
    ),
  );
}

export function candidatesTable(r: Pick<LocalizeResult, 'candidates'>, caption = 'Ranked candidates'): HTMLElement {
  const c = r.candidates;
  const hasLat = c.some((x) => typeof x.lattice_score === 'number');
  const hasRough = c.some((x) => typeof x.roughness_score === 'number');
  const scores = c.map((x) => x.score);
  const lo = Math.min(...scores);
  const hi = Math.max(...scores);
  const norm = (s: number) => (hi > lo ? (s - lo) / (hi - lo) : 1);
  const swatch = (s: number) => {
    const sw = h('span', { class: 'swatch', 'aria-hidden': 'true' });
    sw.style.setProperty('background-color', viridis(norm(s)));
    return sw;
  };
  return h(
    'div',
    { class: 'table-wrap', tabindex: '0', role: 'region', 'aria-label': caption },
    h(
      'table',
      { class: 'data' },
      h('caption', null, caption),
      h(
        'thead',
        null,
        h(
          'tr',
          null,
          h('th', { scope: 'col' }, 'Rank'),
          h('th', { scope: 'col' }, 'x (px)'),
          h('th', { scope: 'col' }, 'y (px)'),
          h('th', { scope: 'col' }, 'Score'),
          hasLat ? h('th', { scope: 'col' }, 'Lattice') : null,
          hasRough ? h('th', { scope: 'col' }, 'Roughness') : null,
        ),
      ),
      h(
        'tbody',
        null,
        c.map((x) =>
          h(
            'tr',
            null,
            h('th', { scope: 'row' }, `#${x.rank}`),
            h('td', null, fmt(x.x_px, 1)),
            h('td', null, fmt(x.y_px, 1)),
            h('td', null, swatch(x.score), fmt(x.score, 3)),
            hasLat ? h('td', null, fmt(x.lattice_score, 3)) : null,
            hasRough ? h('td', null, fmt(x.roughness_score, 3)) : null,
          ),
        ),
      ),
    ),
  );
}

function timingsBlock(r: LocalizeResult): HTMLElement {
  const t = r.timings ?? [];
  const max = Math.max(...t.map((x) => x.ms), 0);
  return h(
    'div',
    { class: 'timings' },
    t.length
      ? h(
          'ul',
          { class: 'bars', 'aria-label': 'Stage timings' },
          t.map((x) => h('li', null, valueBar(x.stage, fmtMs(x.ms), max > 0 ? x.ms / max : 0))),
        )
      : h('p', { class: 'muted' }, 'No per-stage timings reported.'),
    h('p', null, h('strong', null, 'Total: '), fmtMs(r.total_ms)),
  );
}

export function workerBlock(w: WorkerInfo | null | undefined): HTMLElement {
  if (!w) return h('p', { class: 'muted' }, 'Worker not reported yet.');
  return dl([
    ['Instance type', w.instance_type ?? 'unknown'],
    ['Architecture', w.arch ?? 'unknown'],
    ['OpenCV', w.opencv_version],
    ['NumPy', w.numpy_version],
    ['COOL worker', w.cool ? '✔ yes' : 'no'],
  ]);
}

export function artifactGallery(api: Api, job: JobStatus): HTMLElement | null {
  const arts = [...(job.result?.artifacts ?? [])].sort((a, b) => artifactOrder(a) - artifactOrder(b));
  if (!arts.length) return null;
  return h(
    'ul',
    { class: 'gallery' },
    arts.map((a) => {
      const url = api.artifactUrl(job.job_id, a);
      if (!url) return h('li', null, h('p', { class: 'muted' }, `${artifactTitle(a)}: not available.`));
      if (!isImageArtifact(a)) {
        return h('li', null, h('a', { href: url, rel: 'noopener', target: '_blank' }, `${artifactTitle(a)} (JSON, opens in a new tab)`));
      }
      const img = h('img', {
        src: url,
        alt: artifactAlt(a, job, api.artifactAltText?.(job.job_id, a)),
        loading: 'lazy',
        decoding: 'async',
      });
      img.addEventListener('error', () => {
        img.replaceWith(h('p', { class: 'muted img-missing' }, `${artifactTitle(a)} could not be loaded.`));
      });
      return h('li', null, h('figure', null, img, h('figcaption', null, artifactTitle(a))));
    }),
  );
}

const ERROR_TEXT: Record<string, string> = {
  rejected_by_operator: 'Rejected by the operator: no stage move was made.',
  approval_timeout: 'No operator decision arrived in time, so the episode ended safely without a stage move.',
  agent_abort: 'The agent aborted safely (a tool failed or timed out); no stage move was made.',
  bad_input: 'The input images could not be used.',
  engine_error: 'The engine failed repeatedly on this job.',
};

const ERROR_TITLE: Record<string, string> = {
  rejected_by_operator: 'Rejected by operator',
  approval_timeout: 'Approval timed out',
  agent_abort: 'Agent aborted safely',
};

interface OperatorChoice {
  rank?: number;
  approver?: string;
}

/** The service appends a JSON audit warning `{"operator_choice": {...}}` after an approval. */
export function operatorChoice(r: LocalizeResult | null | undefined): OperatorChoice | null {
  for (const w of r?.warnings ?? []) {
    if (!w.trimStart().startsWith('{')) continue;
    try {
      const c = (JSON.parse(w) as { operator_choice?: OperatorChoice }).operator_choice;
      if (c && typeof c === 'object') return c;
    } catch {
      /* not JSON */
    }
  }
  return null;
}

/** One-paragraph outcome of the whole job (final JobStatus). */
export function outcomeSummary(job: JobStatus): HTMLElement | null {
  const r = job.result;
  const choice = operatorChoice(r);
  let title: string;
  let text: string;
  let cls: string;
  if (job.status === 'succeeded') {
    cls = 'notice-ok';
    if (choice) {
      title = `Succeeded on the operator's choice (rank ${choice.rank ?? '?'})`;
      text = `${choice.approver ?? 'The operator'} approved rank ${choice.rank ?? '?'}; the offset below is that candidate. The agent itself never commits below τ.`;
    } else if (r?.margin.decision === 'commit') {
      title = 'Succeeded: unique match, stage correction committed';
      text = `Uniqueness margin ${fmt(r.margin.margin, 3)} ≥ τ ${fmt(r.margin.tau, 3)}.`;
    } else {
      title = 'Succeeded, but uniqueness was not proven';
      text = r ? `Margin ${fmt(r.margin.margin, 3)} < τ ${fmt(r.margin.tau, 3)}: treat the offset as a suggestion for an operator.` : '';
    }
  } else if (job.status === 'failed') {
    cls = 'notice-error';
    const code = job.error?.code ?? 'unknown';
    title = ERROR_TITLE[code] ?? 'The job failed';
    text = `${ERROR_TEXT[code] ?? ''} ${job.error?.message ?? ''} (code: ${code})`.trim();
  } else return null;
  return h(
    'div',
    { class: `notice ${cls} outcome`, tabindex: '-1', 'data-testid': 'outcome' },
    h('p', { class: 'notice-title' }, h('span', { 'aria-hidden': 'true' }, job.status === 'succeeded' ? '✔ ' : '✖ '), title),
    h('p', null, text),
    h('p', { class: 'muted' }, `Finished ${fmtTime(job.updated_at)}${job.engine ? ` · engine ${job.engine}` : ''}${r ? ` · ${fmtMs(r.total_ms)} engine time` : ''}`),
  );
}

function scalar(v: unknown): string {
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : fmt(v, 3);
  if (Array.isArray(v)) return v.map((x) => (typeof x === 'number' ? fmt(x, 2) : String(x))).join(', ');
  return String(v);
}

/** Scalar fields of episode.json (its shape is owned by the service, so it is shown generically). */
export function episodeBlock(ep: unknown): HTMLElement | null {
  if (!ep || typeof ep !== 'object' || Array.isArray(ep)) return null;
  const rows = Object.entries(ep as Record<string, unknown>)
    .filter(([, v]) => v !== null && v !== undefined && (typeof v !== 'object' || (Array.isArray(v) && v.length <= 4)))
    .slice(0, 16)
    .map(([k, v]) => [k.replace(/_/g, ' '), scalar(v)] as [string, string]);
  return rows.length ? dl(rows) : null;
}

export function responsiblePanel(): HTMLElement {
  return h(
    'section',
    { class: 'card responsible', 'aria-labelledby': 'h-resp' },
    h('h3', { id: 'h-resp' }, 'Responsible use'),
    h('ul', { class: 'checklist' }, [
      h('li', null, h('strong', null, 'Synthetic data. '), 'Every demo site comes from a simulated SEM; no fab or customer images are used.'),
      h('li', null, h('strong', null, 'Human in the loop for stage moves. '), 'The agent never commits a move when the margin is below τ; it asks an operator, and the decision and name are logged.'),
      h('li', null, h('strong', null, 'Dose limits. '), 'Re-imaging costs electron dose; each episode has a step and dose budget the agent cannot exceed.'),
      h('li', null, h('strong', null, '7-day retention. '), 'Uploads, results and traces are deleted automatically after 7 days.'),
      h('li', null, h('strong', null, 'Not validated on production tools. '), 'A research prototype evaluated on simulated data only; it must not drive production metrology.'),
    ]),
  );
}

export function renderResult(api: Api, job: JobStatus, pixelNm: number | null = null): HTMLElement {
  const r = job.result!;
  const warnings = r.warnings ?? [];
  return h(
    'div',
    { class: 'result' },
    h(
      'div',
      { class: 'grid-2' },
      h('section', { class: 'card', 'aria-labelledby': 'h-offset' }, h('h3', { id: 'h-offset' }, 'Offset'), offsetBlock(r, pixelNm)),
      h('section', { class: 'card', 'aria-labelledby': 'h-margin' }, h('h3', { id: 'h-margin' }, 'Uniqueness margin vs τ'), marginBlock(r)),
    ),
    h('section', { class: 'card', 'aria-labelledby': 'h-cands' }, h('h3', { id: 'h-cands' }, 'Candidates'), candidatesTable(r)),
    (() => {
      const g = artifactGallery(api, job);
      return g
        ? h(
            'section',
            { class: 'card', 'aria-labelledby': 'h-art' },
            h('h3', { id: 'h-art' }, 'NCC vs roughness: where a copy could be, and which one is real'),
            h('p', { class: 'hint' }, 'Heatmaps use the perceptually uniform inferno colormap: brighter means a higher score.'),
            g,
          )
        : null;
    })(),
    h(
      'div',
      { class: 'grid-2' },
      h('section', { class: 'card', 'aria-labelledby': 'h-time' }, h('h3', { id: 'h-time' }, 'Stage timings'), timingsBlock(r)),
      h('section', { class: 'card', 'aria-labelledby': 'h-worker' }, h('h3', { id: 'h-worker' }, 'Worker'), workerBlock(job.worker)),
    ),
    warnings.length
      ? h(
          'section',
          { class: 'card', 'aria-labelledby': 'h-warn' },
          h('h3', { id: 'h-warn' }, 'Warnings'),
          h('ul', null, warnings.map((w) => h('li', null, w))),
        )
      : null,
  );
}
