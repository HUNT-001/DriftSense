import type { Api } from '../api/types';
import type { JobStatus, LocalizeResult, WorkerInfo } from '../types/contracts';
import { artifactAlt, artifactTitle, isImageArtifact } from '../lib/artifacts';
import { h } from '../lib/dom';
import { fmt, fmtMs, viridis } from '../lib/format';
import { decisionChip, dl, marginBar, marginDomain, valueBar } from './components';

function offsetBlock(r: LocalizeResult): HTMLElement {
  const o = r.offset;
  const rows: [string, string][] = [['Offset (px)', `x ${fmt(o.x_px, 2)}, y ${fmt(o.y_px, 2)}`]];
  if (typeof o.x_nm === 'number' && typeof o.y_nm === 'number') {
    rows.push(['Offset (nm)', `x ${fmt(o.x_nm, 2)}, y ${fmt(o.y_nm, 2)}`]);
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
      ? `Margin ${fmt(m.margin)} ≥ τ ${fmt(m.tau)}: the best candidate is unique enough to commit the stage move.`
      : `Margin ${fmt(m.margin)} < τ ${fmt(m.tau)}: uniqueness is not proven, so the stage move needs an operator.`;
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
  const arts = job.result?.artifacts ?? [];
  if (!arts.length) return null;
  return h(
    'ul',
    { class: 'gallery' },
    arts.map((a) => {
      const url = api.artifactUrl(job.job_id, a);
      if (!isImageArtifact(a)) {
        return h('li', null, h('a', { href: url, rel: 'noopener', target: '_blank' }, `${artifactTitle(a)} (JSON, opens in a new tab)`));
      }
      const img = h('img', { src: url, alt: artifactAlt(a, job), loading: 'lazy', decoding: 'async' });
      img.addEventListener('error', () => {
        img.replaceWith(h('p', { class: 'muted img-missing' }, `${artifactTitle(a)} could not be loaded.`));
      });
      return h('li', null, h('figure', null, img, h('figcaption', null, artifactTitle(a))));
    }),
  );
}

export function renderResult(api: Api, job: JobStatus): HTMLElement {
  const r = job.result!;
  const warnings = r.warnings ?? [];
  return h(
    'div',
    { class: 'result' },
    h(
      'div',
      { class: 'grid-2' },
      h('section', { class: 'card', 'aria-labelledby': 'h-offset' }, h('h3', { id: 'h-offset' }, 'Offset'), offsetBlock(r)),
      h('section', { class: 'card', 'aria-labelledby': 'h-margin' }, h('h3', { id: 'h-margin' }, 'Uniqueness margin vs τ'), marginBlock(r)),
    ),
    h('section', { class: 'card', 'aria-labelledby': 'h-cands' }, h('h3', { id: 'h-cands' }, 'Candidates'), candidatesTable(r)),
    (() => {
      const g = artifactGallery(api, job);
      return g
        ? h('section', { class: 'card', 'aria-labelledby': 'h-art' }, h('h3', { id: 'h-art' }, 'Heatmaps and overlays'), g)
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
