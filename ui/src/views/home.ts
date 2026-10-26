import {
  ApiError,
  episodeParams,
  isEpisodeDemo,
  newClientToken,
  rememberEpisode,
  type DemoInfo,
  type EpisodeFamily,
} from '../api/types';
import type { JobCreate, JobStatus, Mode } from '../types/contracts';
import { clear, h, uid } from '../lib/dom';
import { ALLOWED_CONTENT_TYPES, MAX_UPLOAD_BYTES, contentTypeFor, formatBytes, validateUploadFile } from '../lib/upload';
import { jobHref } from '../router';
import { errorBox, replayBadge } from './components';
import type { AppContext, View } from './context';

export const PROBLEM =
  'On repeating semiconductor arrays a CD-SEM cannot tell which identical copy it is looking at, and measuring the wrong one corrupts process-control data.';
export const TAGLINE =
  'Frequency tells you where a copy could be. Roughness tells you which one is real. The agent decides where to look next, and when to ask a human.';

const MODES: { value: Mode; label: string }[] = [
  { value: 'auto', label: 'Auto (engine decides)' },
  { value: 'fast', label: 'Fast (lattice + roughness)' },
  { value: 'accurate', label: 'Accurate (full pipeline + subpixel)' },
  { value: 'dense', label: 'Dense candidate scan' },
];

const FAMILIES: { value: EpisodeFamily; label: string }[] = [
  { value: 'dram', label: 'DRAM contact array' },
  { value: 'finfet', label: 'FinFET lines' },
  { value: 'random', label: 'Random family' },
];

function errMsg(e: unknown): string {
  if (e instanceof ApiError) return e.code ? `${e.message} (${e.code})` : e.message;
  return (e as Error)?.message ?? String(e);
}

/** One file slot with drag-and-drop + native file input. */
function fileSlot(label: string, hintId: string, disabled: boolean) {
  const id = uid('file');
  const errId = `${id}-err`;
  const input = h('input', {
    id,
    type: 'file',
    accept: '.png,.tif,.tiff,image/png,image/tiff',
    'aria-describedby': `${hintId} ${errId}`,
    disabled,
  });
  const err = h('p', { id: errId, class: 'field-error', 'aria-live': 'polite' });
  const chosen = h('p', { class: 'muted file-chosen' }, 'No file chosen.');
  let file: File | null = null;

  const set = (f: File | null) => {
    file = f;
    const problem = f ? validateUploadFile(f) : null;
    err.textContent = problem ?? '';
    chosen.textContent = f ? `${f.name} (${formatBytes(f.size)})` : 'No file chosen.';
    if (problem) input.setAttribute('aria-invalid', 'true');
    else input.removeAttribute('aria-invalid');
  };
  input.addEventListener('change', () => set(input.files?.[0] ?? null));

  const zone = h(
    'div',
    { class: 'dropzone' },
    h('label', { for: id }, label),
    input,
    h('p', { class: 'hint', 'aria-hidden': 'true' }, 'or drag a file here'),
    chosen,
    err,
  );
  zone.addEventListener('dragover', (ev) => {
    ev.preventDefault();
    zone.classList.add('is-drag');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('is-drag'));
  zone.addEventListener('drop', (ev) => {
    ev.preventDefault();
    zone.classList.remove('is-drag');
    const f = ev.dataTransfer?.files?.[0];
    if (f && !disabled) set(f);
  });

  return {
    el: zone,
    get file() {
      return file;
    },
    validate(): string | null {
      const p = validateUploadFile(file);
      err.textContent = p ?? '';
      if (p) input.setAttribute('aria-invalid', 'true');
      return p;
    },
    focus: () => input.focus(),
  };
}

export function homeView(ctx: AppContext): View {
  const offline = ctx.api.kind === 'replay';
  const heading = h('h1', { id: 'home-h', tabindex: '-1' }, 'DriftSense Pilot');

  let selected: DemoInfo | null = null;
  let busy = false;

  const go = (job: JobStatus, episode: boolean, msg: string) => {
    if (episode) rememberEpisode(job.job_id);
    ctx.announce(msg);
    ctx.navigate(jobHref(job.job_id));
  };

  /** Run `start` with the button in a busy state; errors land in `errSlot`. */
  const withBusy = async (btn: HTMLButtonElement, errSlot: HTMLElement, start: () => Promise<void>) => {
    if (busy) return;
    busy = true;
    const label = btn.textContent;
    btn.textContent = 'Starting…';
    btn.setAttribute('aria-busy', 'true');
    clear(errSlot);
    try {
      await start();
    } catch (e) {
      errSlot.append(errorBox('Could not start the run', errMsg(e)));
    } finally {
      busy = false;
      btn.textContent = label;
      btn.removeAttribute('aria-busy');
    }
  };

  // ---------- one-click agent episode ----------
  const familySelect = h(
    'select',
    { id: 'ep-family', name: 'family' },
    FAMILIES.map((f) => h('option', { value: f.value }, f.label)),
  );
  const seedInput = h('input', { id: 'ep-seed', name: 'seed', type: 'number', min: 0, max: 100000, step: 1, inputmode: 'numeric', value: '0' });
  const episodeBtn = h(
    'button',
    { type: 'button', class: 'btn btn-primary btn-hero', id: 'run-episode' },
    offline ? 'Replay a recorded agent episode' : 'Run agent episode',
  );
  const episodeErr = h('div');
  episodeBtn.addEventListener('click', () =>
    void withBusy(episodeBtn, episodeErr, async () => {
      const seed = Math.max(0, Math.floor(Number(seedInput.value) || 0));
      const body = offline
        ? { family: 'dram' as const, seed: 1, client_token: newClientToken() }
        : { family: familySelect.value as EpisodeFamily, seed, client_token: newClientToken() };
      const job = await ctx.api.createEpisode(body);
      go(job, true, 'Agent episode started.');
    }),
  );

  const episodeCard = h(
    'section',
    { class: 'card card-hero', 'aria-labelledby': 'ep-h' },
    h('h2', { id: 'ep-h' }, 'Watch the agent decide'),
    h(
      'p',
      null,
      'One click runs a full active-perception episode on a simulated SEM: acquire, localize with OpenCV, compare the uniqueness margin with τ, then commit the stage move or ask you to approve. It finishes in well under a minute.',
    ),
    offline
      ? h('p', { class: 'notice notice-replay' }, replayBadge(), ' Plays the recorded escalation episode first, so you can try the approval step. The commit episode is in the list below.')
      : h(
          'div',
          { class: 'field-row' },
          h('div', { class: 'field' }, h('label', { for: 'ep-family' }, 'Array family'), familySelect),
          h('div', { class: 'field field-seed' }, h('label', { for: 'ep-seed' }, 'Seed'), seedInput),
        ),
    h('div', { class: 'actions' }, episodeBtn),
    episodeErr,
  );

  // ---------- hackathon showcase card ----------
  const judgeCommitBtn = h(
    'button',
    { type: 'button', class: 'btn btn-judge-commit', id: 'judge-commit-btn' },
    h('span', { 'aria-hidden': 'true' }, '✔ '),
    'Case A: Autonomous Commit (margin ≥ τ)',
  );
  judgeCommitBtn.addEventListener('click', () => {
    void withBusy(judgeCommitBtn, episodeErr, async () => {
      const job = await ctx.api.createEpisode({ family: 'dram', seed: 0, client_token: newClientToken() });
      go(job, true, 'Commit episode started.');
    });
  });

  const judgeEscalateBtn = h(
    'button',
    { type: 'button', class: 'btn btn-judge-escalate', id: 'judge-escalate-btn' },
    h('span', { 'aria-hidden': 'true' }, '⚠ '),
    'Case B: Human Escalation (margin < τ)',
  );
  judgeEscalateBtn.addEventListener('click', () => {
    void withBusy(judgeEscalateBtn, episodeErr, async () => {
      const job = await ctx.api.createEpisode({ family: 'dram', seed: 1, client_token: newClientToken() });
      go(job, true, 'Escalation episode started.');
    });
  });

  const architectureDiagram = h(
    'div',
    { class: 'arch-flow', role: 'region', 'aria-label': 'Perception Pipeline Architecture' },
    h(
      'div',
      { class: 'arch-step' },
      h('div', { class: 'arch-step-num' }, '1'),
      h('div', { class: 'arch-step-title' }, 'Acquire & FFT'),
      h('div', { class: 'arch-step-desc' }, 'Frequency spectral peaks isolate candidate array period shifts.'),
    ),
    h(
      'div',
      { class: 'arch-step' },
      h('div', { class: 'arch-step-num' }, '2'),
      h('div', { class: 'arch-step-title' }, 'Roughness Fingerprint'),
      h('div', { class: 'arch-step-desc' }, 'Normalized cross-correlation over stochastic edge roughness proves identity.'),
    ),
    h(
      'div',
      { class: 'arch-step' },
      h('div', { class: 'arch-step-num' }, '3'),
      h('div', { class: 'arch-step-title' }, 'Calibrated τ Gate'),
      h('div', { class: 'arch-step-desc' }, 'Margin ≥ τ commits stage move; Margin < τ escalates to human operator.'),
    ),
  );

  const engineCard = h(
    'section',
    { class: 'platform-card', 'aria-label': 'Active Perception Engine Architecture' },
    h(
      'div',
      { class: 'platform-head' },
      h('span', { class: 'platform-badge' }, '◈ ACTIVE PERCEPTION ENGINE'),
      h('span', { class: 'platform-meta' }, 'CALIBRATED NANO-SCALE METROLOGY PIPELINE'),
    ),
    h(
      'p',
      { class: 'platform-pitch' },
      h('strong', null, 'The Nanoscale Challenge: '),
      'On repeating memory and logic arrays, CD-SEMs cannot distinguish identical copies, risking catastrophic false moves. ',
      h('strong', null, 'The Calibrated Solution: '),
      'DriftSense couples active computer vision with a Neyman-Pearson calibrated uniqueness margin τ (0.100). The agent autonomously decides where to look next, minimizes beam exposure dose, and ensures zero false commits.',
    ),
    architectureDiagram,
    h(
      'div',
      { class: 'judge-bar' },
      h('span', { class: 'judge-label' }, '⚡ Verification Scenarios:'),
      judgeCommitBtn,
      judgeEscalateBtn,
    ),
  );

  // ---------- demo picker ----------
  const picker = h('div', { class: 'cards', role: 'radiogroup', 'aria-labelledby': 'pick-h' });
  const pickerStatus = h('div', { role: 'status', class: 'muted' }, 'Loading demo sites…');
  const modeSelect = h('select', { id: 'mode', name: 'mode' }, MODES.map((m) => h('option', { value: m.value }, m.label)));
  const modeField = h('div', { class: 'field field-inline' }, h('label', { for: 'mode' }, 'Engine mode'), modeSelect);
  const runBtn = h('button', { type: 'button', class: 'btn btn-primary', id: 'run-demo', disabled: true }, 'Run localization');
  const runError = h('div');

  const syncRunButton = () => {
    const ep = !!selected && isEpisodeDemo(selected);
    runBtn.textContent = ep ? 'Run agent episode on this site' : 'Run localization';
    modeField.hidden = ep;
  };

  const renderDemos = (list: DemoInfo[]) => {

    clear(picker);
    pickerStatus.textContent = list.length ? '' : 'No demo sites are available.';
    list.forEach((d, i) => {
      const id = `demo-${d.demo_id}`;
      const radio = h('input', { type: 'radio', name: 'demo', id, value: d.demo_id, class: 'card-radio' });
      if (i === 0) {
        radio.checked = true;
        selected = d;
      }
      radio.addEventListener('change', () => {
        if (radio.checked) selected = d;
        syncRunButton();
      });
      const ep = isEpisodeDemo(d);
      picker.append(
        h(
          'div',
          { class: 'card-choice' },
          radio,
          h(
            'label',
            { for: id },
            h('span', { class: 'card-title' }, d.title),
            h(
              'span',
              { class: 'card-tags' },
              ep ? h('span', { class: 'tag tag-agent' }, 'agent episode') : null,
              ' ',
              h('span', { class: 'tag' }, d.lattice),
              ' ',
              h('span', { class: 'tag' }, `difficulty: ${d.difficulty}`),
            ),
            d.description ? h('span', { class: 'card-desc' }, d.description) : null,
          ),
        ),
      );
    });
    runBtn.disabled = !list.length;
    syncRunButton();
  };

  const loadDemos = async () => {
    pickerStatus.textContent = 'Loading demo sites…';
    clear(runError);
    try {
      renderDemos(await ctx.api.demos());
    } catch (e) {
      pickerStatus.textContent = '';
      const retry = h('button', { type: 'button', class: 'btn' }, 'Retry');
      retry.addEventListener('click', () => void loadDemos());
      clear(picker);
      picker.append(errorBox('Could not load demo sites', errMsg(e), retry));
    }
  };

  runBtn.addEventListener('click', () => {
    const d = selected;
    if (!d) return;
    void withBusy(runBtn, runError, async () => {
      if (isEpisodeDemo(d)) {
        const job = await ctx.api.createEpisode({ ...episodeParams(d), client_token: newClientToken() });
        go(job, true, 'Agent episode started.');
      } else {
        const body: JobCreate = { demo_id: d.demo_id, mode: modeSelect.value as Mode, agent: false, client_token: newClientToken() };
        go(await ctx.api.createJob(body), false, 'Run started.');
      }
    });
  });

  // ---------- upload ----------
  const tpl = fileSlot('Template image (the pattern to find)', 'upload-hint', offline);
  const search = fileSlot('Search image (the field of view)', 'upload-hint', offline);
  const uploadBtn = h('button', { type: 'button', class: 'btn', disabled: offline }, 'Upload and run localization');
  const uploadStatus = h('div', { role: 'status', class: 'muted' });
  uploadBtn.addEventListener('click', async () => {
    if (busy) return;
    const p1 = tpl.validate();
    const p2 = search.validate();
    if (p1 || p2) {
      uploadStatus.textContent = 'Fix the highlighted file problems first.';
      (p1 ? tpl : search).focus();
      return;
    }
    busy = true;
    uploadBtn.setAttribute('aria-busy', 'true');
    try {
      const keys: Record<'template' | 'search', string> = { template: '', search: '' };
      let uploadId: string | undefined;
      for (const [slot, role] of [
        [tpl, 'template'],
        [search, 'search'],
      ] as const) {
        const f = slot.file!;
        uploadStatus.textContent = `Uploading ${role} image…`;
        const ticket = await ctx.api.createUpload({
          filename: f.name,
          content_type: contentTypeFor(f),
          size_bytes: f.size,
          role,
          ...(uploadId ? { upload_id: uploadId } : {}),
        });
        uploadId ??= ticket.upload_id; // both files under one uploads/<id>/ prefix
        await ctx.api.putUpload(ticket, f);
        keys[role] = ticket.key;
      }
      uploadStatus.textContent = 'Starting run…';
      const job = await ctx.api.createJob({
        template_key: keys.template,
        search_key: keys.search,
        mode: modeSelect.value as Mode,
        agent: false,
        client_token: newClientToken(),
      });
      go(job, false, 'Upload complete. Run started.');
    } catch (e) {
      uploadStatus.textContent = '';
      uploadStatus.append(errorBox('Upload failed', errMsg(e)));
    } finally {
      busy = false;
      uploadBtn.removeAttribute('aria-busy');
    }
  });

  const el = h(
    'div',
    { class: 'view view-home' },
    h(
      'section',
      { class: 'hero', 'aria-labelledby': 'home-h' },
      h('div', { class: 'hero-tag' }, h('span', { 'aria-hidden': 'true' }, '◈ '), 'SEMICONDUCTOR CD-SEM · ACTIVE PERCEPTION AGENT'),
      heading,
      h('p', { class: 'lead' }, PROBLEM),
      h('blockquote', { class: 'tagline' }, h('p', null, TAGLINE)),
      h(
        'div',
        { class: 'kpi-grid', role: 'region', 'aria-label': 'System Specifications' },
        h(
          'div',
          { class: 'kpi-card' },
          h('div', { class: 'kpi-label' }, 'Target Resolution'),
          h('div', { class: 'kpi-val' }, '< 0.1 nm'),
          h('div', { class: 'kpi-sub' }, 'Sub-pixel 2D quadratic Taylor fit'),
        ),
        h(
          'div',
          { class: 'kpi-card' },
          h('div', { class: 'kpi-label' }, 'Uniqueness Gate τ'),
          h('div', { class: 'kpi-val' }, 'τ = 0.100'),
          h('div', { class: 'kpi-sub' }, 'Calibrated confidence threshold'),
        ),
        h(
          'div',
          { class: 'kpi-card' },
          h('div', { class: 'kpi-label' }, 'Active Tool Budget'),
          h('div', { class: 'kpi-val' }, '≤ 5 steps'),
          h('div', { class: 'kpi-sub' }, 'Minimizes electron beam dose'),
        ),
        h(
          'div',
          { class: 'kpi-card' },
          h('div', { class: 'kpi-label' }, 'Safety Guarantee'),
          h('div', { class: 'kpi-val' }, 'Zero False Commits'),
          h('div', { class: 'kpi-sub' }, 'Human-in-the-loop escalation'),
        ),
      ),
    ),
    engineCard,
    episodeCard,
    h(
      'section',
      { class: 'card', 'aria-labelledby': 'pick-h' },
      h('h2', { id: 'pick-h' }, 'Pick a demo site'),
      h('p', { class: 'muted' }, 'Synthetic sites. Sites tagged “agent episode” run the full agent loop; the others run one localization.'),
      pickerStatus,
      picker,
      modeField,
      h('div', { class: 'actions' }, runBtn),
      runError,
    ),
    h(
      'section',
      { class: 'card', 'aria-labelledby': 'upload-h' },
      h('h2', { id: 'upload-h' }, 'Or use your own images'),
      h(
        'p',
        { id: 'upload-hint', class: 'hint' },
        offline
          ? 'Uploads need the live API and are disabled in offline replay.'
          : `PNG or TIFF, up to ${formatBytes(MAX_UPLOAD_BYTES)} each (${ALLOWED_CONTENT_TYPES.join(', ')}). Files are deleted after 7 days; do not upload confidential images.`,
      ),
      h('div', { class: 'grid-2' }, tpl.el, search.el),
      h('div', { class: 'actions' }, uploadBtn),
      uploadStatus,
    ),
  );

  void loadDemos();
  return { el, heading };
}
