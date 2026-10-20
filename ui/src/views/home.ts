import { ApiError, newClientToken, type DemoSite } from '../api/types';
import type { JobCreate, Mode } from '../types/contracts';
import { clear, h, uid } from '../lib/dom';
import { ALLOWED_CONTENT_TYPES, MAX_UPLOAD_BYTES, contentTypeFor, formatBytes, validateUploadFile } from '../lib/upload';
import { jobHref } from '../router';
import { errorBox } from './components';
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

function errMsg(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  return (e as Error)?.message ?? String(e);
}

/** One file slot with drag-and-drop + native file input. */
function fileSlot(label: string, hintId: string) {
  const id = uid('file');
  const errId = `${id}-err`;
  const input = h('input', {
    id,
    type: 'file',
    accept: '.png,.tif,.tiff,image/png,image/tiff',
    'aria-describedby': `${hintId} ${errId}`,
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
    if (f) set(f);
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
  const heading = h('h1', { id: 'home-h', tabindex: '-1' }, 'DriftSense Pilot');
  let selected: string | null = null;
  let busy = false;

  // ---------- demo picker ----------
  const picker = h('div', { class: 'cards', role: 'radiogroup', 'aria-labelledby': 'pick-h' });
  const pickerStatus = h('div', { role: 'status', class: 'muted' }, 'Loading demo sites…');
  const modeSelect = h(
    'select',
    { id: 'mode', name: 'mode' },
    MODES.map((m) => h('option', { value: m.value }, m.label)),
  );
  const runBtn = h('button', { type: 'button', class: 'btn btn-primary', disabled: true }, 'Run localization');
  const agentBtn = h('button', { type: 'button', class: 'btn btn-primary', disabled: true }, 'Run agent episode');
  const runError = h('div');

  const renderDemos = (demos: DemoSite[]) => {
    clear(picker);
    pickerStatus.textContent = demos.length ? '' : 'No demo sites are available.';
    demos.forEach((d, i) => {
      const id = `demo-${d.id}`;
      const radio = h('input', { type: 'radio', name: 'demo', id, value: d.id, class: 'card-radio' });
      if (i === 0) {
        radio.checked = true;
        selected = d.id;
      }
      radio.addEventListener('change', () => {
        if (radio.checked) selected = d.id;
      });
      picker.append(
        h(
          'div',
          { class: 'card-choice' },
          radio,
          h(
            'label',
            { for: id },
            h('span', { class: 'card-title' }, d.title),
            h('span', { class: 'card-tags' }, h('span', { class: 'tag' }, d.family), ' ', h('span', { class: 'tag' }, `difficulty: ${d.difficulty}`)),
            h('span', { class: 'card-desc' }, d.description),
          ),
        ),
      );
    });
    runBtn.disabled = agentBtn.disabled = !demos.length;
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

  const submitJob = async (body: JobCreate, btn: HTMLButtonElement) => {
    if (busy) return;
    busy = true;
    const label = btn.textContent;
    btn.textContent = 'Starting…';
    btn.setAttribute('aria-busy', 'true');
    clear(runError);
    try {
      const job = await ctx.api.createJob({ ...body, client_token: newClientToken() });
      ctx.announce('Run started.');
      ctx.navigate(jobHref(job.job_id));
    } catch (e) {
      runError.append(errorBox('Could not start the run', errMsg(e)));
    } finally {
      busy = false;
      btn.textContent = label;
      btn.removeAttribute('aria-busy');
    }
  };

  const demoJob = (agent: boolean): JobCreate => ({
    demo_id: selected,
    mode: modeSelect.value as Mode,
    agent,
  });
  runBtn.addEventListener('click', () => selected && void submitJob(demoJob(false), runBtn));
  agentBtn.addEventListener('click', () => selected && void submitJob(demoJob(true), agentBtn));

  // ---------- upload ----------
  const tpl = fileSlot('Template image (the pattern to find)', 'upload-hint');
  const search = fileSlot('Search image (the field of view)', 'upload-hint');
  const uploadBtn = h('button', { type: 'button', class: 'btn' }, 'Upload and run localization');
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
      const keys: string[] = [];
      for (const [slot, name] of [
        [tpl, 'template'],
        [search, 'search'],
      ] as const) {
        const f = slot.file!;
        uploadStatus.textContent = `Uploading ${name} image…`;
        const ticket = await ctx.api.createUpload({ filename: f.name, content_type: contentTypeFor(f), size_bytes: f.size });
        await ctx.api.putUpload(ticket, f);
        keys.push(ticket.key);
      }
      uploadStatus.textContent = 'Starting run…';
      const job = await ctx.api.createJob({
        template_key: keys[0],
        search_key: keys[1],
        mode: modeSelect.value as Mode,
        agent: false,
        client_token: newClientToken(),
      });
      ctx.announce('Upload complete. Run started.');
      ctx.navigate(jobHref(job.job_id));
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
      heading,
      h('p', { class: 'lead' }, PROBLEM),
      h('blockquote', { class: 'tagline' }, h('p', null, TAGLINE)),
    ),
    h(
      'section',
      { class: 'card', 'aria-labelledby': 'pick-h' },
      h('h2', { id: 'pick-h' }, 'Try a site'),
      h('p', { class: 'muted' }, 'Pick a synthetic demo site, then run a single localization or a full agent episode.'),
      pickerStatus,
      picker,
      h('div', { class: 'field field-inline' }, h('label', { for: 'mode' }, 'Engine mode'), modeSelect),
      h(
        'div',
        { class: 'actions' },
        runBtn,
        agentBtn,
      ),
      h(
        'p',
        { class: 'hint' },
        'Localization returns ranked candidates and a uniqueness margin. The agent episode can re-image, change fingerprint, and commit a stage move only when the margin clears τ; otherwise it asks you.',
      ),
      runError,
    ),
    h(
      'section',
      { class: 'card', 'aria-labelledby': 'upload-h' },
      h('h2', { id: 'upload-h' }, 'Or use your own images'),
      h(
        'p',
        { id: 'upload-hint', class: 'hint' },
        `PNG or TIFF, up to ${formatBytes(MAX_UPLOAD_BYTES)} each (${ALLOWED_CONTENT_TYPES.join(', ')}). Files are kept for 7 days.`,
      ),
      h('div', { class: 'grid-2' }, tpl.el, search.el),
      h('div', { class: 'actions' }, uploadBtn),
      uploadStatus,
    ),
  );

  void loadDemos();
  return { el, heading };
}
