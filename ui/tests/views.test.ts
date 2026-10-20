import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mountApp, type MountedApp } from '../src/app';
import { MOCK_EVENT_MS, MOCK_QUEUED_MS, MockApi } from '../src/api/mock';
import type { AppConfig } from '../src/config';

const config: AppConfig = { apiBase: '', mock: true, mockReason: 'no-api-base' };
const tick = () => new Promise((r) => setTimeout(r, 0));
async function flush(n = 8) {
  for (let i = 0; i < n; i++) await tick();
}

let root: HTMLElement;
let app: MountedApp | null = null;
let now = 5_000_000;
let api: MockApi;

async function mountAt(hash: string) {
  window.location.hash = hash;
  await tick(); // let jsdom deliver hashchange before we listen
  app = mountApp(root, api, config);
  await flush();
}

/** Every form control has an accessible label; every image has non-empty alt. */
function assertA11yBasics(scope: HTMLElement) {
  for (const el of scope.querySelectorAll('input:not([type=hidden]), select, textarea')) {
    const id = el.getAttribute('id');
    const labelled = (id && scope.querySelector(`label[for="${id}"]`)) || el.getAttribute('aria-label') || el.closest('label');
    expect(labelled, `label for ${el.outerHTML}`).toBeTruthy();
  }
  for (const img of scope.querySelectorAll('img')) expect(img.getAttribute('alt')?.length).toBeGreaterThan(10);
  const ids = [...scope.querySelectorAll('[id]')].map((e) => e.id);
  expect(new Set(ids).size, 'duplicate ids').toBe(ids.length);
}

beforeEach(() => {
  document.body.innerHTML = '<div id="app"></div>';
  root = document.getElementById('app')!;
  now = 5_000_000;
  api = new MockApi({ latencyMs: 0, now: () => now });
});

afterEach(() => {
  app?.dispose();
  app = null;
});

describe('shell', () => {
  it('has a skip link first, landmarks, live region, demo-mode banner and version footer', async () => {
    await mountAt('#/');
    const first = root.firstElementChild as HTMLAnchorElement;
    expect(first.classList.contains('skip-link')).toBe(true);
    expect(root.querySelector('header nav[aria-label="Main"]')).toBeTruthy();
    expect(root.querySelector('main#main')).toBeTruthy();
    expect(root.querySelector('footer')).toBeTruthy();
    expect(root.querySelector('[aria-live="polite"]')).toBeTruthy();
    expect(root.querySelector('.demo-banner')?.textContent).toMatch(/synthetic/);
    expect(root.querySelector('.version-line')?.textContent).toMatch(/contract 1\.0\.0/);
    expect(root.querySelector('.worker-line')?.textContent).toMatch(/COOL worker: not known/);
    expect(root.querySelector('a[aria-current="page"]')?.textContent).toBe('Try a site');
  });
});

describe('home', () => {
  it('renders problem, tagline, keyboard-selectable demo cards and upload panel', async () => {
    await mountAt('#/');
    const main = app!.main;
    expect(main.querySelector('h1')?.textContent).toBe('DriftSense Pilot');
    expect(main.querySelector('.tagline')?.textContent).toMatch(/Roughness tells you which one is real/);
    const radios = main.querySelectorAll<HTMLInputElement>('input[type=radio][name=demo]');
    expect(radios).toHaveLength(4);
    expect(radios[0]!.checked).toBe(true);
    expect(main.querySelector('[role=radiogroup]')).toBeTruthy();
    expect(main.querySelectorAll('input[type=file]')).toHaveLength(2);
    const buttons = [...main.querySelectorAll('button')].map((b) => b.textContent);
    expect(buttons).toEqual(expect.arrayContaining(['Run localization', 'Run agent episode', 'Upload and run localization']));
    assertA11yBasics(root);
  });

  it('shows validation messages when uploading without files', async () => {
    await mountAt('#/');
    const btn = [...app!.main.querySelectorAll('button')].find((b) => b.textContent === 'Upload and run localization')!;
    btn.click();
    await flush();
    const errs = [...app!.main.querySelectorAll('.field-error')].map((e) => e.textContent);
    expect(errs.filter((t) => /Choose a file/.test(t ?? ''))).toHaveLength(2);
    expect(app!.main.querySelectorAll('[aria-invalid="true"]')).toHaveLength(2);
  });

  it('starts a run and navigates to the live run page', async () => {
    await mountAt('#/');
    const btn = [...app!.main.querySelectorAll('button')].find((b) => b.textContent === 'Run localization')!;
    btn.click();
    await flush();
    expect(window.location.hash).toBe('#/jobs/mock-job-0001');
  });
});

describe('live run', () => {
  it('renders status, result panel, candidates, timings, worker and artifacts', async () => {
    const j = await api.createJob({ demo_id: 'demo_finfet_easy' });
    now += 10_000;
    await mountAt(`#/jobs/${j.job_id}`);
    const main = app!.main;
    expect(main.querySelector('.pill')?.textContent).toMatch(/Succeeded/);
    expect(main.querySelector('table.data caption')?.textContent).toBe('Ranked candidates');
    expect(main.querySelectorAll('table.data tbody tr').length).toBeGreaterThanOrEqual(1);
    expect(main.querySelector('.chip-commit')?.textContent).toMatch(/Commit/);
    expect(main.querySelector('.mbar-track[role=img]')?.getAttribute('aria-label')).toMatch(/at or above the threshold tau/);
    expect(main.querySelectorAll('.timings li').length).toBeGreaterThan(0);
    expect(main.textContent).toMatch(/COOL worker/);
    expect(main.querySelectorAll('.gallery img')).toHaveLength(3);
    expect(root.querySelector('.worker-line')?.textContent).toBe('COOL worker: no (in-browser mock)');
    assertA11yBasics(root);
  });

  it('shows the agent timeline with because, margin vs tau and cumulative dose, labelled synthetic', async () => {
    const j = await api.createJob({ demo_id: 'demo_finfet_easy', agent: true });
    now += MOCK_QUEUED_MS + MOCK_EVENT_MS * 30;
    await mountAt(`#/jobs/${j.job_id}`);
    const main = app!.main;
    const items = main.querySelectorAll('.timeline > li');
    expect(items.length).toBe(10);
    expect(main.querySelector('.badge-synthetic')?.textContent).toMatch(/SYNTHETIC SAMPLE/);
    expect(main.textContent).toMatch(/margin below tau/);
    expect(main.querySelectorAll('.timeline .mbar').length).toBeGreaterThan(3);
    expect(main.textContent).toMatch(/Cumulative dose/);
    expect(main.textContent).toMatch(/Stage move: dx -37\.5 nm, dy 12\.0 nm/);
  });

  it('approval card: requires a name, confirms, posts the decision and resumes', async () => {
    const j = await api.createJob({ demo_id: 'demo_dram_hard', agent: true });
    now += MOCK_QUEUED_MS + MOCK_EVENT_MS * 30;
    await mountAt(`#/jobs/${j.job_id}`);
    const main = app!.main;
    const card = main.querySelector('section.approval') as HTMLElement;
    expect(card).toBeTruthy();
    expect(card.textContent).toMatch(/below the commit threshold/);
    const approve = [...card.querySelectorAll('button')].filter((b) => /^Approve #/.test(b.textContent ?? ''));
    expect(approve.map((b) => b.textContent)).toEqual(['Approve #1', 'Approve #2', 'Approve #3']);
    expect([...card.querySelectorAll('button')].some((b) => b.textContent === 'Reject all')).toBe(true);
    assertA11yBasics(root);

    const name = card.querySelector<HTMLInputElement>('#approver')!;
    name.value = '';
    approve[1]!.click();
    expect(name.getAttribute('aria-invalid')).toBe('true');
    expect(card.querySelector('#approver-err')?.textContent).toMatch(/Enter your name/);
    expect((card.querySelector('.confirm') as HTMLElement).hidden).toBe(true);

    name.value = 'Ada';
    approve[1]!.click();
    const confirm = card.querySelector('.confirm') as HTMLElement;
    expect(confirm.hidden).toBe(false);
    expect(confirm.textContent).toMatch(/Approve candidate #2/);
    [...confirm.querySelectorAll('button')].find((b) => b.textContent === 'Confirm')!.click();
    await flush(12);
    expect(main.querySelector('.pill')?.textContent).toMatch(/Succeeded/);
    expect(main.querySelector('section.approval')).toBeNull();
    expect(main.querySelector('.notice-ok')?.textContent).toMatch(/candidate #2 approved/);
    expect(main.querySelector('.timeline > li:last-child')?.textContent).toMatch(/operator approved candidate #2/);
  });

  it('shows a not-found state for unknown jobs', async () => {
    await mountAt('#/jobs/does-not-exist');
    await flush();
    expect(app!.main.textContent).toMatch(/Job not found/);
  });
});

describe('static pages', () => {
  it('about explains commit-vs-escalate and states no numbers', async () => {
    await mountAt('#/about');
    const t = app!.main.textContent ?? '';
    expect(app!.main.querySelector('h1')?.textContent).toBe('About DriftSense Pilot');
    expect(t).toMatch(/only if margin ≥ τ/);
    expect(t).not.toMatch(/\d+(\.\d+)?\s*(%|x\b|ms\b|times faster)/);
    expect(document.title).toMatch(/About DriftSense Pilot/);
  });

  it('responsible use lists the five commitments', async () => {
    await mountAt('#/responsible');
    const t = app!.main.textContent ?? '';
    for (const re of [/Synthetic data only/, /Human in the loop/, /Dose limits/, /7-day retention/, /Not validated on production tools/]) {
      expect(t).toMatch(re);
    }
  });

  it('unknown routes render a not-found page', async () => {
    await mountAt('#/missing');
    expect(app!.main.querySelector('h1')?.textContent).toBe('Page not found');
  });
});
