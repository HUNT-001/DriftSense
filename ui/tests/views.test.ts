import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mountApp, type MountedApp } from '../src/app';
import { REPLAY_QUEUED_MS, REPLAY_STEP_MS, ReplayApi } from '../src/api/replay';
import { rememberEpisode } from '../src/api/types';
import type { AppConfig } from '../src/config';
import { parseTraceJsonl } from '../src/lib/trace';
import { createTraceView, nextAction, renderTrace } from '../src/views/trace';
import { fsLoader, readSample } from './helpers/samples';

const config: AppConfig = { apiBase: '', offline: true, offlineReason: 'query' };
const tick = () => new Promise((r) => setTimeout(r, 0));
async function flush(n = 10) {
  for (let i = 0; i < n; i++) await tick();
}

let root: HTMLElement;
let app: MountedApp | null = null;
let now = 5_000_000;
let api: ReplayApi;

async function mountAt(hash: string) {
  window.location.hash = hash;
  await tick(); // let jsdom deliver hashchange before we listen
  app = mountApp(root, api, config);
  await flush();
}

/** Every form control has an accessible label; every image has non-empty alt; ids are unique. */
function assertA11yBasics(scope: HTMLElement) {
  for (const el of scope.querySelectorAll('input:not([type=hidden]), select, textarea')) {
    const id = el.getAttribute('id');
    const labelled = (id && scope.querySelector(`label[for="${id}"]`)) || el.getAttribute('aria-label') || el.closest('label');
    expect(labelled, `label for ${el.outerHTML}`).toBeTruthy();
  }
  for (const img of scope.querySelectorAll('img')) expect(img.getAttribute('alt')?.length).toBeGreaterThan(10);
  const ids = [...scope.querySelectorAll('[id]')].map((e) => e.id);
  expect(new Set(ids).size, 'duplicate ids').toBe(ids.length);
  for (const b of scope.querySelectorAll('button')) expect(b.textContent?.trim().length, b.outerHTML).toBeGreaterThan(0);
}

const commitEvents = parseTraceJsonl(readSample('demo_commit.jsonl')).events;
const escalateEvents = parseTraceJsonl(readSample('demo_escalate.jsonl')).events;

beforeEach(() => {
  document.body.innerHTML = '<div id="app"></div>';
  root = document.getElementById('app')!;
  now = 5_000_000;
  api = new ReplayApi({ loader: fsLoader, latencyMs: 0, now: () => now });
});

afterEach(() => {
  app?.dispose();
  app = null;
});

describe('shell', () => {
  it('skip link first, landmarks, live region, offline banner and version footer', async () => {
    await mountAt('#/');
    const first = root.firstElementChild as HTMLAnchorElement;
    expect(first.classList.contains('skip-link')).toBe(true);
    expect(root.querySelector('header nav[aria-label="Main"]')).toBeTruthy();
    expect(root.querySelector('main#main')).toBeTruthy();
    expect(root.querySelector('footer')).toBeTruthy();
    expect(root.querySelector('#live[aria-live="polite"]')).toBeTruthy();
    expect(root.querySelector('.demo-banner')?.textContent).toMatch(/Offline replay of a recorded trace/);
    expect(root.querySelector('.version-line')?.textContent).toMatch(/contract 1\.0\.0/);
    expect(root.querySelector('a[aria-current="page"]')?.textContent).toBe('Try a site');
  });
});

describe('home', () => {
  it('one-click episode button, demo picker with agent tags, upload form (disabled offline)', async () => {
    await mountAt('#/');
    const main = app!.main;
    expect(main.querySelector('h1')?.textContent).toBe('DriftSense Pilot');
    expect(main.querySelector('#run-episode')?.textContent).toMatch(/Replay a recorded agent episode/);
    const radios = main.querySelectorAll<HTMLInputElement>('input[type=radio][name=demo]');
    expect(radios).toHaveLength(2);
    expect(radios[0]!.checked).toBe(true);
    expect(main.querySelectorAll('.tag-agent')).toHaveLength(2);
    expect(main.querySelector('#run-demo')?.textContent).toMatch(/Run agent episode on this site/);
    expect(main.querySelectorAll('input[type=file]')).toHaveLength(2);
    expect(main.querySelector<HTMLButtonElement>('input[type=file]')!.hasAttribute('disabled')).toBe(true);
    assertA11yBasics(root);
  });

  it('one click starts an episode and navigates to the live run', async () => {
    await mountAt('#/');
    main().querySelector<HTMLButtonElement>('#run-episode')!.click();
    await flush();
    expect(window.location.hash).toMatch(/^#\/jobs\/replay-escalate-\d+$/);
  });

  it('live API home shows family + seed inputs', async () => {
    const live = { ...api, kind: 'live' } as unknown as ReplayApi;
    Object.setPrototypeOf(live, ReplayApi.prototype);
    Object.defineProperty(live, 'kind', { value: 'live' });
    api = live;
    await mountAt('#/');
    expect(main().querySelector('#ep-family')).toBeTruthy();
    expect(main().querySelector('#ep-seed')).toBeTruthy();
    expect(main().querySelector('#run-episode')?.textContent).toBe('Run agent episode');
  });
});

function main() {
  return app!.main;
}

describe('agent timeline (margin vs tau drives the next action)', () => {
  it('commit trace: margin bar shows commit, next action is commit_stage_move, because is highlighted', () => {
    const el = renderTrace({ events: commitEvents, errors: [], source: 'live' });
    const rows = el.querySelectorAll('.timeline > li');
    expect(rows).toHaveLength(commitEvents.length);
    const loc = el.querySelector('li[data-step="7"]')!;
    expect(loc.querySelector('.mbar-track[role=img]')?.getAttribute('aria-label')).toMatch(/0\.469 is at or above the threshold tau 0\.100: commit allowed/);
    expect(loc.querySelector('.mbar-verdict')?.textContent).toBe('commit allowed');
    expect(loc.querySelector('.tl-next strong')?.textContent).toBe('commit_stage_move');
    const cm = el.querySelector('li.tl-commit_move')!;
    expect(cm.classList.contains('tl-key')).toBe(true);
    expect(cm.querySelector('.tl-because.is-key q')?.textContent).toBe('margin 0.469 >= tau 0.100');
    expect(cm.textContent).toMatch(/Stage move committed: dx 1\.06 nm, dy -1\.15 nm/);
    const card = el.querySelector('.decision-card')!;
    expect(card.classList.contains('is-commit')).toBe(true);
    expect(card.textContent).toMatch(/margin 0\.469 ≥ τ 0\.100 → commit_stage_move/);
    // Dose bar on acquire results; latency on the localize result.
    expect(el.querySelectorAll('.vbar.dose')).toHaveLength(3);
    expect(el.querySelector('li[data-step="5"] .vbar-value')?.textContent).toMatch(/47\.06 M \(of episode total 59\.56 M\)/);
    expect(loc.querySelector('.tl-latency')?.textContent).toMatch(/294 ms/);
    // Post-commit verification below tau reads as "re-acquire or escalate", not a commit.
    expect(el.querySelector('li[data-step="14"] .mbar-verdict')?.textContent).toBe('re-acquire or escalate');
  });

  it('escalate trace: below tau, escalate row highlighted with the budget reason', () => {
    const el = renderTrace({ events: escalateEvents, errors: [], source: 'replay' });
    expect(el.querySelector('.notice-replay')?.textContent).toMatch(/Offline replay of a recorded trace/);
    const loc = el.querySelector('li[data-step="7"]')!;
    expect(loc.querySelector('.mbar-text')?.textContent).toMatch(/margin -0\.026 < τ 0\.100/);
    expect(loc.querySelector('.tl-next strong')?.textContent).toBe('request_human_approval');
    const esc = el.querySelector('li.tl-escalate')!;
    expect(esc.classList.contains('tl-key')).toBe(true);
    expect(esc.querySelector('.tl-because.is-key q')?.textContent).toBe('budget exhausted at margin -0.026 < tau 0.100');
    expect(esc.querySelector('.mbar-verdict')?.textContent).toBe('escalate to a human');
    expect(el.querySelector('.decision-card')?.classList.contains('is-escalate')).toBe(true);
    expect(el.querySelector('.decision-card')?.textContent).toMatch(/stayed below τ/);
    expect(nextAction(escalateEvents, 7)?.tool).toBe('request_human_approval');
  });

  it('appends new steps incrementally and announces them in a polite live region', () => {
    const v = createTraceView();
    v.update({ events: commitEvents.slice(0, 4), errors: [], source: 'live' });
    expect(v.el.querySelector('li[data-step="3"] .vbar-value')?.textContent).toMatch(/of episode total 34.56 M/);
    const live = v.el.querySelector('[data-testid="trace-live"]')!;
    expect(live.getAttribute('aria-live')).toBe('polite');
    const firstRow = v.el.querySelector('.timeline > li')!;
    v.update({ events: commitEvents.slice(0, 11), errors: [], source: 'live' });
    expect(v.el.querySelector('.timeline > li')).toBe(firstRow); // not re-rendered
    expect(v.el.querySelectorAll('.timeline > li')).toHaveLength(11);
    expect(v.el.querySelectorAll('.tl-new').length).toBeGreaterThan(0);
    // Dose bars of already-rendered rows follow the new episode total.
    expect(v.el.querySelector('li[data-step="3"] .vbar-value')?.textContent).toMatch(/of episode total 47\.06 M/);
    expect(live.textContent).toMatch(/Step 10: Commit stage move, commit_stage_move, margin 0\.469 at or above tau 0\.100, because margin 0\.469 >= tau 0\.100\./);
  });
});

describe('live run (offline replay of the real traces)', () => {
  it('commit episode: timeline, result, NCC vs roughness images with alt text, episode summary', async () => {
    const j = await api.createEpisode({ family: 'dram', seed: 0 });
    rememberEpisode(j.job_id);
    now += 60_000;
    await mountAt(`#/jobs/${j.job_id}`);
    await vi.waitFor(() => expect(main().querySelectorAll('.timeline > li')).toHaveLength(17));
    const m = main();
    expect(m.querySelector('h1')?.textContent).toMatch(/Agent episode/);
    expect(m.querySelector('.pill')?.textContent).toMatch(/Succeeded/);
    expect(m.querySelector('[data-testid=outcome]')?.textContent).toMatch(/unique match, stage correction committed/);
    expect(m.querySelector('.view-job > .notice-replay, .view-job .notice-replay')?.textContent).toMatch(/demo_commit\.jsonl/);
    expect(m.querySelector('table.data caption')?.textContent).toBe('Ranked candidates');
    expect(m.querySelectorAll('table.data tbody tr')).toHaveLength(5);
    expect(m.querySelector('.chip-commit')?.textContent).toMatch(/Commit/);
    const figs = [...m.querySelectorAll('.gallery figcaption')].map((f) => f.textContent);
    expect(figs[0]).toMatch(/NCC heatmap/);
    expect(figs[1]).toMatch(/Roughness heatmap/);
    expect(m.querySelector<HTMLImageElement>('.gallery img')!.getAttribute('src')).toBe('samples/demo_commit/step7_ncc_heatmap.webp');
    await vi.waitFor(() => expect(m.textContent).toMatch(/Offset \(nm\)≈ x 203\.06, y 200\.84/));
    await vi.waitFor(() => expect(m.querySelector('#ep-h')?.textContent).toMatch(/episode\.json/));
    expect(m.querySelector('.responsible')?.textContent).toMatch(/7-day retention/);
    assertA11yBasics(root);
  });

  it('escalate episode: approval card -> name required -> confirm -> POST payload -> outcome', async () => {
    const j = await api.createEpisode({ family: 'dram', seed: 1 });
    rememberEpisode(j.job_id);
    now += 60_000;
    const approveSpy = vi.spyOn(api, 'approve');
    await mountAt(`#/jobs/${j.job_id}`);
    const card = main().querySelector('section.approval') as HTMLElement;
    expect(card).toBeTruthy();
    expect(document.activeElement?.id).toBe('approval-h'); // focus moved to the decision
    expect(card.textContent).toMatch(/SIMULATED APPROVAL/);
    expect(card.textContent).toMatch(/margin -0\.026 is below the commit threshold τ = 0\.100/);
    expect(card.querySelector('.mbar-verdict')?.textContent).toBe('escalate to a human');
    const approve = [...card.querySelectorAll('button')].filter((b) => /^Approve rank/.test(b.textContent ?? ''));
    expect(approve.map((b) => b.textContent)).toEqual(['Approve rank 1', 'Approve rank 2', 'Approve rank 3']);
    expect(card.textContent).toMatch(/fused score 0\.674/);
    expect(card.textContent).toMatch(/roughness 0\.621/);
    assertA11yBasics(root);

    const name = card.querySelector<HTMLInputElement>('#approver')!;
    name.value = '';
    approve[1]!.click();
    expect(name.getAttribute('aria-invalid')).toBe('true');
    expect(card.querySelector('#approver-err')?.textContent).toMatch(/Enter your name/);
    expect((card.querySelector('.confirm') as HTMLElement).hidden).toBe(true);
    expect(document.activeElement).toBe(name);

    name.value = 'Ada';
    card.querySelector<HTMLTextAreaElement>('#approval-note')!.value = 'rank 2 matches the roughness map';
    approve[1]!.click();
    const confirm = card.querySelector('.confirm') as HTMLElement;
    expect(confirm.hidden).toBe(false);
    expect(document.activeElement?.id).toBe('confirm-h');
    expect(confirm.textContent).toMatch(/Approve rank 2/);
    // Escape cancels and returns focus to the trigger.
    confirm.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    expect(confirm.hidden).toBe(true);
    expect(document.activeElement).toBe(approve[1]);

    approve[1]!.click();
    [...card.querySelectorAll<HTMLButtonElement>('.confirm button')].find((b) => b.textContent === 'Confirm')!.click();
    await flush(12);
    expect(approveSpy).toHaveBeenCalledWith(j.job_id, { chosen_rank: 2, approver: 'Ada', note: 'rank 2 matches the roughness map' });
    expect(main().querySelector('section.approval')).toBeNull();
    expect(main().querySelector('.notice-ok[role=status]')?.textContent).toMatch(/rank 2 approved by Ada/);

    now += 10 * REPLAY_STEP_MS;
    await vi.waitFor(() => expect(main().querySelector('.pill')?.textContent).toMatch(/Succeeded/), { timeout: 4000 });
    await vi.waitFor(() => expect(main().querySelector('.timeline > li:last-child')?.textContent).toMatch(/Episode end/), { timeout: 3000 });
    expect(main().querySelector('[data-testid=outcome]')?.textContent).toMatch(/operator's choice \(rank 2\)/);
    expect(main().querySelectorAll('.badge-sim').length).toBe(4);
    expect(document.activeElement?.getAttribute('data-testid')).toBe('outcome');
  });

  it('reject all ends as "Rejected by operator"', async () => {
    const j = await api.createEpisode({ family: 'dram', seed: 1 });
    rememberEpisode(j.job_id);
    now += 60_000;
    await mountAt(`#/jobs/${j.job_id}`);
    const card = main().querySelector('section.approval') as HTMLElement;
    card.querySelector<HTMLInputElement>('#approver')!.value = 'Ada';
    [...card.querySelectorAll('button')].find((b) => /Reject all/.test(b.textContent ?? ''))!.click();
    [...card.querySelectorAll<HTMLButtonElement>('.confirm button')].find((b) => b.textContent === 'Confirm')!.click();
    await flush(12);
    now += 10 * REPLAY_STEP_MS;
    await vi.waitFor(() => expect(main().querySelector('[data-testid=outcome]')?.textContent).toMatch(/Rejected by operator/), { timeout: 4000 });
  });

  it('a running episode shows queued then streams steps', async () => {
    const j = await api.createEpisode({ family: 'dram', seed: 0 });
    rememberEpisode(j.job_id);
    await mountAt(`#/jobs/${j.job_id}`);
    expect(main().querySelector('.pill')?.textContent).toMatch(/Queued/);
    now += REPLAY_QUEUED_MS + 2000;
    await vi.waitFor(() => expect(main().querySelectorAll('.timeline > li').length).toBeGreaterThan(1), { timeout: 3000 });
    expect(main().querySelector('.decision-card')?.textContent).toMatch(/Agent is working/);
  });

  it('shows a not-found state for unknown jobs', async () => {
    await mountAt('#/jobs/does-not-exist');
    await flush();
    expect(main().textContent).toMatch(/Job not found/);
  });
});

describe('static pages', () => {
  it('about explains commit-vs-escalate and states no numbers', async () => {
    await mountAt('#/about');
    const t = main().textContent ?? '';
    expect(main().querySelector('h1')?.textContent).toBe('About DriftSense Pilot');
    expect(t).toMatch(/only if margin ≥ τ/);
    expect(t).not.toMatch(/\d+(\.\d+)?\s*(%|x\b|ms\b|times faster)/);
    expect(document.title).toMatch(/About DriftSense Pilot/);
  });

  it('responsible use lists the five commitments', async () => {
    await mountAt('#/responsible');
    const t = main().textContent ?? '';
    for (const re of [/Synthetic data only/, /Human in the loop/, /Dose limits/, /7-day retention/, /Not validated on production tools/]) {
      expect(t).toMatch(re);
    }
  });

  it('unknown routes render a not-found page', async () => {
    await mountAt('#/missing');
    expect(main().querySelector('h1')?.textContent).toBe('Page not found');
  });
});
