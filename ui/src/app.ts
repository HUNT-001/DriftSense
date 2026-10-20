import type { Api, VersionInfo } from './api/types';
import type { AppConfig } from './config';
import type { WorkerInfo } from './types/contracts';
import { clear, h } from './lib/dom';
import { shortSha } from './lib/format';
import { parseRoute, type Route } from './router';
import type { AppContext, View } from './views/context';
import { homeView } from './views/home';
import { jobView } from './views/job';
import { aboutView, notFoundView, responsibleView } from './views/static';

const NAV: { href: string; label: string; route: Route['name'] }[] = [
  { href: '#/', label: 'Try a site', route: 'home' },
  { href: '#/about', label: 'About', route: 'about' },
  { href: '#/responsible', label: 'Responsible use', route: 'responsible' },
];

export interface MountedApp {
  ctx: AppContext;
  main: HTMLElement;
  render(): View;
  dispose(): void;
}

export function versionText(v: VersionInfo): string {
  const parts = [
    `service ${v.service_version}`,
    `contract ${v.contract_version}`,
    `git ${shortSha(v.git_sha)}`,
    v.cv2_version ? `OpenCV ${v.cv2_version}` : null,
    `Python ${v.python}`,
    v.arch,
  ];
  return parts.filter(Boolean).join(' · ');
}

export function workerText(w: WorkerInfo | null | undefined): string {
  if (!w) return 'COOL worker: not known yet (run a job)';
  const where = [w.arch, w.instance_type].filter((x) => x && x !== 'unknown').join(', ');
  return `COOL worker: ${w.cool ? 'yes' : 'no'}${where ? ` (${where})` : ''}`;
}

/** Build the page shell into `root` and start the hash router. */
export function mountApp(root: HTMLElement, api: Api, config: AppConfig, win: Window = window): MountedApp {
  clear(root);
  const live = h('div', { id: 'live', class: 'sr-only', 'aria-live': 'polite', 'aria-atomic': 'true' });
  const main = h('main', { id: 'main', tabindex: '-1' });
  const skip = h('a', { class: 'skip-link', href: '#main' }, 'Skip to main content');
  skip.addEventListener('click', (ev) => {
    ev.preventDefault(); // keep the hash router's route
    main.focus();
  });
  const navLinks = NAV.map((n) => h('a', { href: n.href, 'data-route': n.route }, n.label));
  const versionLine = h('p', { class: 'version-line' }, 'API version: loading…');
  const workerLine = h('p', { class: 'worker-line' }, workerText(null));

  const modeLink = config.mock
    ? h(
        'p',
        null,
        'Demo mode: responses come from an in-browser mock. ',
        config.apiBase ? h('a', { href: win.location.pathname }, 'Switch to the live API') : null,
      )
    : h('p', null, h('a', { href: `${win.location.pathname}?mock=1#/` }, 'Try demo mode (no backend, synthetic data)'));

  const banner = config.mock
    ? h(
        'div',
        { class: 'demo-banner', role: 'note' },
        h('span', { 'aria-hidden': 'true' }, '⚑ '),
        h('strong', null, 'Demo mode. '),
        config.mockReason === 'query'
          ? 'You asked for the offline demo: '
          : config.mockReason === 'config-error'
            ? 'The API configuration could not be loaded, so the offline demo is running: '
            : 'No API is configured, so the offline demo is running: ',
        'every result, trace and number on screen is synthetic (engine "fake"), not a measurement.',
      )
    : null;

  root.append(
    skip,
    banner ?? '',
    h(
      'header',
      { class: 'site-header' },
      h('div', { class: 'wrap header-inner' }, h('a', { class: 'brand', href: '#/' }, 'DriftSense Pilot'), h('nav', { 'aria-label': 'Main' }, h('ul', null, navLinks.map((a) => h('li', null, a))))),
    ),
    h('div', { class: 'wrap' }, main),
    h('footer', { class: 'site-footer' }, h('div', { class: 'wrap' }, versionLine, workerLine, modeLink)),
    live,
  );

  let announceTimer: ReturnType<typeof setTimeout> | undefined;
  const ctx: AppContext = {
    api,
    config,
    announce(message: string) {
      live.textContent = '';
      clearTimeout(announceTimer);
      // Re-set after a tick so identical consecutive messages are still announced.
      announceTimer = setTimeout(() => (live.textContent = message), 30);
    },
    navigate(hash: string) {
      if (win.location.hash === hash) render();
      else win.location.hash = hash;
    },
    setWorker(worker) {
      workerLine.textContent = workerText(worker);
    },
  };

  api
    .version()
    .then((v) => {
      versionLine.textContent = `API: ${versionText(v)}`;
      if (v.cv2_file) versionLine.append(h('span', { class: 'muted' }, ` · cv2 from ${v.cv2_file}`));
    })
    .catch(() => {
      versionLine.textContent = 'API version: unavailable';
    });

  let current: View | null = null;
  let first = true;

  function render(): View {
    current?.dispose?.();
    const route = parseRoute(win.location.hash);
    let view: View;
    switch (route.name) {
      case 'home':
        view = homeView(ctx);
        break;
      case 'job':
        view = jobView(ctx, route.id);
        break;
      case 'about':
        view = aboutView(ctx);
        break;
      case 'responsible':
        view = responsibleView(ctx);
        break;
      default:
        view = notFoundView(ctx, route.path);
    }
    current = view;
    clear(main);
    main.append(view.el);
    for (const a of navLinks) {
      if (a.dataset['route'] === route.name) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
    }
    const title = view.heading?.textContent?.trim();
    win.document.title = title && route.name !== 'home' ? `${title} – DriftSense Pilot` : 'DriftSense Pilot';
    // Move focus to the new page heading (not on first load, so the skip link stays first).
    if (!first) (view.heading ?? main).focus();
    first = false;
    return view;
  }

  const onHash = () => render();
  win.addEventListener('hashchange', onHash);
  render();

  return {
    ctx,
    main,
    render,
    dispose() {
      win.removeEventListener('hashchange', onHash);
      current?.dispose?.();
      clearTimeout(announceTimer);
    },
  };
}
