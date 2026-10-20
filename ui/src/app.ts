import type { Api, VersionInfo } from './api/types';
import { OFFLINE_LABEL, offlineReasonText, type AppConfig } from './config';
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
    v.service_version ? `service ${v.service_version}` : null,
    v.contract_version ? `contract ${v.contract_version}` : null,
    v.git_sha ? `git ${shortSha(v.git_sha)}` : null,
    v.engine ? `engine ${v.engine}` : null,
    v.opencv_version ? `OpenCV ${v.opencv_version}` : null,
    v.numpy_version ? `NumPy ${v.numpy_version}` : null,
    v.python_version ? `Python ${v.python_version}` : null,
    v.arch ?? null,
  ];
  return parts.filter(Boolean).join(' · ');
}

export function workerText(w: WorkerInfo | null | undefined): string {
  if (!w) return 'Worker: not known yet (run a job)';
  const where = [w.arch, w.instance_type].filter((x) => x && x !== 'unknown').join(', ');
  return `Worker: OpenCV ${w.opencv_version} · COOL ${w.cool ? 'yes' : 'no'}${where ? ` (${where})` : ''}`;
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

  const here = `${win.location.pathname}`;
  const modeLink = config.offline
    ? h(
        'p',
        null,
        `${OFFLINE_LABEL}: no API is contacted. `,
        config.apiBase ? h('a', { href: here }, 'Retry the live API') : null,
      )
    : h('p', null, h('a', { href: `${here}?demo=offline#/` }, 'Open the offline replay (no backend needed)'));

  const banner = config.offline
    ? h(
        'div',
        { class: 'demo-banner', role: 'note' },
        h('span', { class: 'badge badge-replay' }, h('span', { 'aria-hidden': 'true' }, '⏵ '), 'OFFLINE REPLAY'),
        ' ',
        h('strong', null, `${OFFLINE_LABEL}. `),
        offlineReasonText(config.offlineReason),
        ' The agent steps, margins and images are the real recorded output of the engine; only the operator decision is simulated by you.',
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
      if (v.opencv_file) versionLine.append(h('span', { class: 'muted' }, ` · cv2 from ${v.opencv_file}`));
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
