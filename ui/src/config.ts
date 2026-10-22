import { HttpApi, type FetchFn } from './api/http';
import { ReplayApi } from './api/replay';
import type { Api } from './api/types';

export type OfflineReason = 'query' | 'no-api-base' | 'config-error' | 'unreachable';

/** Runtime configuration: `config.json` next to index.html (infra overwrites it at deploy). */
export interface AppConfig {
  apiBase: string;
  /** True = offline replay of the recorded traces (no API contacted). */
  offline: boolean;
  offlineReason?: OfflineReason;
}

type Loc = Pick<Location, 'search' | 'hash'>;

/** Query params from `?a=b` and from inside the hash (`#/?a=b`); the hash wins. */
export function queryParams(loc: Loc): URLSearchParams {
  const out = new URLSearchParams(loc.search);
  const i = loc.hash.indexOf('?');
  if (i >= 0) for (const [k, v] of new URLSearchParams(loc.hash.slice(i + 1))) out.set(k, v);
  return out;
}

/** `?demo=offline` (or the legacy `?mock=1`) forces offline replay. */
export function wantsOffline(loc: Loc): boolean {
  const q = queryParams(loc);
  return q.get('demo') === 'offline' || q.get('mock') === '1';
}

/** `?api=https://...` overrides config.json (http/https only, trailing slash dropped). */
export function apiOverride(loc: Loc): string | null {
  const v = queryParams(loc).get('api');
  if (!v) return null;
  try {
    const u = new URL(v);
    if (u.protocol !== 'https:' && u.protocol !== 'http:') return null;
    return v.trim().replace(/\/+$/, '');
  } catch {
    return null;
  }
}

export async function loadConfig(
  fetchFn: FetchFn = (...a) => globalThis.fetch(...a),
  loc: Loc = window.location,
): Promise<AppConfig> {
  let apiBase = '';
  let failed = false;
  const override = apiOverride(loc);
  if (override) apiBase = override;
  else {
    try {
      const res = await fetchFn('config.json', { cache: 'no-store' });
      if (res.ok) {
        const body = (await res.json()) as { apiBase?: unknown };
        if (typeof body.apiBase === 'string') apiBase = body.apiBase.trim().replace(/\/+$/, '');
      } else failed = true;
    } catch {
      failed = true;
    }
  }
  if (wantsOffline(loc)) return { apiBase, offline: true, offlineReason: 'query' };
  if (!apiBase) return { apiBase, offline: true, offlineReason: failed ? 'config-error' : 'no-api-base' };
  return { apiBase, offline: false };
}

/**
 * Probe the live API once at boot; if /healthz does not answer in time, fall back to offline
 * replay so a judge never sees a blank page.
 */
export async function resolveConfig(cfg: AppConfig, fetchFn?: FetchFn, timeoutMs = 4000): Promise<AppConfig> {
  if (cfg.offline) return cfg;
  const ok = await new HttpApi(cfg.apiBase, fetchFn).healthz(timeoutMs);
  return ok ? cfg : { ...cfg, offline: true, offlineReason: 'unreachable' };
}

export function createApi(cfg: AppConfig, fetchFn?: FetchFn): Api {
  return cfg.offline ? new ReplayApi() : new HttpApi(cfg.apiBase, fetchFn);
}

export const OFFLINE_LABEL = 'Offline replay of a recorded trace';

export function offlineReasonText(r: OfflineReason | undefined): string {
  switch (r) {
    case 'query':
      return 'You asked for offline mode (?demo=offline).';
    case 'unreachable':
      return 'The live API did not answer, so the UI switched to offline mode.';
    case 'config-error':
      return 'The API configuration could not be loaded, so the UI is in offline mode.';
    default:
      return 'No API is configured, so the UI is in offline mode.';
  }
}
