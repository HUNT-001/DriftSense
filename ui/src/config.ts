import { HttpApi, type FetchFn } from './api/http';
import { MockApi } from './api/mock';
import type { Api } from './api/types';

/** Runtime configuration: `config.json` next to index.html (infra overwrites it at deploy). */
export interface AppConfig {
  apiBase: string;
  /** True = in-browser mock API (demo mode). */
  mock: boolean;
  /** Why mock mode is on, for the banner. */
  mockReason?: 'no-api-base' | 'query' | 'config-error';
}

/** `?mock=1` in the query string or inside the hash (e.g. `#/?mock=1`). */
export function wantsMock(search: string, hash: string): boolean {
  const q = new URLSearchParams(search);
  if (q.get('mock') === '1') return true;
  const i = hash.indexOf('?');
  return i >= 0 && new URLSearchParams(hash.slice(i + 1)).get('mock') === '1';
}

export async function loadConfig(
  fetchFn: FetchFn = (...a) => globalThis.fetch(...a),
  loc: Pick<Location, 'search' | 'hash'> = window.location,
): Promise<AppConfig> {
  let apiBase = '';
  let failed = false;
  try {
    const res = await fetchFn('config.json', { cache: 'no-store' });
    if (res.ok) {
      const body = (await res.json()) as { apiBase?: unknown };
      if (typeof body.apiBase === 'string') apiBase = body.apiBase.trim().replace(/\/+$/, '');
    } else failed = true;
  } catch {
    failed = true;
  }
  if (wantsMock(loc.search, loc.hash)) return { apiBase, mock: true, mockReason: 'query' };
  if (!apiBase) return { apiBase, mock: true, mockReason: failed ? 'config-error' : 'no-api-base' };
  return { apiBase, mock: false };
}

export function createApi(cfg: AppConfig, fetchFn?: FetchFn): Api {
  return cfg.mock ? new MockApi() : new HttpApi(cfg.apiBase, fetchFn);
}
