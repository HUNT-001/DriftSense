export type Route =
  | { name: 'home' }
  | { name: 'job'; id: string }
  | { name: 'about' }
  | { name: 'responsible' }
  | { name: 'notfound'; path: string };

/** Parse a location hash such as `#/jobs/abc?mock=1`. */
export function parseRoute(hash: string): Route {
  let path = hash.replace(/^#/, '');
  const q = path.indexOf('?');
  if (q >= 0) path = path.slice(0, q);
  path = path.replace(/\/+$/, '') || '/';
  if (!path.startsWith('/')) path = `/${path}`;
  if (path === '/') return { name: 'home' };
  if (path === '/about') return { name: 'about' };
  if (path === '/responsible') return { name: 'responsible' };
  const m = /^\/jobs\/([^/]+)$/.exec(path);
  if (m) {
    try {
      return { name: 'job', id: decodeURIComponent(m[1]!) };
    } catch {
      return { name: 'notfound', path };
    }
  }
  return { name: 'notfound', path };
}

export function jobHref(id: string): string {
  return `#/jobs/${encodeURIComponent(id)}`;
}
