import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { SampleLoader } from '../../src/api/replay';

export const uiDir = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
export const samplesDir = resolve(uiDir, 'public', 'samples');
export const repoTracesDir = resolve(uiDir, '..', 'traces');

export function readSample(name: string): string {
  return readFileSync(resolve(samplesDir, name), 'utf8');
}

/** Filesystem loader for ReplayApi (tests have no static server). */
export const fsLoader: SampleLoader = async (p) => readSample(p);

export function repoTrace(name: string): string | null {
  const p = resolve(repoTracesDir, name);
  return existsSync(p) ? readFileSync(p, 'utf8') : null;
}
