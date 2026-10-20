#!/usr/bin/env node
// Copy the REAL committed agent traces into ui/public/samples/ so the build is hermetic.
//
//   npm run samples            # copy ../traces/demo_{commit,escalate}.jsonl -> public/samples/
//   npm run samples -- --check # exit 1 if a committed copy differs from ../traces (CI guard)
//
// The copies are committed; `npm run build` never reaches outside ui/. When ../traces is absent
// (e.g. ui/ checked out alone) --check is a no-op and the committed copies are used as-is.
// The matching *.capture.json files (full LocalizeResult, approval request, explain images) come
// from scripts/capture_samples.py, which needs the Python engine.
import { copyFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const SAMPLES = ['demo_commit.jsonl', 'demo_escalate.jsonl'];

const uiDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const srcDir = resolve(uiDir, '..', 'traces');
const outDir = resolve(uiDir, 'public', 'samples');
const check = process.argv.includes('--check');

if (!existsSync(srcDir)) {
  console.log(`no ${srcDir}; keeping the committed copies in public/samples/`);
  process.exit(0);
}
mkdirSync(outDir, { recursive: true });
let stale = 0;
for (const name of SAMPLES) {
  const src = resolve(srcDir, name);
  const dst = resolve(outDir, name);
  if (!existsSync(src)) {
    console.error(`missing ${src}`);
    process.exit(1);
  }
  // Compare LF-normalised text so a CRLF checkout (core.autocrlf) is not reported as stale.
  const norm = (p) => readFileSync(p, 'utf8').replace(/\r\n/g, '\n');
  const same = existsSync(dst) && norm(src) === norm(dst);
  if (check) {
    if (!same) {
      console.error(`public/samples/${name} differs from traces/${name}; run npm run samples`);
      stale++;
    }
  } else if (!same) {
    copyFileSync(src, dst);
    console.log(`copied traces/${name} -> public/samples/${name}`);
  } else console.log(`public/samples/${name} is up to date`);
}
process.exit(stale ? 1 : 0);
