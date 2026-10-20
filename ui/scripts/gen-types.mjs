#!/usr/bin/env node
// Generate ui/src/types/contracts.ts from the contract JSON Schemas.
//
// Usage (from ui/):
//   npm run gen:types                       # reads ../contracts/*.schema.json
//   npm run gen:types -- --schemas <dir>    # reads <dir>/*.schema.json
//   npm run gen:types -- --check            # exit 1 if the committed file is stale
//
// The schemas are exported by driftsense/contracts.py (export_schemas). All files share the
// same $defs names (Candidate, Margin, ...), so they are merged into one root schema and compiled
// once; that keeps a single TypeScript declaration per contract model.
import { readFileSync, readdirSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, resolve, basename, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import { compile } from 'json-schema-to-typescript';

const uiDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const outFile = resolve(uiDir, 'src', 'types', 'contracts.ts');

function parseArgs(argv) {
  const args = { schemas: resolve(uiDir, '..', 'contracts'), check: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--schemas') args.schemas = resolve(process.cwd(), argv[++i] ?? '');
    else if (a.startsWith('--schemas=')) args.schemas = resolve(process.cwd(), a.slice(10));
    else if (a === '--check') args.check = true;
    else if (a === '-h' || a === '--help') {
      console.log('usage: gen-types.mjs [--schemas <dir>] [--check]');
      process.exit(0);
    } else {
      console.error(`unknown argument: ${a}`);
      process.exit(2);
    }
  }
  return args;
}

/** Pydantic puts a `title` on every property; json-schema-to-typescript would turn each into a
 * named alias (and collide, e.g. Margin.margin -> "Margin"). Strip them, convert 2020-12
 * `prefixItems` tuples to the draft-07 `items: [...]` form the compiler understands. */
function normalize(node) {
  if (Array.isArray(node)) return node.map(normalize);
  if (!node || typeof node !== 'object') return node;
  const out = {};
  for (const [k, v] of Object.entries(node)) {
    if (k === 'title') continue;
    if (k === 'prefixItems') {
      out.items = normalize(v);
      continue;
    }
    if (k === 'properties' || k === '$defs') {
      out[k] = Object.fromEntries(Object.entries(v).map(([pk, pv]) => [pk, normalize(pv)]));
      continue;
    }
    out[k] = normalize(v);
  }
  return out;
}

function pascal(s) {
  return s.replace(/(^|_)([a-z])/g, (_, __, c) => c.toUpperCase());
}

const args = parseArgs(process.argv.slice(2));
if (!existsSync(args.schemas)) {
  console.error(`schema dir not found: ${args.schemas}\n(pass --schemas <dir>)`);
  process.exit(1);
}
const files = readdirSync(args.schemas).filter((f) => f.endsWith('.schema.json')).sort();
if (files.length === 0) {
  console.error(`no *.schema.json in ${args.schemas}`);
  process.exit(1);
}

const defs = {};
const rootProps = {};
for (const f of files) {
  const raw = JSON.parse(readFileSync(resolve(args.schemas, f), 'utf8'));
  const name = raw.title || pascal(basename(f, '.schema.json'));
  const { $defs = {}, ...top } = raw;
  for (const [k, v] of Object.entries($defs)) {
    if (defs[k] && !isDeepStrictEqual(defs[k], v)) {
      console.error(`conflicting $defs.${k} between schema files (last: ${f})`);
      process.exit(1);
    }
    defs[k] = v;
  }
  if (defs[name] && !isDeepStrictEqual(defs[name], top)) {
    console.error(`conflicting definition for ${name} (${f})`);
    process.exit(1);
  }
  defs[name] = top;
  rootProps[basename(f, '.schema.json')] = { $ref: `#/$defs/${name}` };
}

const normDefs = {};
for (const [k, v] of Object.entries(defs)) normDefs[k] = { ...normalize(v), title: k };

const root = {
  title: 'DriftSenseContracts',
  description: 'Index of all contract schemas (generated root; not a wire type).',
  type: 'object',
  properties: rootProps,
  required: Object.keys(rootProps),
  additionalProperties: false,
  $defs: normDefs,
};

const banner = `/* eslint-disable */
/**
 * GENERATED FILE - DO NOT EDIT BY HAND.
 * Source: contracts/{${files.join(', ')}}
 * (JSON Schemas exported by driftsense/contracts.py export_schemas.)
 * Regenerate (from ui/):  npm run gen:types            # uses ../contracts
 *                         npm run gen:types -- --schemas <dir>
 */`;

/** A `$ref` with sibling keywords (e.g. `{"$ref": "#/$defs/Mode", "default": "auto"}`) is emitted
 * as a numbered duplicate (`Mode1`). Fold duplicates whose body equals the base declaration. */
function dedupe(src) {
  const decl = /(?:\/\*\*(?:(?!\*\/)[\s\S])*\*\/\n)?export (type|interface) (\w+?)(\d+)\b( =[^\n]*;| \{\n[\s\S]*?\n\})\n/g;
  let out = src;
  for (const m of src.matchAll(decl)) {
    const [whole, kind, base, num, body] = m;
    const baseDecl = new RegExp(`export ${kind} ${base}${body.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\n`);
    if (!baseDecl.test(out)) continue;
    out = out.replace(whole, '').replace(new RegExp(`\\b${base}${num}\\b`, 'g'), base);
  }
  return out;
}

const compiled = await compile(root, 'DriftSenseContracts', {
  bannerComment: banner,
  additionalProperties: false,
  unreachableDefinitions: false,
  declareExternallyReferenced: true,
  format: true,
  style: { singleQuote: true, printWidth: 100 },
});
const ts = dedupe(compiled);

if (args.check) {
  const current = existsSync(outFile) ? readFileSync(outFile, 'utf8') : '';
  if (current !== ts) {
    console.error(`${relative(uiDir, outFile)} is stale; run npm run gen:types`);
    process.exit(1);
  }
  console.log('contracts.ts is up to date');
} else {
  mkdirSync(dirname(outFile), { recursive: true });
  writeFileSync(outFile, ts);
  console.log(`wrote ${relative(uiDir, outFile)} from ${files.length} schema files`);
}
