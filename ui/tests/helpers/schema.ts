/**
 * Test helpers: locate the contract JSON Schemas and validate data against them with a small
 * JSON-Schema subset validator (enough for pydantic's output: type, enum, const, required,
 * properties, additionalProperties, anyOf, $ref to #/$defs, items/prefixItems, min/max,
 * minItems/maxItems, format date-time). Avoids adding ajv as a dependency.
 */
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const uiDir = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');

/** Same convention as scripts/gen-types.mjs: DRIFTSENSE_SCHEMAS env var, else ../contracts. */
export function schemaDir(): string {
  const env = process.env['DRIFTSENSE_SCHEMAS'];
  return env ? resolve(env) : resolve(uiDir, '..', 'contracts');
}

export function loadSchema(name: string): Record<string, unknown> | null {
  const p = resolve(schemaDir(), `${name}.schema.json`);
  if (!existsSync(p)) return null;
  return JSON.parse(readFileSync(p, 'utf8')) as Record<string, unknown>;
}

type S = Record<string, unknown>;

function typeOk(t: string, v: unknown): boolean {
  switch (t) {
    case 'object':
      return typeof v === 'object' && v !== null && !Array.isArray(v);
    case 'array':
      return Array.isArray(v);
    case 'string':
      return typeof v === 'string';
    case 'integer':
      return Number.isInteger(v);
    case 'number':
      return typeof v === 'number' && Number.isFinite(v);
    case 'boolean':
      return typeof v === 'boolean';
    case 'null':
      return v === null;
    default:
      return true;
  }
}

export function validate(schema: S, value: unknown, root: S = schema, path = '$'): string[] {
  const errs: string[] = [];
  if (typeof schema['$ref'] === 'string') {
    const ref = schema['$ref'] as string;
    const m = /^#\/\$defs\/(.+)$/.exec(ref);
    const target = m ? ((root['$defs'] as S | undefined)?.[m[1]!] as S | undefined) : undefined;
    if (!target) return [`${path}: unresolved $ref ${ref}`];
    errs.push(...validate(target, value, root, path));
  }
  if (Array.isArray(schema['anyOf'])) {
    const ok = (schema['anyOf'] as S[]).some((s) => validate(s, value, root, path).length === 0);
    if (!ok) errs.push(`${path}: matches none of anyOf`);
  }
  if (typeof schema['type'] === 'string' && !typeOk(schema['type'] as string, value)) {
    return [...errs, `${path}: expected ${schema['type'] as string}`];
  }
  if ('const' in schema && value !== schema['const']) errs.push(`${path}: expected const ${String(schema['const'])}`);
  if (Array.isArray(schema['enum']) && !(schema['enum'] as unknown[]).includes(value)) {
    errs.push(`${path}: not in enum`);
  }
  if (typeof value === 'number') {
    if (typeof schema['minimum'] === 'number' && value < schema['minimum']) errs.push(`${path}: below minimum`);
    if (typeof schema['maximum'] === 'number' && value > schema['maximum']) errs.push(`${path}: above maximum`);
  }
  if (typeof value === 'string' && schema['format'] === 'date-time' && Number.isNaN(Date.parse(value))) {
    errs.push(`${path}: invalid date-time`);
  }
  if (Array.isArray(value)) {
    if (typeof schema['minItems'] === 'number' && value.length < schema['minItems']) errs.push(`${path}: too few items`);
    if (typeof schema['maxItems'] === 'number' && value.length > schema['maxItems']) errs.push(`${path}: too many items`);
    const prefix = schema['prefixItems'] as S[] | undefined;
    value.forEach((item, i) => {
      const s = prefix?.[i] ?? (schema['items'] as S | undefined);
      if (s && typeof s === 'object' && !Array.isArray(s)) errs.push(...validate(s, item, root, `${path}[${i}]`));
    });
  }
  if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>;
    const props = (schema['properties'] as Record<string, S> | undefined) ?? {};
    for (const r of (schema['required'] as string[] | undefined) ?? []) {
      if (!(r in obj)) errs.push(`${path}: missing ${r}`);
    }
    for (const [k, v] of Object.entries(obj)) {
      if (props[k]) errs.push(...validate(props[k]!, v, root, `${path}.${k}`));
      else if (schema['additionalProperties'] === false) errs.push(`${path}: unexpected property ${k}`);
    }
  }
  return errs;
}
