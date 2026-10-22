import { describe, expect, it } from 'vitest';
import { buildApprovalDecision, describeDecision, explainEscalation } from '../src/lib/approval';
import { MAX_UPLOAD_BYTES, contentTypeFor, validateUploadFile } from '../src/lib/upload';
import type { ApprovalRequest } from '../src/types/contracts';
import { loadSchema, validate } from './helpers/schema';

describe('upload validation', () => {
  it('accepts PNG and TIFF within the limit', () => {
    expect(validateUploadFile({ name: 'a.png', type: 'image/png', size: 10 })).toBeNull();
    expect(validateUploadFile({ name: 'a.tiff', type: 'image/tiff', size: MAX_UPLOAD_BYTES })).toBeNull();
  });
  it('infers the type from the extension when the browser gives none', () => {
    expect(contentTypeFor({ name: 'scan.TIF', type: '', size: 1 })).toBe('image/tiff');
    expect(contentTypeFor({ name: 'x.png', type: '', size: 1 })).toBe('image/png');
    expect(validateUploadFile({ name: 'scan.tif', type: '', size: 5 })).toBeNull();
  });
  it('rejects other types, empty and oversize files', () => {
    expect(validateUploadFile({ name: 'a.jpg', type: 'image/jpeg', size: 10 })).toMatch(/PNG or TIFF/);
    expect(validateUploadFile({ name: 'a.png', type: 'image/png', size: 0 })).toMatch(/empty/);
    expect(validateUploadFile({ name: 'a.png', type: 'image/png', size: MAX_UPLOAD_BYTES + 1 })).toMatch(/limit is 20\.0 MB/);
    expect(validateUploadFile(null)).toMatch(/Choose a file/);
  });
});

describe('approval payload', () => {
  it('builds an approve decision with trimmed fields', () => {
    const r = buildApprovalDecision({ choice: 2, approver: '  Ada  ', note: ' looks right ', optionCount: 3 });
    expect(r).toEqual({ ok: true, decision: { chosen_rank: 2, approver: 'Ada', note: 'looks right' } });
    const schema = loadSchema('approval_decision');
    if (schema && r.ok) expect(validate(schema, r.decision)).toEqual([]);
  });
  it('builds a reject-all decision (chosen_rank null) and omits an empty note', () => {
    const r = buildApprovalDecision({ choice: 'reject', approver: 'Ada', note: '   ', optionCount: 3 });
    expect(r).toEqual({ ok: true, decision: { chosen_rank: null, approver: 'Ada' } });
  });
  it('requires an approver', () => {
    const r = buildApprovalDecision({ choice: 1, approver: '  ', optionCount: 3 });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.errors[0]!.field).toBe('approver');
  });
  it('rejects a rank beyond the offered options', () => {
    const r = buildApprovalDecision({ choice: 3, approver: 'Ada', optionCount: 2 });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.errors[0]!.field).toBe('choice');
  });
  it('explains the escalation and the decision in words', () => {
    const req: ApprovalRequest = {
      options: [{ rank: 1, x_px: 1, y_px: 2, score: 0.5 }],
      margin: { margin: 0.15, tau: 0.22, calibrated: true, decision: 'escalate' },
    };
    expect(explainEscalation(req)).toMatch(/0\.150 is below the commit threshold τ = 0\.220/);
    expect(describeDecision(1, req.options)).toMatch(/Approve rank 1 at \(1\.0, 2\.0\) px/);
    expect(describeDecision('reject', req.options)).toMatch(/No stage move/);
  });
});
