import type { ApprovalDecision, ApprovalRequest, Candidate } from '../types/contracts';
import { fmt } from './format';

export type ApprovalChoice = 1 | 2 | 3 | 'reject';

export interface ApprovalInput {
  choice: ApprovalChoice;
  approver: string;
  note?: string | null;
  /** Number of options offered (1..3); a chosen rank must be within it. */
  optionCount: number;
}

export interface FieldError {
  field: 'approver' | 'note' | 'choice';
  message: string;
}

export type ApprovalBuild = { ok: true; decision: ApprovalDecision } | { ok: false; errors: FieldError[] };

export const APPROVER_MAX = 120;
export const NOTE_MAX = 1000;

/** Validate the approval form and build the ApprovalDecision payload (contract shape). */
export function buildApprovalDecision(input: ApprovalInput): ApprovalBuild {
  const errors: FieldError[] = [];
  const approver = (input.approver ?? '').trim();
  const note = (input.note ?? '').trim();
  if (!approver) errors.push({ field: 'approver', message: 'Enter your name before approving or rejecting.' });
  else if (approver.length > APPROVER_MAX) {
    errors.push({ field: 'approver', message: `Name must be at most ${APPROVER_MAX} characters.` });
  }
  if (note.length > NOTE_MAX) errors.push({ field: 'note', message: `Note must be at most ${NOTE_MAX} characters.` });
  let chosen: number | null = null;
  if (input.choice !== 'reject') {
    const max = Math.min(3, Math.max(0, input.optionCount));
    if (!(Number.isInteger(input.choice) && input.choice >= 1 && input.choice <= max)) {
      errors.push({ field: 'choice', message: `Choose a rank between 1 and ${max}.` });
    } else chosen = input.choice;
  }
  if (errors.length) return { ok: false, errors };
  const decision: ApprovalDecision = { chosen_rank: chosen, approver };
  if (note) decision.note = note;
  return { ok: true, decision };
}

/** Plain-language sentence explaining why the agent is asking. */
export function explainEscalation(req: ApprovalRequest): string {
  const { margin, tau } = req.margin;
  return (
    `The agent could not prove which copy is the real one: its uniqueness margin ${fmt(margin, 3)} ` +
    `is below the commit threshold τ = ${fmt(tau, 3)}. It will not move the stage on its own. ` +
    `Approve the candidate you believe is correct, or reject all of them.`
  );
}

/** Confirmation sentence shown before the decision is posted. */
export function describeDecision(choice: ApprovalChoice, options: Candidate[]): string {
  if (choice === 'reject') return 'Reject all candidates. No stage move will be made for this site.';
  const c = options.find((o) => o.rank === choice);
  const where = c ? ` at (${fmt(c.x_px, 1)}, ${fmt(c.y_px, 1)}) px, score ${fmt(c.score, 3)}` : '';
  return `Approve rank ${choice}${where}. The stage correction for this candidate will be committed on your authority.`;
}
