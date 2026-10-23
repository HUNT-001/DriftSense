/* eslint-disable */
/**
 * GENERATED FILE - DO NOT EDIT BY HAND.
 * Source: contracts/{approval_decision.schema.json, job_create.schema.json, job_status.schema.json, localize_result.schema.json, results_file.schema.json, trace_event.schema.json}
 * (JSON Schemas exported by driftsense/contracts.py export_schemas.)
 * Regenerate (from ui/):  npm run gen:types            # uses ../contracts
 *                         npm run gen:types -- --schemas <dir>
 */

export type Mode = 'fast' | 'accurate' | 'dense' | 'auto';
export type LatticeType = 'finfet' | 'dram' | 'grid' | 'staggered' | 'unknown';

/**
 * Index of all contract schemas (generated root; not a wire type).
 */
export interface DriftSenseContracts {
  approval_decision: ApprovalDecision;
  job_create: JobCreate;
  job_status: JobStatus;
  localize_result: LocalizeResult;
  results_file: ResultsFile;
  trace_event: TraceEvent;
}
export interface ApprovalDecision {
  approver: string;
  /**
   * None = reject all
   */
  chosen_rank?: number | null;
  note?: string | null;
}
export interface JobCreate {
  agent?: boolean;
  budget?: Budget;
  /**
   * Idempotency key
   */
  client_token?: string | null;
  demo_id?: string | null;
  mode?: Mode;
  search_key?: string | null;
  template_key?: string | null;
}
export interface Budget {
  max_candidates?: number;
  /**
   * Soft wall-clock budget for one localize call
   */
  max_ms?: number;
  /**
   * Override calibrated uniqueness-margin threshold; None = calibrated
   */
  tau?: number | null;
}
export interface JobStatus {
  approval?: ApprovalRequest | null;
  created_at: string;
  engine?: ('fake' | 'v1' | 'v2') | null;
  error?: ErrorInfo | null;
  job_id: string;
  result?: LocalizeResult | null;
  status: 'queued' | 'running' | 'awaiting_approval' | 'succeeded' | 'failed';
  trace_key?: string | null;
  updated_at: string;
  worker?: WorkerInfo | null;
}
export interface ApprovalRequest {
  margin: Margin;
  /**
   * @minItems 1
   * @maxItems 3
   */
  options: [Candidate] | [Candidate, Candidate] | [Candidate, Candidate, Candidate];
}
export interface Margin {
  calibrated: boolean;
  decision: 'commit' | 'escalate';
  /**
   * Calibrated uniqueness margin between rank-1 and rank-2
   */
  margin: number;
  p_correct?: number | null;
  tau: number;
}
export interface Candidate {
  lattice_score?: number | null;
  rank: number;
  roughness_score?: number | null;
  /**
   * Final fused score, higher is better
   */
  score: number;
  x_px: number;
  y_px: number;
}
export interface ErrorInfo {
  code: string;
  message: string;
}
export interface LocalizeResult {
  artifacts?: Artifact[];
  /**
   * @minItems 1
   */
  candidates: [Candidate, ...Candidate[]];
  contract_version?: string;
  engine: 'fake' | 'v1' | 'v2';
  engine_version: string;
  lattice_type?: LatticeType;
  margin: Margin;
  mode: Mode;
  offset: Offset;
  timings?: StageTiming[];
  total_ms: number;
  warnings?: string[];
}
export interface Artifact {
  kind: 'heatmap' | 'overlay' | 'contour' | 'candidates' | 'json';
  sha256?: string | null;
  /**
   * Relative path locally, s3:// key in service
   */
  uri: string;
}
/**
 * Template top-left location in search-image coordinates (x right, y down), subpixel.
 */
export interface Offset {
  pixel_size_nm?: number | null;
  x_nm?: number | null;
  x_px: number;
  y_nm?: number | null;
  y_px: number;
}
export interface StageTiming {
  ms: number;
  /**
   * e.g. ingest, lattice, rectify, roughness, score, cascade, contour, confidence, subpixel, explain
   */
  stage: string;
}
export interface WorkerInfo {
  arch?: 'x86_64' | 'aarch64' | 'arm64' | 'unknown';
  cool?: boolean;
  instance_type?: string | null;
  numpy_version: string;
  opencv_version: string;
}
export interface ResultsFile {
  created_at: string;
  dataset: 'dev' | 'sealed' | 'synthetic' | 'bench';
  env: EnvInfo;
  git_sha: string;
  metrics: {
    [k: string]: Metric;
  };
  name: string;
  /**
   * Script path that generated this file
   */
  produced_by: string;
  protocol_sha256?: string | null;
  schema_id?: 'driftsense.results/1';
}
export interface EnvInfo {
  arch: string;
  cool?: boolean;
  instance_type?: string | null;
  numpy: string;
  opencv: string;
  platform: string;
  python: string;
}
export interface Metric {
  ci_high?: number | null;
  ci_low?: number | null;
  n?: number | null;
  unit: string;
  value: number;
}
export interface TraceEvent {
  args?: {
    [k: string]: unknown;
  };
  /**
   * Human-readable reason, e.g. 'margin 0.31 >= tau 0.22'
   */
  because?: string | null;
  kind:
    | 'observe'
    | 'tool_call'
    | 'tool_result'
    | 'decision'
    | 'commit_move'
    | 'escalate'
    | 'human_approval'
    | 'error'
    | 'episode_end';
  latency_ms?: number | null;
  margin?: number | null;
  policy?: ('rules' | 'llm') | null;
  result_summary?: {
    [k: string]: unknown;
  };
  run_id: string;
  stage_move_nm?: [number, number] | null;
  step: number;
  tau?: number | null;
  tool?: string | null;
  ts: string;
  v?: '1';
}
