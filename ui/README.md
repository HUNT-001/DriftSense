# DriftSense Pilot UI

A static single-page app (Vite + vanilla TypeScript, no framework, no runtime CDN, no web fonts).
S3 + CloudFront host it; it talks only to the DriftSense job API (and, in offline mode, to its own
`samples/` files).

| Route | Screen |
| --- | --- |
| `#/` | Home: problem + tagline, one-click **Run agent episode** (family + seed), demo-site picker (agent-episode sites are tagged), upload form (template + search, PNG/TIFF ≤ 20 MB) |
| `#/jobs/<id>` | Live run: status, outcome summary, **agent timeline**, approval card, episode summary (`episode.json`), localization result (offset px/nm, margin vs τ, candidates, NCC vs roughness heatmaps, overlay, timings, worker), responsible-use panel |
| `#/about` | How it decides (commit only if margin ≥ τ, otherwise ask a human). States no numbers |
| `#/responsible` | Responsible-use commitments |

## The agent timeline (Agentic Vision evidence)

Every `TraceEvent` is a row: kind, tool, latency (`latency_ms`, else the Δt between timestamps),
args/results, and for every event with `margin` + `tau` a **margin-vs-τ bar** that states the
consequence in words: `margin 0.469 ≥ τ 0.100 → commit allowed` or `margin -0.026 < τ 0.100 →
re-acquire or escalate` (`escalate to a human` on the escalate event). Each localize result also shows
**"Agent's next action: …"**, so the trace visibly shows the OpenCV margin choosing the next tool.
Acquire results carry a cumulative-dose bar (`cumulative_dose`). The `commit_move` and `escalate`
events are highlighted and their `because` string is quoted, and a headline card summarises the decisive
step. New rows are appended in place (no re-render) and announced through a polite ARIA live region.

## Develop

```bash
cd ui
npm ci
npm run dev        # http://localhost:5173 (public/config.json has apiBase "" -> offline replay)
npm test           # vitest (jsdom)
npm run build      # tsc --noEmit && vite build -> ui/dist/
```

Point the UI at an API without editing files: `http://localhost:5173/?api=http://localhost:8000`.
`?api=` (http/https only) overrides `config.json` and also works inside the hash (`#/?api=...`).

### Offline mode (judge mode)

The UI replays the **real committed traces** instead of calling an API when:

- the URL has `?demo=offline` (legacy alias `?mock=1`), or
- `config.json` has an empty `apiBase` or cannot be loaded, or
- `GET {apiBase}/healthz` does not answer within 4 s at boot.

A banner and badges say **"Offline replay of a recorded trace"**. The replay plays
`traces/demo_commit.jsonl` (margin ≥ τ, so the agent commits) and `traces/demo_escalate.jsonl`
(the margin stays below τ when the budget runs out, so the agent escalates), paced at about 0.5 s per step
plus the recorded tool latency. Each replay takes under 20 s. The escalation pauses for a **simulated
approval**: the person at the keyboard picks rank 1–3 or rejects all. Only the operator events after that
are synthesised; they carry `result_summary.simulated = true` and are badged "simulated operator step".
Uploads are disabled offline.

### Sample data (`public/samples/`, committed so the build is hermetic)

| File | From |
| --- | --- |
| `demo_commit.jsonl`, `demo_escalate.jsonl` | Byte copies of `../traces/*.jsonl`. `npm run samples` copies them; `npm run samples:check` fails if they drift (the trace test checks this too) |
| `*.capture.json`, `demo_*/step7_*.webp` | `npm run samples:capture` (`python ui/scripts/capture_samples.py`, needs the engine venv). It re-runs the same seeds and knobs as `eval/agent_eval.py` and keeps only the `localize` results whose margin reproduces the committed trace line. It records the full `LocalizeResult` (top-5 candidates, timings), the `ApprovalRequest`, and the S9 explain images with the engine's alt text. The images are re-encoded PNG → WebP to keep about 0.6 MB total; the original PNG sha256 stays in `result.artifacts` |

### Contract types

`src/types/contracts.ts` is **generated** from `../contracts/*.schema.json` (CONTRACT_VERSION 1.0.0):

```bash
npm run gen:types            # regenerate
npm run gen:types:check      # fail if the committed file is stale
```

The tests validate every line of both sample traces, every replayed `JobStatus`, the captured
`LocalizeResult`s and the approval payload against those schemas (subset validator in
`tests/helpers/schema.ts`; `$DRIFTSENSE_SCHEMAS` overrides the directory).

## Deploy

1. `cd ui && npm ci && npm run build`.
2. Sync `ui/dist/` to the UI bucket root: `aws s3 sync dist/ s3://<ui-bucket>/ --delete`.
   - `assets/*` are content-hashed. Serve them with `Cache-Control: public, max-age=31536000, immutable`.
   - Serve `index.html`, `config.json` and `samples/*` with `Cache-Control: no-cache`.
3. **Write `config.json`** with the API URL (no trailing slash), then upload it last:

   ```json
   { "apiBase": "https://<api-id>.execute-api.<region>.amazonaws.com" }
   ```

4. Asset URLs are relative (`base: './'`) and routing is hash-based, so the app needs no SPA rewrite rules.
5. The API's `UI_ORIGIN` must be the CloudFront origin (CORS). The data bucket's CORS must allow `PUT` from it
   for presigned uploads.
6. Suggested CSP: `default-src 'self'; img-src 'self' <apiBase> https://*.amazonaws.com; connect-src 'self' <apiBase> https://*.amazonaws.com; style-src 'self'; script-src 'self'; frame-ancestors 'none'`.

## API used (service/README.md)

| Call | Use |
| --- | --- |
| `GET /healthz` | Boot probe (4 s timeout) → offline fallback |
| `GET /v1/version` | Footer: service, contract, git sha, engine, OpenCV (+ `opencv_file`), NumPy, Python, arch |
| `GET /v1/demos` | Picker. Agent-episode demos are recognised by `agent`/`episode: true`, `kind` matching `episode`/`agent`, or the id prefix `vsem-`. Family and seed come from `family`/`seed`, else from `vsem-<family>-<seed>` |
| `POST /v1/uploads` | `{filename, content_type, size_bytes, role, upload_id?}`; the second file reuses the first `upload_id`. The PUT sends exactly the returned `headers` (browsers set `Content-Length` themselves) |
| `POST /v1/jobs` | `JobCreate` with a fresh `client_token` → 202 `JobStatus` |
| `POST /v1/agent/episodes` | `{family, seed, client_token}` → 202 `JobStatus`. The id is remembered in `sessionStorage` because `JobStatus` does not echo `agent` |
| `GET /v1/jobs/{id}` | Polled with backoff 0.5 → 2 s (reset on each status change). Polling stops on `succeeded`/`failed` and pauses on `awaiting_approval`. After 5 consecutive errors the UI shows Retry; a 404 stops it at once |
| `GET /v1/jobs/{id}/trace` | NDJSON `TraceEvent`s. Re-fetched every 1 s while the job is queued/running, once more after it ends. `404 not_ready` = no events yet. Bad lines are skipped and reported by line number |
| `GET /v1/jobs/{id}/artifacts/{name}` | Images, `episode.json` (shown generically as key/value), `result.json`. `Artifact.uri` (`jobs/<id>/<name>`) is resolved **by basename**; `http(s)` URIs pass through |
| `POST /v1/jobs/{id}/approval` | `ApprovalDecision {chosen_rank: 1..3 \| null, approver, note?}`. Polling then resumes, so an episode can go back to `queued`/`running` before it finishes |

Final states shown: `succeeded` (an approved choice is detected from the `{"operator_choice": …}` warning),
or `failed` with `rejected_by_operator`, `approval_timeout`, `agent_abort`, `bad_input` or `engine_error`.

## Accessibility

- A skip link comes first in the tab order. The page has landmarks and one `h1` per view, and focus moves to the heading on navigation.
- Everything works by keyboard and every control is at least 44 px tall. When the approval card appears, focus moves to it. The confirm step takes focus, Escape cancels and returns focus to the button that opened it, and the outcome notice takes focus after a decision.
- Two polite live regions: one for status and approval results, one for new trace steps (the decisive commit/escalate step is announced first).
- Every input has a `<label>`, and errors use `aria-describedby` and `aria-invalid`. Images get the engine's alt text, or a generated description when there is none.
- Meaning never depends on colour alone: icons, text verdicts and a hatched below-τ fill. The palette is Okabe-Ito derived (bluish-green = commit, orange = escalate, purple = replay), scores use viridis swatches, and the engine heatmaps use inferno.
- Light and dark follow `prefers-color-scheme`. `tests/contrast.test.ts` checks every text token pair for at least 4.5:1, and borders for at least 3:1.
- The layout works at 375 px with no horizontal page scroll (tables scroll inside their own region). `prefers-reduced-motion` and `forced-colors` are respected.
