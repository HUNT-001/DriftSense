# DriftSense Pilot UI

This is a static single-page app (Vite + vanilla TypeScript, no framework, no runtime CDN). S3 + CloudFront host it, and it talks to the job API.

Screens (hash router):

| Route | Screen |
| --- | --- |
| `#/` | Home: problem statement and tagline, demo-site picker (radio cards), **Run localization** and **Run agent episode** buttons, upload panel (template + search, PNG/TIFF ≤ 20 MB) |
| `#/jobs/<id>` | Live run: status pill, agent decision timeline, approval card, result panel (offset, candidates, margin-vs-τ gauge, timings, worker, heatmaps) |
| `#/about` | What the system does and how it decides (commit only if margin ≥ τ, otherwise ask a human). It states no numbers. |
| `#/responsible` | Responsible-use commitments |

## Develop

```bash
cd ui
npm ci
npm run dev        # http://localhost:5173 ; public/config.json has apiBase "" -> demo mode
npm test           # vitest (jsdom)
npm run build      # tsc --noEmit && vite build -> ui/dist/
```

**Demo mode (mock API).** The UI uses an in-browser mock API in two cases: when `config.json` has an empty `apiBase`, or when the URL contains `?mock=1` (also `#/?mock=1`). The mock has the same interface as the real one. It is deterministic and adds simulated latency. Its responses come from the `fake` engine, and the UI shows a "Demo mode" banner, so judges can still try the UI if the backend is down. In the mock, `demo_dram_hard` combined with **Run agent episode** escalates to the approval card. The other demos commit.

To point dev at a real API, set `apiBase` in `public/config.json` and do not commit that change.

### Contract types

`src/types/contracts.ts` is **generated** from the contract JSON Schemas (`driftsense/contracts.py` → `contracts/*.schema.json`):

```bash
npm run gen:types                         # reads ../contracts/*.schema.json
npm run gen:types -- --schemas <dir>      # any other export dir
npm run gen:types -- --check              # fail if the committed file is stale
```

The schema-validation test (`tests/trace.test.ts`) reads the schemas from `$DRIFTSENSE_SCHEMAS`, or from `../contracts` when that variable is unset. If neither has them, the test is skipped. The hand-written structural check still runs either way.

### Sample traces (SYNTHETIC)

Two sample traces are committed: `src/fixtures/sample_trace.jsonl` (the commit story) and `src/fixtures/sample_trace_escalate.jsonl` (escalate → human_approval). Both are valid `TraceEvent` JSONL. Every event has `result_summary.synthetic = true`. The UI shows a **SYNTHETIC SAMPLE** badge on any trace containing such events. Demo mode replays these files until real traces exist (G5).

## Deploy contract (infra)

1. `npm ci && npm run build` in `ui/`.
2. Upload `ui/dist/` to the UI bucket root (`index.html`, `assets/*`, `config.json`).
   - `assets/*` are content-hashed. Serve them with `Cache-Control: public, max-age=31536000, immutable`.
   - `index.html` and `config.json` must get `Cache-Control: no-cache`.
3. **Overwrite `config.json`** with the deployed API base. Use no trailing slash. CORS must allow the CloudFront origin.

   ```json
   { "apiBase": "https://<api-id>.execute-api.ap-south-1.amazonaws.com" }
   ```

4. Asset URLs are relative (`base: './'`), so the bundle works at the distribution root or under a prefix. Routing is hash-based, so no SPA rewrite rules are needed.
5. Suggested CSP: `default-src 'self'; img-src 'self' data: https:; connect-src 'self' <apiBase> https://*.amazonaws.com; style-src 'self'; script-src 'self'; frame-ancestors 'none'`. `data:` is needed for demo-mode SVG artifacts. The `https://*.amazonaws.com` entry covers presigned S3 PUT/GET URLs.

## API assumptions: handoffs for the service

The UI mirrors contract **1.0.0** exactly. The points below are **assumptions** that the service (B2 / I-G5) must honour or tell us to change:

| # | Assumption | Where in UI |
| --- | --- | --- |
| A1 | `GET /healthz`, `GET /v1/version` → `{git_sha, contract_version, service_version, cv2_version?, cv2_file?, python, arch}` | footer |
| A2 | `GET /v1/demos` → `[{id, title, family, difficulty, description}]` | home |
| A3 | `POST /v1/uploads {filename, content_type, size_bytes}` → `{upload_id, key, url, headers, expires_in}`. The browser then sends `PUT url` with exactly `headers`. Each returned `key` becomes `template_key` / `search_key`. Only `image/png` and `image/tiff` up to 20 MB are accepted, and the client checks this too. The S3 bucket CORS must allow `PUT` from the UI origin. | home |
| A4 | `POST /v1/jobs` (JobCreate, with a new `client_token` UUID per click) → 202 JobStatus | home |
| A5 | `GET /v1/jobs/{id}` → JobStatus. The UI polls with backoff 0.5 → 0.75 → 1.1 → 1.7 → 2 s. It resets to 0.5 s on each status change, stops on `succeeded`/`failed`, pauses on `awaiting_approval`, and gives up after 5 consecutive errors (Retry button). | live run |
| A6 | `POST /v1/jobs/{id}/approval` (ApprovalDecision) returns the updated JobStatus. If the body is empty or not a JobStatus, the UI re-GETs the job. | approval card |
| A7 | **Artifact URL route.** If `Artifact.uri` starts with `http(s)://` (e.g. a presigned GET), the UI uses it as-is. Otherwise it requests `GET {apiBase}/v1/jobs/{id}/artifacts/{uri}`. For `s3://bucket/key` it uses only the key's file name. The service must serve the bytes from that route or 302 to a presigned URL. | result panel, approval card |
| A8 | **Trace route.** When `JobStatus.trace_key` is set, the UI calls `GET {apiBase}/v1/jobs/{id}/trace`, which must return the JSONL of TraceEvent (`application/x-ndjson` or `text/plain`). A 404 means "not available yet". Bad lines are skipped and reported by line number. The UI re-fetches the trace on every poll while the job runs. | timeline |
| A9 | **Dose in traces.** `result_summary.dose` is the virtual SEM's **cumulative** dose counter. `result_summary.dose_budget` and `dose_unit` are optional. The timeline carries the last value forward. | timeline |
| A10 | **Reject all.** After `chosen_rank: null`, the mock ends the job as `failed` with `error.code = "rejected_by_operator"`, because the status enum has no "rejected" value. The UI shows it as "Rejected by operator". If the service picks something else, tell us. | live run |
| A11 | **Contract gap: alt text.** `Artifact` has no alt/description field. The UI generates alt text from `kind` and the job's result. Suggestion: add an optional `alt: str` in a minor contract bump, filled by S9 Explain. | result panel |
| A12 | **Contract gap: agent flag.** `JobStatus` does not echo `agent`. The UI shows the trace panel whenever `trace_key` is set. | live run |

## Accessibility

- A skip link comes first in the tab order. The page has landmarks (`header`/`nav`/`main`/`footer`) and a single `h1` per view. Focus moves to the new heading on navigation.
- Everything works by keyboard. The demo picker uses native radio inputs (arrow keys). The confirm step can be cancelled with Escape. Every control has a visible focus ring and is at least 44 px tall.
- A polite ARIA live region announces status changes, run start and approval results.
- Every input has a `<label>`. Errors are linked with `aria-describedby` and `aria-invalid`.
- Meaning never depends on colour alone. Status, decision and kind badges all carry icons and text. The margin bar has a text equivalent and a hatched fill when below τ. Score colours use the viridis map, and the number is always printed.
- Light and dark themes follow `prefers-color-scheme`. `tests/contrast.test.ts` checks every text token pair for at least 4.5:1, and borders for at least 3:1.
- The layout works at 360 px with no horizontal page scroll (tables scroll inside their own region). `prefers-reduced-motion` and `forced-colors` are respected.
