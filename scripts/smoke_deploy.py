#!/usr/bin/env python3
"""Post-deploy smoke test for a DriftSense API (deployed stack or the local dev server).

Usage:
  python scripts/smoke_deploy.py --api https://<id>.execute-api.ap-south-1.amazonaws.com
                                 [--ui https://<dist>.cloudfront.net] [--agent] [--expect-cool]
                                 [--timeout 300] [--out results-smoke.json]
  python scripts/smoke_deploy.py --outputs infra/cdk-outputs.json --agent --expect-cool
  python scripts/smoke_deploy.py --api http://127.0.0.1:8000 --agent     # python -m service.dev_server

Checks, in order (stdlib only, so it runs on any machine with Python; no jq, curl or AWS CLI):
  healthz       GET /healthz returns {"ok": true}
  version       GET /v1/version answers with a git sha and contract version
  demos         GET /v1/demos lists at least one localize demo
  job           POST /v1/jobs on the first localize demo reaches "succeeded"
  worker        that job's JobStatus.worker is reported (with --expect-cool: cool, aarch64, c8g.*)
  artifact      the first result artifact downloads through /v1/jobs/{id}/artifacts/{name}
  episode       (--agent) POST /v1/agent/episodes; an approval card is answered with "reject all"
  trace         (--agent) GET /v1/jobs/{id}/trace returns JSONL events
  ui / config   (--ui) the SPA index loads and /config.json points at --api

Exit codes: 0 every check passed; 1 a check failed; 2 bad arguments. --out writes the evidence
(responses, timings, worker info) as JSON with UTC and IST timestamps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

IST = timezone(timedelta(hours=5, minutes=30))
TERMINAL = ("succeeded", "failed")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    data: Any = None


@dataclass
class Report:
    api: str
    started_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    started_ist: str = field(default_factory=lambda: datetime.now(IST).isoformat(timespec="seconds"))
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str, data: Any = None) -> bool:
        self.checks.append(Check(name, ok, detail, data))
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<9} {detail}", flush=True)
        return ok


def http(method: str, url: str, body: Any = None, timeout: float = 30.0) -> tuple[int, bytes]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def get_json(url: str) -> tuple[int, Any]:
    code, raw = http("GET", url)
    try:
        return code, json.loads(raw or b"null")
    except json.JSONDecodeError:
        return code, raw[:200].decode(errors="replace")


def poll(api: str, job_id: str, timeout: float, until: tuple[str, ...] = TERMINAL) -> dict:
    deadline, status = time.monotonic() + timeout, {}
    while time.monotonic() < deadline:
        _, status = get_json(f"{api}/v1/jobs/{job_id}")
        if isinstance(status, dict) and status.get("status") in until:
            return status
        time.sleep(2)
    return status if isinstance(status, dict) else {"status": "timeout"}


def check_job(rep: Report, api: str, demos: list[dict], args: argparse.Namespace) -> None:
    demo = next((d for d in demos if not d.get("agent")), None)
    if not rep.add(
        "demos", demo is not None, f"{len(demos)} demo(s); localize demo: {demo and demo['demo_id']}"
    ):
        return
    token = f"smoke-{int(time.time())}"
    t0 = time.monotonic()
    code, raw = http("POST", f"{api}/v1/jobs", {"demo_id": demo["demo_id"], "client_token": token})
    if not rep.add("submit", code in (200, 202), f"POST /v1/jobs -> {code}"):
        return
    job = json.loads(raw)
    status = poll(api, job["job_id"], args.timeout)
    dt = time.monotonic() - t0
    result = status.get("result") or {}
    offset = result.get("offset") or {}
    ok = rep.add(
        "job",
        status.get("status") == "succeeded",
        f"{job['job_id']} {status.get('status')} in {dt:.1f}s; engine={status.get('engine')} "
        f"decision={(result.get('margin') or {}).get('decision')} "
        f"offset_px=({offset.get('x_px')}, {offset.get('y_px')})",
        {"job_id": job["job_id"], "seconds": round(dt, 2), "status": status},
    )
    worker = status.get("worker") or {}
    if args.expect_cool:
        wok = (
            worker.get("cool") is True
            and worker.get("arch") in ("aarch64", "arm64")
            and str(worker.get("instance_type", "")).startswith("c8g")
        )
    else:
        wok = bool(worker)
    rep.add("worker", wok, json.dumps(worker, sort_keys=True), worker)
    if not ok:
        return
    arts = result.get("artifacts") or []
    if arts:
        name = str(arts[0].get("uri", "")).rsplit("/", 1)[-1]
        code, body = http("GET", f"{api}/v1/jobs/{job['job_id']}/artifacts/{name}")
        rep.add("artifact", code == 200 and len(body) > 0, f"{name}: HTTP {code}, {len(body)} bytes")
    else:
        rep.add("artifact", False, "the job result lists no artifacts")


def check_episode(rep: Report, api: str, demos: list[dict], args: argparse.Namespace) -> None:
    demo = next((d for d in demos if d.get("agent")), None)
    family, seed = (demo["family"], demo["seed"]) if demo else ("finfet", 1)
    body = {"family": family, "seed": seed, "client_token": f"smoke-ep-{int(time.time())}"}
    code, raw = http("POST", f"{api}/v1/agent/episodes", body)
    if not rep.add("episode", code in (200, 202), f"POST /v1/agent/episodes {family}/{seed} -> {code}"):
        return
    job_id = json.loads(raw)["job_id"]
    status = poll(api, job_id, args.timeout, until=(*TERMINAL, "awaiting_approval"))
    approved = False
    if status.get("status") == "awaiting_approval":
        decision = {"chosen_rank": None, "approver": "smoke_deploy", "note": "smoke test: reject all"}
        code, _ = http("POST", f"{api}/v1/jobs/{job_id}/approval", decision)
        approved = rep.add("approval", code == 200, f"approval card answered (reject all) -> {code}")
        status = poll(api, job_id, args.timeout)
    rep.add(
        "episode",
        status.get("status") == "succeeded",
        f"{job_id} {status.get('status')}" + (" after human approval" if approved else ""),
        {"job_id": job_id, "status": status},
    )
    code, raw = http("GET", f"{api}/v1/jobs/{job_id}/trace")
    events = [ln for ln in raw.decode(errors="replace").splitlines() if ln.strip()] if code == 200 else []
    rep.add("trace", bool(events), f"HTTP {code}, {len(events)} trace event(s)")


def check_ui(rep: Report, api: str, ui: str) -> None:
    code, body = http("GET", f"{ui}/")
    rep.add("ui", code == 200 and b"<html" in body.lower(), f"GET {ui}/ -> {code}, {len(body)} bytes")
    code, cfg = get_json(f"{ui}/config.json")
    base = cfg.get("apiBase", "").rstrip("/") if isinstance(cfg, dict) else ""
    rep.add("config", code == 200 and base == api, f"/config.json apiBase={base!r} (expected {api!r})")


def from_outputs(path: Path) -> tuple[str, str | None]:
    out = json.loads(path.read_text(encoding="utf-8"))
    api = out.get("DriftSense-Api", {}).get("ApiUrl")
    ui = out.get("DriftSense-Storage", {}).get("UiUrl")
    if not api:
        raise SystemExit(f"smoke_deploy: no DriftSense-Api.ApiUrl in {path}")
    return api, ui


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--api", help="API base URL")
    ap.add_argument("--ui", help="UI (CloudFront) base URL")
    ap.add_argument("--outputs", type=Path, help="cdk-outputs.json from `cdk deploy --outputs-file`")
    ap.add_argument("--agent", action="store_true", help="also run an agent episode through approval")
    ap.add_argument("--expect-cool", action="store_true", help="require the COOL Graviton worker")
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per job (default 300)")
    ap.add_argument("--out", type=Path, help="write the evidence JSON here")
    args = ap.parse_args(argv)
    api, ui = args.api, args.ui
    if args.outputs:
        api_o, ui_o = from_outputs(args.outputs)
        api, ui = api or api_o, ui or ui_o
    if not api:
        ap.error("--api or --outputs is required")
    api, ui = api.rstrip("/"), ui.rstrip("/") if ui else None

    rep = Report(api=api)
    try:
        code, health = get_json(f"{api}/healthz")
        if rep.add(
            "healthz", code == 200 and isinstance(health, dict) and health.get("ok"), f"HTTP {code} {health}"
        ):
            code, ver = get_json(f"{api}/v1/version")
            vok = code == 200 and isinstance(ver, dict) and bool(ver.get("git_sha"))
            detail = (
                f"git_sha={ver.get('git_sha')} engine={ver.get('engine')} arch={ver.get('arch')}"
                if vok
                else f"HTTP {code}"
            )
            rep.add("version", vok, detail, ver)
            code, demos = get_json(f"{api}/v1/demos")
            demos = demos if code == 200 and isinstance(demos, list) else []
            check_job(rep, api, demos, args)
            if args.agent:
                check_episode(rep, api, demos, args)
        if ui:
            check_ui(rep, api, ui)
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        rep.add("error", False, f"{type(exc).__name__}: {exc}")

    failed = [c.name for c in rep.checks if not c.ok]
    print(
        f"\nsmoke_deploy: {len(rep.checks) - len(failed)}/{len(rep.checks)} checks passed"
        + (f"; failed: {', '.join(failed)}" if failed else "")
    )
    if args.out:
        doc = asdict(rep) | {"passed": not failed}
        args.out.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"evidence written to {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
