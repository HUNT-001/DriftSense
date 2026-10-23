#!/usr/bin/env python3
"""Upload the built SPA (ui/dist) to the deployed UI bucket and invalidate CloudFront.

Usage:
  python scripts/publish_ui.py [--outputs infra/cdk-outputs.json] [--dist ui/dist] [--dry-run]

Python + boto3 equivalent of step 4 of the infra/README.md runbook (no jq or AWS CLI needed;
credentials come from the usual boto3 chain, e.g. AWS_PROFILE after `aws sso login`).
Reads DriftSense-Api.ApiUrl and DriftSense-Storage.{UiBucketName, UiDistributionId, UiUrl} from
the `cdk deploy --outputs-file` JSON, writes dist/config.json = {"apiBase": <ApiUrl>}, uploads
hashed assets as immutable and index.html/config.json as no-cache, deletes stale objects, and
invalidates /index.html and /config.json.

Exit codes: 0 ok; 1 AWS error; 2 bad input (missing outputs or dist).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
from pathlib import Path

NO_CACHE = ("index.html", "config.json")
IMMUTABLE = "public,max-age=31536000,immutable"
# Explicit types: the OS mimetypes table varies (Windows lacks .webp/.jsonl, maps .js oddly).
TYPES = {
    ".js": "text/javascript",
    ".css": "text/css",
    ".html": "text/html",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".webp": "image/webp",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}


def read_outputs(path: Path) -> dict[str, str]:
    out = json.loads(path.read_text(encoding="utf-8"))
    api, storage = out.get("DriftSense-Api", {}), out.get("DriftSense-Storage", {})
    vals = {
        "api": api.get("ApiUrl", "").rstrip("/"),
        "bucket": storage.get("UiBucketName", ""),
        "dist_id": storage.get("UiDistributionId", ""),
        "ui_url": storage.get("UiUrl", ""),
    }
    missing = [k for k, v in vals.items() if not v]
    if missing:
        raise SystemExit(f"publish_ui: {path} lacks {missing} (deploy with --outputs-file)")
    return vals


def plan(dist: Path) -> list[tuple[str, Path, str, str]]:
    """(key, file, content_type, cache_control) for every file under dist, no-cache files last."""
    items = []
    for f in sorted(p for p in dist.rglob("*") if p.is_file()):
        key = f.relative_to(dist).as_posix()
        ctype = TYPES.get(f.suffix.lower()) or mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        items.append((key, f, ctype, "no-cache" if key in NO_CACHE else IMMUTABLE))
    return sorted(items, key=lambda it: it[0] in NO_CACHE)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outputs", type=Path, default=Path("infra/cdk-outputs.json"))
    ap.add_argument("--dist", type=Path, default=Path("ui/dist"))
    ap.add_argument("--dry-run", action="store_true", help="print the upload plan only")
    args = ap.parse_args(argv)
    if not args.outputs.is_file():
        print(f"publish_ui: {args.outputs} not found", file=sys.stderr)
        return 2
    if not (args.dist / "index.html").is_file():
        print(f"publish_ui: {args.dist}/index.html not found; run `npm run build` in ui/", file=sys.stderr)
        return 2
    cfg = read_outputs(args.outputs)
    (args.dist / "config.json").write_text(json.dumps({"apiBase": cfg["api"]}) + "\n", encoding="utf-8")
    items = plan(args.dist)
    for key, _, ctype, cache in items:
        print(f"{'would put' if args.dry_run else 'put'} s3://{cfg['bucket']}/{key} ({ctype}; {cache})")
    if args.dry_run:
        return 0

    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        s3 = boto3.client("s3")
        for key, f, ctype, cache in items:
            s3.upload_file(
                str(f), cfg["bucket"], key, ExtraArgs={"ContentType": ctype, "CacheControl": cache}
            )
        keep = {key for key, *_ in items}
        stale = [
            obj["Key"]
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=cfg["bucket"])
            for obj in page.get("Contents", [])
            if obj["Key"] not in keep
        ]
        for i in range(0, len(stale), 1000):
            batch = [{"Key": k} for k in stale[i : i + 1000]]
            s3.delete_objects(Bucket=cfg["bucket"], Delete={"Objects": batch, "Quiet": True})
        boto3.client("cloudfront").create_invalidation(
            DistributionId=cfg["dist_id"],
            InvalidationBatch={
                "Paths": {"Quantity": 2, "Items": ["/index.html", "/config.json"]},
                "CallerReference": f"publish-ui-{int(time.time())}",
            },
        )
    except (BotoCoreError, ClientError) as exc:
        print(f"publish_ui: {exc}", file=sys.stderr)
        return 1
    print(f"uploaded {len(items)} file(s), removed {len(stale)} stale; UI: {cfg['ui_url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
