"""Local live stack with no AWS account: ``python -m service.dev_server [--port 8000] [--ui ui/dist]``.

Runs the real API (FastAPI under uvicorn) and the real SQS worker on a thread, against in-process
moto fakes of DynamoDB/SQS/S3. Demo jobs and agent episodes run the real v2 engine and the real
``driftsense.agent.runner``, so the UI can drive a live episode end to end, including the human
approval card. With ``--ui`` it also serves the built SPA from the same origin, plus a
``/config.json`` pointing the SPA at this server (use it for the video or as the judging fallback).

Not supported: browser uploads. Presigned PUT URLs point at real S3, which moto's in-process mock
cannot receive. Use the demo sites. Dev-only: needs moto (never imported by the API or the worker)."""

from __future__ import annotations

import argparse
import logging
import os
import threading
from dataclasses import replace
from pathlib import Path

from fastapi import Request  # module level: string annotations resolve against module globals

from service.e2e_local import LOCAL_ENV, provision

log = logging.getLogger("service.dev_server")


def build(ui_dir: Path | None):
    """Provision the fakes and return (app, worker). Call inside an active ``mock_aws()``."""
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles

    from service.api import Services, create_app
    from service.settings import Settings
    from service.worker import Worker

    env = os.environ
    env["JOBS_QUEUE_URL"] = provision(env["AWS_REGION"], env["JOBS_TABLE"], env["DATA_BUCKET"])
    # Snappy local polling; the deployed defaults stay as configured.
    settings = replace(Settings.from_env(), approval_poll_s=0.5, trace_upload_s=0.5)
    services = Services.from_settings(settings)
    app = create_app(settings=settings, services=services)
    worker = Worker(settings, jobs=services.jobs, artifacts=services.artifacts, queue=services.queue)
    if ui_dir is not None:

        @app.get("/config.json", include_in_schema=False)
        def ui_config(request: Request) -> JSONResponse:
            # Same origin as the page, whichever host name the browser used (localhost vs 127.0.0.1).
            return JSONResponse({"apiBase": str(request.base_url).rstrip("/")})

        app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")
    return app, worker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m service.dev_server", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ui", type=Path, default=None, help="built SPA directory (e.g. ui/dist)")
    parser.add_argument("--engine", default="v2", help="engine for plain localize jobs (fake|v1|v2)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.ui is not None and not (args.ui / "index.html").is_file():
        parser.error(f"{args.ui}/index.html not found: run `npm run build` in ui/ first")

    for key, value in LOCAL_ENV.items():
        os.environ.setdefault(key, value)
    os.environ["DRIFTSENSE_ENGINE"] = args.engine
    base_url = f"http://{args.host}:{args.port}"

    import uvicorn
    from moto import mock_aws

    with mock_aws():
        app, worker = build(args.ui)
        thread = threading.Thread(target=worker.run_forever, kwargs={"wait_seconds": 1}, daemon=True)
        thread.start()
        log.info("API on %s%s; worker polling the in-process queue", base_url, " (+UI)" if args.ui else "")
        try:
            uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        finally:
            worker.request_stop()
            thread.join(timeout=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
