"""
DriftSense web demo — a small, self-contained FastAPI app.

Pick a built-in dataset pair (or upload your own reference + search SEM images),
run the pipeline, and see the annotated result: classical NCC vs DriftSense, the
reciprocal lattice, and the LER fingerprint decision.

    pip install -r demo/requirements-demo.txt
    uvicorn demo.app:app --host 0.0.0.0 --port 8000
    # open http://localhost:8000

No GPU, no toolchain, no network — it runs anywhere Python does.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from demo.visualize_result import render, _load  # noqa: E402

app = FastAPI(title="DriftSense")

DATASET = ROOT / "outputs" / "dataset"
TIERS = ["hard", "nominal", "clean"]


def _manifest(tier: str) -> dict:
    p = DATASET / tier / "manifest.json"
    if not p.exists():
        return {}
    return {m["pair_id"]: m for m in json.load(open(p))["pairs"]}


def _b64_png(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>DriftSense</title>
<style>
:root{{--bg:#0b1220;--card:#111a2e;--ink:#e8eefc;--mut:#9fb0d0;--acc:#16A34A;--edge:#22314f}}
@media(prefers-color-scheme:light){{:root{{--bg:#f4f7fc;--card:#fff;--ink:#0b1220;--mut:#51617f;--edge:#dbe3f0}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 2px}}.sub{{color:var(--mut);margin:0 0 20px}}
.card{{background:var(--card);border:1px solid var(--edge);border-radius:14px;padding:18px;margin-bottom:18px}}
label{{font-weight:600;font-size:13px;color:var(--mut);display:block;margin-bottom:6px}}
select,input[type=file]{{width:100%;padding:9px;border:1px solid var(--edge);border-radius:9px;
background:transparent;color:var(--ink)}}
.row{{display:flex;gap:16px;flex-wrap:wrap}}.row>div{{flex:1;min-width:220px}}
button{{margin-top:14px;background:var(--acc);color:#fff;border:0;border-radius:10px;
padding:11px 20px;font-weight:700;font-size:15px;cursor:pointer}}
.result img{{width:100%;border-radius:10px;border:1px solid var(--edge)}}
.metrics{{display:flex;gap:22px;flex-wrap:wrap;margin:6px 0 14px;font-size:14px}}
.metrics b{{font-size:20px;display:block}}.ok{{color:var(--acc)}}.muted{{color:var(--mut)}}
hr{{border:0;border-top:1px solid var(--edge);margin:18px 0}}
</style></head><body><div class=wrap>
<h1>DriftSense</h1>
<p class=sub>Physics-aware localization of a reference patch inside a periodic SEM wafer image.</p>
<form class=card method=post action=/run enctype=multipart/form-data>
  <div class=row>
    <div><label>Built-in dataset pair</label>
      <select name=tier>{tier_opts}</select></div>
    <div><label>Pair ID (0–39)</label>
      <select name=pair>{pair_opts}</select></div>
  </div>
  <hr><p class=muted style=margin:0>…or upload your own SEM images:</p>
  <div class=row style=margin-top:10px>
    <div><label>Reference patch</label><input type=file name=ref accept=image/*></div>
    <div><label>Search image</label><input type=file name=search accept=image/*></div>
  </div>
  <button type=submit>Locate patch →</button>
</form>
{result}
</div></body></html>"""


def _form(result_html: str = "") -> str:
    tier_opts = "".join(f"<option value={t}>{t}</option>" for t in TIERS)
    pair_opts = "".join(f"<option value={i}>{i}</option>" for i in range(40))
    return PAGE.format(tier_opts=tier_opts, pair_opts=pair_opts, result=result_html)


@app.get("/", response_class=HTMLResponse)
def index():
    return _form()


@app.post("/run", response_class=HTMLResponse)
async def run(tier: str = Form("hard"), pair: int = Form(0),
              ref: UploadFile = File(None), search: UploadFile = File(None)):
    tmp = Path(tempfile.mkdtemp())
    gt = None
    if ref is not None and ref.filename and search is not None and search.filename:
        rp, sp = tmp / "ref.png", tmp / "search.png"
        rp.write_bytes(await ref.read()); sp.write_bytes(await search.read())
        ref_img, search_img = _load(rp), _load(sp)
        subtitle = "uploaded images"
    else:
        d = DATASET / tier
        meta = _manifest(tier).get(int(pair))
        ref_img = _load(d / f"pair_{int(pair):04d}_ref.png")
        search_img = _load(d / f"pair_{int(pair):04d}_search.png")
        gt = (meta["gt_x"], meta["gt_y"]) if meta else None
        subtitle = f"{tier} · pair {pair}"

    out = tmp / "result.png"
    info = render(ref_img, search_img, str(out), gt=gt, title_extra=subtitle)
    img64 = _b64_png(str(out))

    if info["ds_error"] is not None:
        located = info["ds_error"] <= 5
        m = (f"<div class=metrics>"
             f"<div>NCC error<b class=muted>{info['ncc_error']:.0f} px</b></div>"
             f"<div>DriftSense error<b class=ok>{info['ds_error']:.1f} px</b></div>"
             f"<div>Result<b class={'ok' if located else 'muted'}>"
             f"{'✓ located' if located else 'miss'}</b></div>"
             f"<div>Candidates<b class=muted>{info['n_candidates']}</b></div></div>")
    else:
        m = (f"<div class=metrics><div>Prediction<b class=ok>"
             f"({info['pred'][0]:.0f}, {info['pred'][1]:.0f})</b></div>"
             f"<div>Candidates<b class=muted>{info['n_candidates']}</b></div></div>")

    result = (f"<div class='card result'><h3 style=margin-top:0>Result — {subtitle}</h3>"
              f"{m}<img src='data:image/png;base64,{img64}'></div>")
    return _form(result)
