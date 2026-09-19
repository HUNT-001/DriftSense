# DriftSense demo

Deployable, CPU-only. No GPU, no toolchain, no network.

## Install

```bash
pip install -r requirements.txt          # core
pip install -r demo/requirements-demo.txt # web app only
```

## Command line

```bash
python -m demo.cli locate \
    --ref    outputs/dataset/hard/pair_0016_ref.png \
    --search outputs/dataset/hard/pair_0016_search.png \
    --figure outputs/figures/my_result.png
```

Prints the predicted `(x, y)`, score, candidate count and runtime as JSON; with
`--figure` also writes the annotated montage.

## Web app

```bash
uvicorn demo.app:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

Pick a built-in dataset pair (hard / nominal / clean × 40 pairs) or upload your
own reference + search SEM images, then hit **Locate patch**. You get the
classical-NCC-vs-DriftSense overlay, the reciprocal lattice, and the LER decision.

## Result montages

```bash
python -m demo.visualize_result --tier hard --pair 16 \
    --out outputs/figures/demo_pair16.png
```

## What "deployable" means here

Everything runs in one Python process on a CPU. The pipeline is classical
CV/DSP — there is no model to serve, no weights to ship, and fab image data never
leaves the host. Package `demo/cli.py` as an on-prem microservice, or call
`localization.ler_localizer.localize(...)` directly from an existing tool.
