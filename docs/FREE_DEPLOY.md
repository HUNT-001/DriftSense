# Free deployment (no AWS account)

The AWS CDK deployment in [infra/README.md](../infra/README.md) is the production path. If you
only want a live URL, the same app (real API, real OpenCV 5 worker, web UI, agent episodes with
the approval card) runs in one container on free hosts. Browser uploads are the only feature
that needs real S3; the demo sites and agent episodes work.

## Option 1: Hugging Face Space (recommended: free, no card, 16 GB RAM, stays up)

1. Create a Hugging Face account, then **New Space** -> SDK **Docker** -> **Blank** -> CPU basic (free).
2. In the Space, **Files -> Add file -> Upload**, and upload the two files in
   [`deploy/hf-space/`](../deploy/hf-space/): `README.md` and `Dockerfile`.
3. The Space builds from the GitHub repo (about 5 minutes) and serves the app at
   `https://<user>-<space>.hf.space`. To pin a revision, set the build argument `REF` to a tag or branch.

## Option 2: Render (free web service)

New -> **Blueprint** -> select this repository. Render reads [`render.yaml`](../render.yaml)
and builds the root [`Dockerfile`](../Dockerfile). Free services sleep when idle, so the first
request after a pause takes about a minute.

## Option 3: your own machine

```bash
docker build -t driftsense . && docker run -p 7860:7860 driftsense   # http://localhost:7860
# or, without Docker:
python -m service.dev_server --ui ui/dist                            # http://127.0.0.1:8000
```

## Static replay only

`ui/dist` also works as a plain static site (Netlify, Vercel or GitHub Pages: drag and drop the
folder after `cd ui && npm ci && npm run build`). With no API configured it offers the offline
replay of two recorded episodes.
