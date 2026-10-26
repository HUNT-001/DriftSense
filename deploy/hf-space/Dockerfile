# Hugging Face Space (Docker SDK): clones the repo and runs the same app as the root Dockerfile.
ARG REF=main
FROM node:22-slim AS ui
ARG REF
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${REF} https://github.com/HUNT-001/DriftSense /src
WORKDIR /src/ui
RUN npm ci && npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860
COPY --from=ui /src /app
WORKDIR /app
RUN pip install -r requirements.lock && pip install -e . --no-deps
RUN useradd -m -u 1000 user && chown -R user /app
USER user
EXPOSE 7860
CMD ["sh", "-c", "python -m service.dev_server --host 0.0.0.0 --port ${PORT:-7860} --ui ui/dist"]
