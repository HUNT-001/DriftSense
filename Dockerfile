# Free, no-AWS deployment: the real API + worker + web UI in one container.
# docker build -t driftsense . && docker run -p 7860:7860 driftsense   -> http://localhost:7860
FROM node:22-slim AS ui
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860
WORKDIR /app
COPY requirements.lock ./
RUN pip install -r requirements.lock
COPY . .
RUN pip install -e . --no-deps
COPY --from=ui /ui/dist ui/dist
EXPOSE 7860
# service.dev_server = real FastAPI + real worker, with in-process fake DynamoDB/SQS/S3 (moto).
CMD ["sh", "-c", "python -m service.dev_server --host 0.0.0.0 --port ${PORT:-7860} --ui ui/dist"]
