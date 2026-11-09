# Jev-Mem Dify sidecar (the ONLY part that calls the LLM / Jev / embedding APIs).
#
# Build from the repo root:
#   docker build -f integrations/dify/service/Dockerfile -t jev-mem-sidecar .
#
# Run (mount your real .env + persistent memory data):
#   docker run -d --name jev-mem \
#     -v /host/integrations/dify/service/.env:/app/integrations/dify/service/.env:ro \
#     -v jev-mem-data:/data \
#     -p 8700:8700 jev-mem-sidecar
#
# Endpoints inside the container must use host.docker.internal (or a shared
# docker network), NOT localhost, to reach the model server on the host.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JEV_MEM_CACHE_DIR=/data

WORKDIR /app

# Core repo dependencies + the service's own (fastapi/uvicorn).
# NOTE: requirements pull in torch + sentence-transformers (local minilm
# fallback), so the image is large. If you only use API embeddings,
# trimming those lines is possible but diverges from the local dev env.
COPY requirements.txt ./
COPY integrations/dify/service/requirements.txt ./service-requirements.txt
RUN pip install --no-cache-dir \
      -r requirements.txt -r service-requirements.txt

# Source tree the service imports (app.py inserts JEV_MEM_HOME=/app into
# sys.path). Selective COPY on purpose: a blanket "COPY . /app" would bake
# the real .env (secrets) into the image — the .dockerignore also guards
# against that.
COPY jev_mem ./jev_mem
COPY memory ./memory
COPY utils ./utils
COPY config ./config
COPY integrations/dify/service/app.py integrations/dify/service/app.py

# Non-root runtime user; /data (JEV_MEM_CACHE_DIR) must be its writable home
# for the named-volume ownership to initialize correctly.
RUN useradd --create-home app && mkdir -p /data && chown app:app /data

USER app
WORKDIR /app/integrations/dify/service

EXPOSE 8700
VOLUME ["/data"]

# --host 0.0.0.0 is required: the default 127.0.0.1 is unreachable from
# outside the container. JEV_MEM_HOME is not set on purpose — the app
# resolves it to /app (three levels above app.py) automatically.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8700"]
