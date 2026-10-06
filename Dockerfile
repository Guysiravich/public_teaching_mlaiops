# ITCS355 Lab 1 — training image
#
# Base image pinned BY DIGEST, not by tag (Lab 1, Task 2).
#   Tags move. `python:3.11-slim` today is not `python:3.11-slim` next month; a digest
#   names exact bytes. The tag stays beside the digest for readability only.
#   Refresh with:
#       docker pull python:3.11-slim && docker inspect --format='{{index .RepoDigests 0}}' python:3.11-slim
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534 AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Dependencies first so this layer caches independently of your source.
COPY requirements.txt ./
# --require-hashes: every package must match its hash in requirements.txt (Lab 1, Task 2).
# It turns a silently-substituted package into a build failure, which is what you want.
RUN pip install --require-hashes --prefix=/install -r requirements.txt


FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534 AS runtime

# Non-root. A training container has no reason to run as root, and graders check.
RUN useradd --create-home --uid 10001 runner
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

COPY --from=builder /install /usr/local
WORKDIR /app
COPY --chown=runner:runner src/ ./src/
COPY --chown=runner:runner cloudlayer/ ./cloudlayer/
COPY --chown=runner:runner scripts/ ./scripts/
# Lab 4: the scheduled drift job runs from this image (python -m monitoring.drift_job).
COPY --chown=runner:runner monitoring/ ./monitoring/

# MLflow logs artifacts to ./mlruns relative to WORKDIR. /app is root-owned,
# so the non-root runner cannot create it. Make the workdir writable.
RUN mkdir -p /app/mlruns && chown -R runner:runner /app

USER runner

# Credentials NEVER enter an image layer. They arrive at runtime from SECRET_STORE_PATH
# or from the platform's identity. If you find yourself adding an ARG for a key, stop.
ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--n-estimators", "200", "--max-depth", "8"]
