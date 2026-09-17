# MLflow tracking server (Lab 2). Same base image, pinned by digest, as the training image.
FROM python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1

# mlflow matches requirements.txt. psycopg2 for the PostgreSQL backend store; the Azure
# packages let the server write artifacts to Blob Storage with its managed identity.
RUN pip install \
      mlflow==3.16.0 \
      psycopg2-binary==2.9.13 \
      azure-storage-blob==12.30.1 \
      azure-identity==1.25.3

RUN useradd --create-home --uid 10002 mlflow && mkdir -p /artifacts && chown mlflow /artifacts
USER mlflow
EXPOSE 5000
ENTRYPOINT ["mlflow", "server"]
