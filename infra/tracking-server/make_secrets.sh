#!/usr/bin/env bash
# Create .env and auth.caddy for the tracking server. Both are gitignored; never commit them.
#
#   ./make_secrets.sh <hostname> <artifacts-destination>
#
# Prints the username and password ONCE. Put them in your cloud.env as
# MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD, and share them with the grader privately.
set -euo pipefail
cd "$(dirname "$0")"

HOSTNAME_ARG="${1:?usage: ./make_secrets.sh <hostname> <artifacts-destination>}"
ARTIFACTS_ARG="${2:?usage: ./make_secrets.sh <hostname> <artifacts-destination>}"
USERNAME="itcs355"

if [ -e .env ] || [ -e auth.caddy ]; then
  echo ".env or auth.caddy already exists; remove them first if you mean to rotate the secrets."
  exit 1
fi

rand() { head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c "$1"; }
POSTGRES_PASSWORD="$(rand 32)"
MLFLOW_PASSWORD="$(rand 24)"

CADDY_IMAGE="$(awk '/image: caddy/ {print $2}' docker-compose.yml)"
HASH="$(docker run --rm "$CADDY_IMAGE" caddy hash-password --plaintext "$MLFLOW_PASSWORD")"

umask 077
cat > .env <<ENV
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
MLFLOW_HOSTNAME=${HOSTNAME_ARG}
ARTIFACTS_DESTINATION=${ARTIFACTS_ARG}
ENV
printf 'basic_auth {\n\t%s %s\n}\n' "$USERNAME" "$HASH" > auth.caddy

echo "wrote .env and auth.caddy (mode 600)"
echo "MLFLOW_TRACKING_USERNAME=${USERNAME}"
echo "MLFLOW_TRACKING_PASSWORD=${MLFLOW_PASSWORD}"
