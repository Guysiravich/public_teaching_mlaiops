"""Azure adapter. Implement upload/download/push_image for Lab 1.

SDK:  pip install azure-storage-blob azure-identity azure-containerregistry
Docs: BlobServiceClient for storage; ACR push goes through `docker push` after
      `az acr login --name <registry>`.

Hints for Lab 1:
  * BLOB_URI is either abfss://container@account.dfs.core.windows.net/prefix or
    https://account.blob.core.windows.net/container/prefix. Pick one form and parse
    it here, never in src/.
  * Use DefaultAzureCredential rather than a connection string. It picks up your CLI
    login locally and your managed identity in CI, which is what Lab 4 needs.
  * push_image must return the digest reference: registry.azurecr.io/repo@sha256:...
  * Azure tags live on the resource, not the blob. Tag the storage account, the
    registry, and later the workspace with cfg.tags(1).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cloudlayer.base import CloudAdapter

# This adapter uses the https form of BLOB_URI:
#   https://<account>.blob.core.windows.net/<container>/<path>
_BLOB_HOST_SUFFIX = ".blob.core.windows.net"
_DIGEST = re.compile(r"digest:\s*(sha256:[0-9a-f]{64})")


def _parse_blob_uri(uri: str) -> tuple[str, str, str]:
    """Split a blob URI into (account, container, path). Path may be empty."""
    parsed = urlparse(uri)
    if parsed.scheme != "https" or not parsed.netloc.endswith(_BLOB_HOST_SUFFIX):
        raise ValueError(
            f"Expected https://<account>{_BLOB_HOST_SUFFIX}/<container>/<path>, got {uri!r}"
        )
    account = parsed.netloc[: -len(_BLOB_HOST_SUFFIX)]
    container, _, path = parsed.path.lstrip("/").partition("/")
    if not account or not container:
        raise ValueError(f"Blob URI {uri!r} is missing the account or the container")
    return account, container, path.strip("/")


def _blob_url(account: str, container: str, path: str) -> str:
    return f"https://{account}{_BLOB_HOST_SUFFIX}/{container}/{path}"


def _run(cmd: list[str]) -> str:
    """Run a CLI command; raise with its output if it fails."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(cmd)}` failed ({result.returncode}):\n{result.stdout}{result.stderr}"
        )
    return result.stdout + result.stderr


class AzureAdapter(CloudAdapter):
    def _blob_service(self, account: str) -> Any:
        # Imported here so selecting the adapter does not require the SDK to be installed.
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        return BlobServiceClient(
            account_url=f"https://{account}{_BLOB_HOST_SUFFIX}",
            credential=DefaultAzureCredential(),
        )

    def upload(self, local_path: str, key: str) -> str:
        account, container, prefix = _parse_blob_uri(self.cfg.blob_uri)
        blob_name = "/".join(part for part in (prefix, key.strip("/")) if part)
        blob = self._blob_service(account).get_blob_client(container=container, blob=blob_name)
        with open(local_path, "rb") as fh:
            blob.upload_blob(fh, overwrite=True)
        return _blob_url(account, container, blob_name)

    def download(self, uri: str, local_path: str) -> None:
        account, container, blob_name = _parse_blob_uri(uri)
        if not blob_name:
            raise ValueError(f"Blob URI {uri!r} names a container, not an object")
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob = self._blob_service(account).get_blob_client(container=container, blob=blob_name)
        with dest.open("wb") as fh:
            blob.download_blob().readinto(fh)

    def push_image(self, local_tag: str) -> str:
        registry = self.cfg.container_registry.strip("/")  # <name>.azurecr.io/<namespace>
        login_server = registry.split("/", 1)[0]
        registry_name = login_server.split(".", 1)[0]

        name, sep, tag = local_tag.rpartition(":")
        if not sep or "/" in tag:  # no tag given
            name, tag = local_tag, "latest"
        remote_repo = f"{registry}/{name.rsplit('/', 1)[-1]}"
        remote_tag = f"{remote_repo}:{tag}"

        _run(["az", "acr", "login", "--name", registry_name])
        _run(["docker", "tag", local_tag, remote_tag])
        output = _run(["docker", "push", remote_tag])

        match = _DIGEST.search(output)
        if not match:
            raise RuntimeError(f"`docker push` reported no digest for {remote_tag}:\n{output}")
        return f"{remote_repo}@{match.group(1)}"

    # submit_training / register_model  -> Lab 2 (Azure ML command job + model registry)
    # deploy / invoke                   -> Lab 3 (managed online endpoint + deployment)
    # emit_metric                       -> Lab 4 (Azure Monitor custom metric)
    # generate                          -> Lab 5 (managed LLM endpoint; read the usage block for tokens)
    # teardown                          -> Lab 5 (resource graph query by tag)
