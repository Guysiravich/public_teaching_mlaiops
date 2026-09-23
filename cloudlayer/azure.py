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


def _job_uri(uri: str) -> str:
    """BLOB_URI form -> the form Azure ML jobs accept. Azure ML treats https:// as public,
    read-only storage and rejects it as an output (NotSupportedAssetOutputUri)."""
    account, container, path = _parse_blob_uri(uri)
    return f"wasbs://{container}@{account}{_BLOB_HOST_SUFFIX}/{path}"


def _run(cmd: list[str]) -> str:
    """Run a CLI command; raise with its output if it fails."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(cmd)}` failed ({result.returncode}):\n{result.stdout}{result.stderr}"
        )
    return result.stdout + result.stderr


_POLL_SECONDS = 30
# The Azure CLI can take over 10 s (the SDK default) to refresh a token from WSL.
_CLI_TIMEOUT_SECONDS = 60
_TRAINING_TARGET = re.compile(
    r"^/subscriptions/(?P<sub>[^/]+)/resourceGroups/(?P<rg>[^/]+)"
    r"/providers/Microsoft\.MachineLearningServices/workspaces/(?P<ws>[^/]+)"
    r"/computes/(?P<compute>[^/]+)$",
    re.IGNORECASE,
)


def _parse_training_target(target: str) -> tuple[str, str, str, str]:
    """TRAINING_TARGET is the compute's resource ID; split it into its four names."""
    m = _TRAINING_TARGET.match((target or "").strip())
    if not m:
        raise ValueError(
            "TRAINING_TARGET must be an Azure ML compute resource ID: /subscriptions/<id>/"
            "resourceGroups/<rg>/providers/Microsoft.MachineLearningServices/workspaces/<ws>/"
            f"computes/<name>; got {target!r}"
        )
    return m["sub"], m["rg"], m["ws"], m["compute"]


def _run_id_from_uri(model_uri: str) -> str:
    """runs:/<run_id>/<path> -> <run_id>."""
    if not model_uri.startswith("runs:/"):
        raise ValueError(f"expected runs:/<run_id>/model, got {model_uri!r}")
    run_id = model_uri[len("runs:/"):].split("/", 1)[0]
    if not run_id:
        raise ValueError(f"no run id in {model_uri!r}")
    return run_id


class AzureAdapter(CloudAdapter):
    def _blob_service(self, account: str) -> Any:
        # Imported here so selecting the adapter does not require the SDK to be installed.
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        return BlobServiceClient(
            account_url=f"https://{account}{_BLOB_HOST_SUFFIX}",
            credential=DefaultAzureCredential(process_timeout=_CLI_TIMEOUT_SECONDS),
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

    # --- Lab 2 -------------------------------------------------------------------------

    def _ml_client(self) -> tuple[Any, str]:
        # Imported here so Lab 1 commands never need the Azure ML SDK installed.
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        subscription, resource_group, workspace, compute = _parse_training_target(
            self.cfg.training_target
        )
        client = MLClient(DefaultAzureCredential(process_timeout=_CLI_TIMEOUT_SECONDS), subscription, resource_group, workspace)
        return client, compute

    def submit_training(self, image_uri: str, args: dict[str, Any]) -> str:
        """Run a module from the training image as an Azure ML command job.

        args (all optional except where noted):
          module      python module to run inside the image, e.g. "src.tune" (default "src.train")
          arguments   list of CLI arguments; the token {output} becomes the job's output folder
          data_uri    blob folder mounted read-only and copied to /app/data (default BLOB_URI/data)
          output_uri  blob folder the job writes to; reuse it to resume a study (default per job)
          env         extra environment variables (non-secret)
          experiment  experiment name shown in Azure ML (default "itcs355")
          lab         lab number for resource tags (default 2)
        """
        import shlex
        import uuid

        from azure.ai.ml import Input, Output, command
        from azure.ai.ml.constants import AssetTypes, InputOutputModes
        from azure.ai.ml.entities import Environment

        client, compute = self._ml_client()
        job_name = f"itcs355-{uuid.uuid4().hex[:12]}"
        blob_root = self.cfg.blob_uri.rstrip("/")
        data_uri = args.get("data_uri", f"{blob_root}/data")
        output_uri = args.get("output_uri", f"{blob_root}/jobs/{job_name}")

        module = args.get("module", "src.train")
        arguments = " ".join(
            shlex.quote(str(a)).replace("{output}", "${{outputs.output}}")
            for a in args.get("arguments", [])
        )
        cmd = (
            "mkdir -p /app/data && cp -r ${{inputs.data}}/. /app/data/ && "
            f"cd /app && python -m {module} {arguments}"
        ).strip()

        env = {
            "TRAINING_JOB_ID": job_name,
            "IMAGE_DIGEST": image_uri.split("@", 1)[1] if "@" in image_uri else image_uri,
            **{k: str(v) for k, v in args.get("env", {}).items()},
        }
        # Azure ML overwrites MLFLOW_TRACKING_URI in every job with its own azureml:// store and
        # adds MLFLOW_RUN_ID / MLFLOW_EXPERIMENT_* for it. Pass our server under another name and
        # restore it inside the job, so runs reach the self-hosted server (reference section 5).
        if "MLFLOW_TRACKING_URI" in env:
            env["ITCS355_MLFLOW_TRACKING_URI"] = env.pop("MLFLOW_TRACKING_URI")
            cmd = (
                'export MLFLOW_TRACKING_URI="$ITCS355_MLFLOW_TRACKING_URI" && '
                "unset MLFLOW_RUN_ID MLFLOW_EXPERIMENT_ID MLFLOW_EXPERIMENT_NAME MLFLOW_TRACKING_TOKEN && "
                + cmd
            )
        job = command(
            name=job_name,
            display_name=args.get("display_name", module),
            experiment_name=args.get("experiment", "itcs355"),
            command=cmd,
            environment=Environment(image=image_uri),
            compute=compute,
            inputs={"data": Input(type=AssetTypes.URI_FOLDER, path=_job_uri(data_uri),
                                  mode=InputOutputModes.RO_MOUNT)},
            outputs={"output": Output(type=AssetTypes.URI_FOLDER, path=_job_uri(output_uri),
                                      mode=InputOutputModes.RW_MOUNT)},
            environment_variables=env,
            tags=self.cfg.tags(int(args.get("lab", 2))),
        )
        submitted = client.jobs.create_or_update(job)
        return submitted.name

    def wait_training(self, job_id: str) -> dict[str, Any]:
        """Poll the job until it reaches a terminal state. Returns its final status."""
        import time

        client, _ = self._ml_client()
        terminal = {"Completed", "Failed", "Canceled", "NotResponding"}
        last = None
        while True:
            job = client.jobs.get(job_id)
            if job.status != last:
                print(f"  {job_id}: {job.status}", flush=True)
                last = job.status
            if job.status in terminal:
                return {
                    "job_id": job_id,
                    "status": job.status,
                    "studio_url": getattr(job, "studio_url", None),
                }
            time.sleep(_POLL_SECONDS)

    # --- Lab 3 ---------------------------------------------------------------
    def teardown(self, tags: dict[str, str], dry_run: bool = False) -> list[str]:
        """Delete every resource carrying ALL of these tags. Returns what was deleted.

        Scoped by tag, never by resource group: Labs 1, 2 and 3 share one group, and the
        group also holds the tracking server Lab 3 still needs. A missing "lab" tag is
        refused rather than widened, because the widening is what deletes Lab 1.
        """
        if "lab" not in tags or "course" not in tags:
            raise ValueError(f"teardown needs the course and lab tags; got {tags}")
        query = " && ".join(f"tags.{k} == '{v}'" for k, v in tags.items())
        ids = _run([
            "az", "resource", "list", "--query", f"[?{query}].id", "-o", "tsv",
        ]).split()
        if dry_run:
            return ids
        for resource_id in ids:
            _run(["az", "resource", "delete", "--ids", resource_id, "--verbose"])
        return ids

    def register_model(self, model_uri: str, name: str) -> str:
        """Register a tracked run's model and put its lineage ON THE VERSION.

        The registry is the MLflow registry at MLFLOW_TRACKING_URI (self-hosted, reference §5).
        model_uri is runs:/<run_id>/model. Every lineage field is copied from that run; if one
        is missing, nothing is registered, because a version without lineage cannot answer
        "which code, which data, which parameters".
        """
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
        client = MlflowClient()
        run_id = _run_id_from_uri(model_uri)
        run = client.get_run(run_id)
        tags, params, metrics = run.data.tags, run.data.params, run.data.metrics

        lineage = {
            "git_commit": tags.get("git_commit"),
            "data_version": tags.get("data_version"),
            "mlflow_run_id": run_id,
            "training_job_id": tags.get("training_job_id"),
            "image_digest": tags.get("image_digest"),
            "seed": params.get("seed"),
            "metric_val": metrics.get("val_roc_auc"),
            "metric_test": metrics.get("test_roc_auc"),
        }
        missing = sorted(k for k, v in lineage.items() if v in (None, "", "unknown"))
        if missing:
            raise ValueError(f"run {run_id} lacks lineage fields {missing}; not registering")

        version = mlflow.register_model(model_uri, name)
        for key, value in lineage.items():
            client.set_model_version_tag(name, version.version, key, str(value))
        return str(version.version)

    # deploy / invoke                   -> Lab 3 (managed online endpoint + deployment)
    # emit_metric                       -> Lab 4 (Azure Monitor custom metric)
    # generate                          -> Lab 5 (managed LLM endpoint; read the usage block for tokens)
    # teardown                          -> Lab 5 (resource graph query by tag)
