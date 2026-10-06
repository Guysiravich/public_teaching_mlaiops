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

import os
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


def _query(cmd: list[str]) -> str:
    """Run a CLI query and return STDOUT only.

    `_run` deliberately returns stdout and stderr together, which is right for `docker push`
    (the digest is on stderr) and wrong for a lookup: the containerapp extension prints
    "WARNING: The behavior of this command has been altered by the following extension"
    on stderr, and an emptiness test on that sees a non-empty string and skips the create.
    """
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed ({result.returncode}):\n{result.stderr}")
    return result.stdout.strip()


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


# Lab 3. Azure ML managed online endpoints are unavailable on Azure for Students: they need
# ceil(1.2 x instances) x cores of quota, so one Standard_DS3_v2 asks for 8 vCPU against a cap
# of about 3 that cannot be raised (getting-started-azure.md). The container goes to Azure
# Container Apps instead: no Azure ML core quota, scales to zero, consumption pricing.
_CONTAINER_APP_ENV = "itcs355-env"
_SERVE_PORT = 8080


def _parse_model_ref(model_ref: str) -> tuple[str, str]:
    """models:/<name>/<version> -> (name, version)."""
    match = re.fullmatch(r"models:/([^/]+)/(\w+)", model_ref)
    if not match:
        raise ValueError(f"Expected models:/<name>/<version>, got {model_ref!r}")
    return match.group(1), match.group(2)


def _parse_instance(instance: str) -> tuple[str, str]:
    """"0.5/1.0Gi" -> ("0.5", "1.0Gi"). The provider's unit of size, as Azure ML took a VM SKU."""
    cpu, _, memory = instance.partition("/")
    if not cpu or not memory:
        raise ValueError(f"Expected <cpu>/<memory>, for example 0.5/1.0Gi, got {instance!r}")
    return cpu, memory


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
        client, job = self._command_job(image_uri, args)
        return client.jobs.create_or_update(job).name

    def _command_job(self, image_uri: str, args: dict[str, Any]) -> tuple[Any, Any]:
        """Build (not submit) the command job submit_training describes. schedule() wraps the
        same job in a trigger, so a scheduled run is exactly a submitted one."""
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
            # args["output"]=False drops the output folder: a pipeline step (which is how a
            # schedule runs this) rejects a wasbs:// output with "DataStore name is missing",
            # and the drift job writes nothing worth keeping there anyway.
            outputs=({"output": Output(type=AssetTypes.URI_FOLDER, path=_job_uri(output_uri),
                                       mode=InputOutputModes.RW_MOUNT)}
                     if args.get("output", True) else {}),
            environment_variables=env,
            tags=self.cfg.tags(int(args.get("lab", 2))),
        )
        return client, job

    def schedule(self, name: str, image_uri: str, args: dict[str, Any], cron: str) -> str:
        """An Azure ML job schedule around the same command job submit_training sends.

        Azure ML schedules are workspace objects, not ARM resources: they carry no resource
        tags, and `make teardown` cannot find them by tag. scripts/teardown.py deletes them
        by name — the handout's warning about schedules that outlive their endpoint.
        """
        from azure.ai.ml.entities import CronTrigger, JobSchedule

        client, job = self._command_job(image_uri, args)
        if not cron:
            try:
                client.schedules.begin_disable(name).result()
            except Exception as exc:  # already disabled, or never created
                print(f"  disable {name}: {exc.__class__.__name__}")
            try:
                client.schedules.begin_delete(name).result()
            except Exception as exc:
                if "NotFound" not in str(exc) and "not found" not in str(exc).lower():
                    raise
            return name
        # A schedule will not take a bare command job ("Unsupported job type 'CommandJob'"),
        # so the same command runs as the single step of a pipeline job.
        from azure.ai.ml.entities import PipelineJob

        _, compute = self._ml_client()
        pipeline = PipelineJob(jobs={"drift": job}, display_name=job.display_name,
                               experiment_name=job.experiment_name, tags=job.tags)
        pipeline.settings.default_compute = compute
        # Without this Azure ML reuses the previous run's result whenever the step's command,
        # image and inputs are unchanged — which, for a scheduled detector, is every run. The
        # 14:00 run of the Lab 4 exercise "completed" in zero seconds on cached output and
        # never looked at the drifted inputs.
        pipeline.settings.force_rerun = True
        trigger = CronTrigger(expression=cron, time_zone="UTC")
        created = client.schedules.begin_create_or_update(
            JobSchedule(name=name, trigger=trigger, create_job=pipeline,
                        tags=self.cfg.tags(int(args.get("lab", 4))))
        ).result()
        return created.name

    def scheduled(self, prefix: str = "itcs355") -> list[str]:
        """Names of the workspace's job schedules that start with `prefix`, enabled or not."""
        client, _ = self._ml_client()
        # "All" includes disabled schedules, which still exist and can be re-enabled.
        return [s.name for s in client.schedules.list(list_view_type="All")
                if s.name.startswith(prefix)]

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
    def deploy(self, model_ref: str, endpoint: str, instance: str) -> str:
        """Deploy the serving image as a Container Apps revision. Returns the endpoint URL.

        model_ref  models:/<name>/<version> — the version the container loads at startup
        endpoint   container app name
        instance   "<cpu>/<memory>", e.g. "0.5/1.0Gi"

        The image comes from SERVE_IMAGE (digest-pinned, as Lab 1 requires), the registry is
        read with a user-assigned identity rather than a registry password, and the tracking
        server's credentials are Container Apps secrets, never plain environment values.
        """
        import json
        import tempfile

        name, version = _parse_model_ref(model_ref)
        cpu, memory = _parse_instance(instance)
        image = os.environ["SERVE_IMAGE"]
        identity = os.environ["SERVE_IDENTITY_ID"]
        group = self.cfg.project_id
        registry = self.cfg.container_registry.split("/")[0]
        # Lab 4 deploys staging from CD with RESOURCE_LAB=4, so `make teardown LAB=4` finds it.
        tags = self.cfg.tags(int(os.environ.get("RESOURCE_LAB", "3")))
        environment = os.environ.get("SERVE_ENVIRONMENT", _CONTAINER_APP_ENV)

        existing = _query(["az", "containerapp", "env", "list", "-g", group,
                           "--query", f"[?name=='{environment}'].name", "-o", "tsv"])
        if not existing:
            # Lab 4 needs the container's stdout in Log Analytics: the drift job reads the
            # logged inputs back from there. Without a workspace, logs go nowhere.
            workspace = os.environ.get("SERVE_LOG_WORKSPACE", "")
            if workspace:
                customer_id = _query(["az", "monitor", "log-analytics", "workspace", "show",
                                      "--ids", workspace, "--query", "customerId", "-o", "tsv"])
                # get-shared-keys, unlike show, does not accept --ids.
                ws_group, ws_name = workspace.split("/")[4], workspace.rstrip("/").split("/")[-1]
                key = _query(["az", "monitor", "log-analytics", "workspace", "get-shared-keys",
                              "-g", ws_group, "-n", ws_name,
                              "--query", "primarySharedKey", "-o", "tsv"])
                logs = ["--logs-destination", "log-analytics",
                        "--logs-workspace-id", customer_id, "--logs-workspace-key", key]
            else:
                logs = ["--logs-destination", "none"]
            # Workload profiles, not the CLI's default express environment: express refuses
            # Multiple revision mode (Lab 3) and a revision suffix (Lab 4). Apps still run on
            # the Consumption profile, billed per second like before, inside the free grant.
            _run(["az", "containerapp", "env", "create", "-g", group, "-n", environment,
                  "-l", self.cfg.region, "--environment-mode", "WorkloadProfiles", *logs,
                  "--tags", *[f"{k}={v}" for k, v in tags.items()], "-o", "none"])

        env_id = _query(["az", "containerapp", "env", "show", "-g", group, "-n", environment,
                         "--query", "id", "-o", "tsv"])
        spec = {
            "location": self.cfg.region,
            "tags": tags,
            "identity": {"type": "UserAssigned", "userAssignedIdentities": {identity: {}}},
            "properties": {
                "environmentId": env_id,
                "configuration": {
                    # Multiple, so Lab 3 Task 4 can put a canary beside the current revision.
                    # Staging (Lab 4) runs Single: each green commit replaces the last.
                    "activeRevisionsMode": os.environ.get("SERVE_REVISIONS_MODE", "Multiple"),
                    "ingress": {"external": True, "targetPort": _SERVE_PORT, "transport": "auto"},
                    "registries": [{"server": registry, "identity": identity}],
                    "secrets": [
                        {"name": "mlflow-username", "value": os.environ["MLFLOW_TRACKING_USERNAME"]},
                        {"name": "mlflow-password", "value": os.environ["MLFLOW_TRACKING_PASSWORD"]},
                    ],
                },
                "template": {
                    # CD sets the commit, so every revision names the code it runs.
                    "revisionSuffix": os.environ.get("REVISION_SUFFIX", f"v{version}"),
                    "containers": [{
                        "name": "predict",
                        "image": image,
                        "resources": {"cpu": float(cpu), "memory": memory},
                        "env": [
                            {"name": "MODEL_REGISTRY_NAME", "value": name},
                            {"name": "MODEL_VERSION", "value": version},
                            {"name": "MLFLOW_TRACKING_URI", "value": self.cfg.mlflow_tracking_uri},
                            {"name": "MLFLOW_TRACKING_USERNAME", "secretRef": "mlflow-username"},
                            {"name": "MLFLOW_TRACKING_PASSWORD", "secretRef": "mlflow-password"},
                        ],
                        # Liveness and readiness are different questions: the process is up,
                        # against the model is loaded and can score. Routing traffic on the
                        # first one is the bug the lab is about.
                        "probes": [
                            {"type": "Liveness", "httpGet": {"path": "/health", "port": _SERVE_PORT},
                             "initialDelaySeconds": 5, "periodSeconds": 10},
                            {"type": "Readiness", "httpGet": {"path": "/ready", "port": _SERVE_PORT},
                             "initialDelaySeconds": 3, "periodSeconds": 5, "failureThreshold": 30},
                        ],
                    }],
                    "scale": {"minReplicas": int(os.environ.get("SERVE_MIN_REPLICAS", "1")),
                              "maxReplicas": int(os.environ.get("SERVE_MAX_REPLICAS", "1"))},
                },
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(spec, handle)
            spec_path = handle.name

        # The spec is already an ARM body, so it goes to ARM directly. `az containerapp
        # create --yaml` with the same content fails with a 400 ("could not be converted to
        # System.Boolean") — met in Lab 3 and again on the first Lab 4 CD run. A PUT creates
        # the app or replaces its configuration, which makes a new revision either way.
        subscription = env_id.split("/")[2]
        url = (f"https://management.azure.com/subscriptions/{subscription}/resourceGroups/{group}"
               f"/providers/Microsoft.App/containerApps/{endpoint}?api-version=2024-03-01")
        _run(["az", "rest", "--method", "put", "--url", url, "--body", f"@{spec_path}", "-o", "none"])

        import time
        deadline = time.time() + 900
        while True:  # the PUT returns at once; provisioning finishes in the background
            state = _query(["az", "containerapp", "show", "-g", group, "-n", endpoint,
                            "--query", "properties.provisioningState", "-o", "tsv"])
            if state == "Succeeded":
                break
            if state == "Failed" or time.time() > deadline:
                raise RuntimeError(f"container app {endpoint} provisioning ended as {state}")
            time.sleep(10)

        fqdn = _query(["az", "containerapp", "show", "-g", group, "-n", endpoint,
                       "--query", "properties.configuration.ingress.fqdn", "-o", "tsv"])
        return f"https://{fqdn}"

    def invoke(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call the deployed service once. `endpoint` is the URL deploy() returned."""
        import json
        import urllib.request

        route = "/predict/batch" if "rows" in payload else "/predict"
        request = urllib.request.Request(
            endpoint.rstrip("/") + route,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read())
            body["model_version_header"] = response.headers.get("x-model-version", "")
            return body

    # --- Lab 4 ---------------------------------------------------------------
    def _token(self, scope: str) -> str:
        from azure.identity import DefaultAzureCredential

        # CLI login on a laptop, the federated identity in GitHub Actions, the compute
        # cluster's managed identity inside an Azure ML job. No key anywhere.
        credential = DefaultAzureCredential(process_timeout=_CLI_TIMEOUT_SECONDS)
        return credential.get_token(scope).token

    def emit_metric(self, name: str, value: float, unit: str = "None") -> None:
        """One data point to Azure Monitor custom metrics, on the resource METRICS_RESOURCE_ID
        (the staging container app), namespace itcs355. A metric alert on that resource
        turns it into an email (monitoring/README in the Lab 4 section)."""
        import datetime
        import json
        import urllib.request

        resource = os.environ["METRICS_RESOURCE_ID"]
        point = {
            "time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data": {"baseData": {
                "metric": name.replace(".", "_"),
                "namespace": "itcs355",
                "series": [{"min": value, "max": value, "sum": value, "count": 1}],
            }},
        }
        request = urllib.request.Request(
            f"https://{self.cfg.region}.monitoring.azure.com{resource}/metrics",
            data=json.dumps(point).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._token('https://monitoring.azure.com/.default')}"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status >= 300:
                raise RuntimeError(f"emit_metric {name}: HTTP {response.status}")

    def recent_inputs(self, limit: int) -> list[dict[str, float]]:
        """The last `limit` inputs the serving container logged, read from the environment's
        Log Analytics workspace (LOG_WORKSPACE_ID, the workspace GUID). service/app.py writes
        one {"event":"input"} line per scored row; the platform ships stdout there."""
        import json
        import urllib.request

        app = os.environ.get("STAGING_APP", "itcs355-staging")
        query = (
            "ContainerAppConsoleLogs_CL"
            f" | where ContainerAppName_s == '{app}'"
            " | where Log_s has '\"event\":\"input\"'"
            f" | top {int(limit)} by TimeGenerated desc"
            " | project Log_s"
        )
        request = urllib.request.Request(
            f"https://api.loganalytics.io/v1/workspaces/{os.environ['LOG_WORKSPACE_ID']}/query",
            data=json.dumps({"query": query, "timespan": "PT24H"}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._token('https://api.loganalytics.io/.default')}"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            tables = json.loads(response.read())["tables"]
        rows = []
        for (line,) in tables[0]["rows"] if tables else []:
            try:
                rows.append(json.loads(line)["msg"]["features"])
            except (ValueError, KeyError, TypeError):
                continue  # a line the logger split or truncated is skipped, not guessed at
        return rows

    # --- Lab 5 (teardown is used by Lab 3's handout too) ------------------------
    def teardown(self, tags: dict[str, str], dry_run: bool = False) -> list[str]:
        """Delete every resource carrying ALL of these tags. Returns what was deleted.

        Scoped by tag, never by resource group: Labs 1, 2 and 3 share one group, and the
        group also holds the tracking server Lab 3 still needs. A missing "lab" tag is
        refused rather than widened, because the widening is what deletes Lab 1.
        """
        if "lab" not in tags or "course" not in tags:
            raise ValueError(f"teardown needs the course and lab tags; got {tags}")
        query = " && ".join(f"tags.{k} == '{v}'" for k, v in tags.items())
        ids = _query([
            "az", "resource", "list", "--query", f"[?{query}].id", "-o", "tsv",
        ]).split()
        # Dependents first. `az resource list` returns no particular order, and an environment
        # that still holds an app refuses deletion (ManagedEnvironmentHasContainerApps — the
        # first Lab 4 teardown); an action group is still referenced by its alert rule.
        first = ("/containerapps/", "/metricalerts/", "/onlineendpoints/")
        last = ("/managedenvironments/", "/actiongroups/", "/userassignedidentities/")
        ids.sort(key=lambda i: 0 if any(t in i.lower() for t in first)
                 else 2 if any(t in i.lower() for t in last) else 1)
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

    # deploy / invoke                   -> Lab 3 (Container Apps revision + HTTP call)
    # emit_metric                       -> Lab 4 (Azure Monitor custom metric)
    # generate                          -> Lab 5 (managed LLM endpoint; read the usage block for tokens)
    # teardown                          -> Lab 5 (resource graph query by tag)
