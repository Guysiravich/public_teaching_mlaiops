"""Lab 5 — compile pipeline/pipeline.yaml into your provider's pipeline service.

This module is Layer 3, so provider SDKs are allowed here and nowhere else. The DAG stays
in YAML; only the translation lives here. That separation is what you argue for or against
in Lab 5 Task 4.

Implement ONE compile function, for your provider.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

SPEC = Path(__file__).resolve().parents[1] / "pipeline" / "pipeline.yaml"


def load_spec(path: Path = SPEC) -> dict[str, Any]:
    """Read the neutral DAG and expand ${VAR} references from the environment."""
    raw = path.read_text()
    for key, value in os.environ.items():
        raw = raw.replace(f"${{{key}}}", value)
    return yaml.safe_load(raw)


def compile_aws(spec: dict[str, Any]):
    """SageMaker Pipelines.

    Each step becomes a ProcessingStep or TrainingStep; `condition` becomes a
    ConditionStep. Note that SageMaker's condition steps branch rather than abort, so
    `on_failure: abort` needs a FailStep on the else branch — an asymmetry worth
    mentioning in your Lab 5 write-up.
    """
    raise NotImplementedError("TODO Lab 5: build a sagemaker.workflow.pipeline.Pipeline")


def compile_azure(spec: dict[str, Any]):
    """Azure ML Pipelines: one command node per step, wired by the YAML's inputs and outputs.

    Returns (ml_client, pipeline_job); submit() and schedule() below send it. Mapping:
      * {inputs.x} / {outputs.x}  ->  ${{inputs.x}} / ${{outputs.x}}, uri_folder each;
        outputs land in the workspace's default datastore (a pipeline step rejects a raw
        wasbs:// output path — Lab 4).
      * depends_on  ->  data dependencies: Azure ML orders steps by what they consume, so
        validate writes a `contract` folder that train consumes, purely to order them.
      * on_failure: abort, and condition  ->  continue_on_step_failure = False: a step whose
        upstream failed never starts. The gate step exits non-zero on 'fail', so register
        and deploy_staging are skipped. azure.ai.ml.dsl.condition exists, but it needs a
        control-flow output and turns a rejected candidate into a successful run; here a
        rejection shows as a pipeline stopped at evaluate, which is what it is.
      * force_rerun  ->  True: otherwise identical steps are served from cache (the Lab 4
        drift job "ran" in zero seconds that way).
    Every step runs as the compute cluster's user-assigned identity (the training identity).
    """
    import re

    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.entities import Environment

    from cloudlayer.factory import get_adapter
    from src import config

    adapter = get_adapter(config.load(strict=False))
    client, compute = adapter._ml_client()
    env = Environment(image=spec["steps"][0]["image"])
    variables = {k: os.environ[k] for k in PIPELINE_ENV if os.environ.get(k)}
    # Which of the cluster's identities DefaultAzureCredential should use inside the job. Kept
    # out of the laptop's own environment, where it would hijack the developer's CLI login.
    if os.environ.get("AZURE_CLIENT_ID_TRAIN"):
        variables["AZURE_CLIENT_ID"] = os.environ["AZURE_CLIENT_ID_TRAIN"]
    # Azure ML replaces MLFLOW_TRACKING_URI in every job with its own store (Lab 2); pass ours
    # under another name, and scripts/pipeline_step.py restores it.
    if "MLFLOW_TRACKING_URI" in variables:
        variables["ITCS355_MLFLOW_TRACKING_URI"] = variables.pop("MLFLOW_TRACKING_URI")

    from azure.ai.ml import dsl

    builders: dict[str, Any] = {}
    for step in spec["steps"]:
        text = " ".join(step["command"])
        text = re.sub(r"\{(inputs|outputs)\.(\w+)\}", r"${{\1.\2}}", text)
        builders[step["name"]] = command(
            display_name=step["name"], command=f"cd /app && {text}",
            environment=env, compute=compute, environment_variables=variables,
            inputs={name: Input(type="uri_folder") for name in step.get("inputs", {})},
            outputs={name: Output(type="uri_folder") for name in step.get("outputs", [])},
        )

    @dsl.pipeline(name=spec["name"].replace("-", "_"), default_compute=compute)
    def dag():
        # The SDK only wires one node's output into another inside a pipeline function.
        nodes: dict[str, Any] = {}
        for step in spec["steps"]:
            wired = {name: nodes[src.split(".")[0]].outputs[src.split(".")[1]]
                     for name, src in step.get("inputs", {}).items()}
            nodes[step["name"]] = builders[step["name"]](**wired)
            nodes[step["name"]].name = step["name"]
        return {}

    job = dag()
    job.display_name = spec["name"]
    job.experiment_name = "itcs355-lab5-pipeline"
    job.tags = {k: str(v) for k, v in spec.get("tags", {}).items()}
    job.settings.default_compute = compute
    job.settings.continue_on_step_failure = False
    job.settings.force_rerun = True
    return client, job


def submit(client, job, wait: bool = True) -> dict[str, Any]:
    """Submit a compiled pipeline; with wait, return each step's final status."""
    import time

    run = client.jobs.create_or_update(job)
    print(f"submitted {run.name}: {run.studio_url}", flush=True)
    if not wait:
        return {"name": run.name}
    while client.jobs.get(run.name).status not in ("Completed", "Failed", "Canceled"):
        time.sleep(30)
    final = client.jobs.get(run.name)
    steps = {child.display_name: child.status
             for child in client.jobs.list(parent_job_name=run.name)}
    return {"name": run.name, "status": final.status, "steps": steps}


def schedule(client, job, name: str, cron: str | None) -> str:
    """Put the compiled pipeline on a cron schedule (UTC); cron=None deletes it."""
    from azure.ai.ml.entities import CronTrigger, JobSchedule

    if cron is None:
        try:
            client.schedules.begin_disable(name).result()
        except Exception:
            pass
        client.schedules.begin_delete(name).result()
        return name
    return client.schedules.begin_create_or_update(JobSchedule(
        name=name, trigger=CronTrigger(expression=cron, time_zone="UTC"), create_job=job,
        tags={"course": "itcs355", "lab": "5"})).result().name


# Configuration the steps read, passed into every node (none of it secret: the tracking
# server's password comes from Key Vault inside the step).
PIPELINE_ENV = ("CLOUD_PROVIDER", "PROJECT_ID", "REGION", "MLFLOW_TRACKING_URI",
                "MODEL_REGISTRY_NAME", "KEY_VAULT_NAME", "AZURE_CLIENT_ID", "TRAINING_TARGET",
                "STAGING_APP", "GIT_COMMIT", "IMAGE_DIGEST", "GATE_INCUMBENT")


def compile_gcp(spec: dict[str, Any]):
    """Vertex AI Pipelines.

    Kubeflow Pipelines underneath: steps become container components, conditions become
    dsl.If blocks. The compiled artifact is a JSON file you submit, so your build step
    produces a file rather than an object — different from the other two.
    """
    raise NotImplementedError("TODO Lab 5: compile a KFP pipeline and submit to Vertex")


COMPILERS = {"aws": compile_aws, "azure": compile_azure, "gcp": compile_gcp}


def compile_for(provider: str, spec: dict[str, Any] | None = None):
    spec = spec or load_spec()
    try:
        return COMPILERS[provider.lower()](spec)
    except KeyError:
        raise ValueError(f"No pipeline compiler for {provider!r}") from None
