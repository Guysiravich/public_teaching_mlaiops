# Lab 5 Part A — pipeline, identities, retraining (Tasks 1–3)

## Task 1 — the managed pipeline

[`pipeline/pipeline.yaml`](../pipeline/pipeline.yaml) is the DAG. Every step runs
`scripts/pipeline_step.py <step>` from the training image, pushed by digest.
`cloudlayer/pipelines.py` (`compile_azure`) turns the file into an Azure ML pipeline on the
cluster `l5-f2s`.

```
ingest → validate (tests/test_data.py) → train → evaluate (gate) → register → deploy_staging
```

| Run (16 Oct, UTC) | Steps | What it shows |
|---|---|---|
| `boring_curtain_3t81t1rx2h` 19:05 | ingest ✓ · **validate ✗** | The image had no `tests/`, so the contract step could not run its tests. **Nothing was trained.** The abort works, and it was a bug of mine that proved it. |
| `neat_truck_ysfvyx59ln` 19:16 | ingest ✓ validate ✓ train ✓ · **evaluate ✗** | **The gate rejecting a real candidate.** It scored test ROC-AUC 0.8482. The registry's `staging` version 1 scores 0.8543, so delta −0.0061 against a required +0.003. Register and deploy never started. |
| `careful_basket_q2vqkdklcm` 19:37 | all six ✓ | **The pass branch, forced** with `--incumbent 0.80` and labelled as a demonstration. It registered version 3 with full lineage, moved `staging` → 3 and rolled out revision `itcs355-staging--v3-fd6b6`. The endpoint answered `x-model-version: 3`. |
| (rollback, 20:06) | — | Version 3 is worse than version 1, so I moved `staging` back to 1. The rollout took 19 s to provision revision `--v1-4d009`. **The first request after it still got version 3.** The old revision serves until the new one is ready, so a rollback is done when the response header says so, not when the deployment does. Version 3 is tagged with a note and never kept. |
| `itcs355-pipeline-…CU19` 20:50 | ingest ✓ validate ✓ train ✓ · evaluate ✗ | Started by the **schedule** (`runSource: Schedule`). Same outcome as the manual run (Task 3). |

**Why no candidate passes on its own.** The pipeline retrains on the same data. When it uses
version 1's own configuration (depth 4), it reproduces version 1 exactly: test ROC-AUC
`0.8543112516622451` both times, same data fingerprint `422cccb9136e8140`. That gives a delta
of 0.0000, and the gate refuses. That is the correct behaviour. Retraining unchanged data is
not an improvement, and a gate that registered it would fill the registry with copies. The
pass branch is reachable only when the data or the code changes. I had to force it once to
show it. I did not leave it forced.

**The gate's numbers** (`scripts/evaluation_gate.py`):

- **`MIN_IMPROVEMENT = 0.003`** is about two standard deviations of Lab 2's seed variance
  (0.8408 ± 0.0013). A smaller "improvement" is the seed, not the model.
- **`MIN_ABSOLUTE = 0.80`** sits below the model that Lab 3's canary showed was harmful in
  production (0.826).
- **The incumbent** is read from the registry: the `staging` alias, through that version's
  lineage tag `metric_test`.

`--incumbent 0.99` exits 1, as the course's grader checks.

**How the YAML's `condition:` and `on_failure: abort` map onto Azure.** I set
`continue_on_step_failure = False`, so a step whose upstream failed never starts. The gate
step exits non-zero on "fail". The register step also reads `gate_decision` and refuses on a
fail. `azure.ai.ml.dsl.condition` exists, but it would show a rejected candidate as a
*successful* run. Here a rejection shows as a pipeline stopped at evaluate, which is what it
is. That is an asymmetry with SageMaker, which needs a FailStep for the same effect.

**Other things the platform taught:**

- **Azure ML compute rejects `Standard_B2s`**, the size the handout's student note
  recommends: B-series is not a supported VM size for a compute cluster. I used
  `Standard_F2s_v2` (0.108 USD/h) instead. *The instance I would choose is not one I can
  run.*
- **The pipeline SDK only wires one step's output into another inside a `@dsl.pipeline`
  function.**
- **`force_rerun` stays on**, or identical steps are served from cache (Lab 4).
- **The subscription allows one Container Apps environment in total, in any region.** The
  staging app shares `bike-env` with the capstone project.

## Task 2 — least privilege, and what broke

| Identity | Used by | Can do | Why each permission |
|---|---|---|---|
| `itcs355-train-l5` | the pipeline cluster `l5-f2s`; every step runs as it | **AcrPull** on the registry | pull the training image (proved below) |
| | | **Key Vault Secrets User** on `mlflow-username` and `mlflow-password` only, not the vault | read the tracking server's credentials at run time. Lab 2 passed them as job environment variables; now nothing secret is in the job definition |
| | | **Contributor** on the container app `itcs355-staging` only | the deploy_staging step rolls the staging app to the registered version, and nothing else in the group |
| `itcs355-serve-l5` | the staging container app | **AcrPull** on the registry | pull the serving image; the model comes from the registry through Container Apps secrets |
| `itcs355-ci-l5` | GitHub Actions via an OIDC federated credential that trusts only `repo:Guysiravich@52388254/public_teaching_mlaiops@1367593417:environment:staging` | **AcrPush** on the registry | push the serving image |
| | | **AzureML Data Scientist** on the workspace | trigger the pipeline |
| | | **Contributor** on `itcs355-staging` only | deploy the app |
| | | **Managed Identity Operator** on `itcs355-serve-l5` only | attach the serving identity to the app when deploying |
| | | **Reader** on the resource group | `deploy()` looks the environment up by name |

Lab 4's CD identity held Contributor on the whole resource group. The three identities above
replace it.

**Experiment 1 — removing a permission I believed unnecessary.** I removed **Storage Blob Data
Contributor** on the container `itcs355` from the training identity (20:08). In Lab 5 the data
comes from the ingest step, not from that container. The outputs between steps go to the
workspace's own container, `azureml-blobstore-…`, on which the identity never held a role.
Run `placid_goat_cq5s8t4dyp`: ingest, validate and train succeeded, and evaluate stopped at the
gate for the gate's own reason (delta −0.0061). **Nothing broke.** The permission stays removed.
The training identity is narrower than the one Lab 2 needed.

**Experiment 2 — removing one I believed necessary, to see the failure.** I removed
**AcrPull** (20:20). Run `khaki_cord_mr2mrcmpqy` failed at its **first step after 50 seconds**:
*"Failed to pull Docker image … the compute could not authenticate with the Docker registry …
ensure … a Managed Identity with AcrPull access to the ACR is assigned to the compute."* This
answers a question the table could not. Azure ML pulls the image as the **cluster's**
identity, not the workspace's or the submitter's. I restored the permission at 20:36.

**The scheduled run** (Task 3) succeeded with the same identity. That is the handout's "fails
only when scheduled" trap avoided by design: every step runs as the cluster's identity
whoever, or whatever, submits it.

## Task 3 — scheduled retraining, and what should trigger it

The schedule `itcs355-pipeline` runs at **02:00 UTC on the 1st of each month**
(`pipeline.yaml`). I first set a one-off cron to watch a scheduled run happen (20:50, above),
then set the monthly one.

| Strategy | When it is right for this system | How it fails here |
|---|---|---|
| **Fixed schedule** (monthly) | The label is "fails within 7 days" and the fleet changes with seasons, duty cycles and new machines, on a scale of weeks. Monthly matches Lab 2's decision and Lab 4's 35-day freshness SLO. | It retrains when nothing changed, which costs a node per run; the gate then rejects the result, as it did on every run tonight. It also misses a fast change for up to a month. |
| **Data-volume threshold** (e.g. 1,000 new labelled readings) | When labels arrive unevenly — a quiet month should not trigger a retrain on the same data. | Labels lag by 7 days, so "new data" is always a week old. And volume says nothing about whether the new data is different or clean: a backfilled duplicate export would count. |
| **Drift-triggered** (the Lab 4 detector) | When a real, persistent shift in the inputs is the reason accuracy would fall: new machines, a new operating regime. | **This is the dangerous one.** In Lab 4 the drift was a sensor reading 6 °C high. A drift trigger would have retrained on that corrupted data within the hour and registered a model that treats the offset as normal. The data contract tests would not have stopped it: 85 °C is a plausible temperature. The model that knew the truth would be the one replaced. |

**Chosen: the monthly schedule, gated twice.** Every run passes the data contracts and must
beat the registry's `staging` version by more than seed noise. Drift alerts page a person,
the Lab 4 post-mortem path, and never trigger a retrain directly.

**The worst case** is a slow upstream corruption that stays inside the plausible ranges for
a month, such as a gradual sensor offset. The monthly run would train on it, and the gate
would compare the candidate on a test split drawn from the same corrupted data, where it
could look better. The defences are the drift alert, which fires on the shift before the
month ends; the post-mortem's rule, which checks for broken sensors before retraining; and
the registry, which keeps version 1 so a rollback is one alias move and a 19-second rollout.
