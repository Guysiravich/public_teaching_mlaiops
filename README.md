# ITCS355 Lab 1 — Reproducible Training

> **Course materials live in [`course/`](course/README.md)** — syllabus, slides, the faculty
> specification, all five lab handouts, and the project brief. Every document is Markdown and
> renders on GitHub, diagrams included. New to the repo? Start with the
> [portability reference](course/reference/cloud-portability-reference.md).
> Keep this block when you edit the rest of this file; it is not part of the Lab 1 deliverable.

Predicting machine failure within 7 days from sensor readings. The model is not the point;
whether a stranger can reproduce it is.

---

## Reproduce

```bash
make reproduce
```

expected test_roc_auc: 0.8482 ± 0.001

Runtime: about 80 seconds on a 12-core machine when the image is rebuilt from cached layers, and
about 35 seconds once the image already exists. The very first build on a machine with nothing
cached also downloads the base image and every dependency, which adds a few minutes.

**What the grader's machine needs:** Docker (with `buildx`), `make`, and `git`. Nothing else — no
Python packages, no cloud account, no credentials, no `cloud.env`. `make reproduce` builds the
image for `linux/amd64`, generates the dataset *inside* that image, then trains inside it.

To check the result against the line above, run `make verify`. It uses only the Python standard
library, so any `python` on the PATH will do.

### How the tolerance was chosen

`make reproduce` fixes the seed (`20260101`), so the tolerance has to cover how much *the same run*
can drift between machines — not how much the metric moves when the seed changes. A different seed
reshuffles which machines land in the test split, which answers a different question.
The course's repository guide says the same (`docs/REPO-GUIDE.md`, *Two things that will catch
you out*): at this seed the reference run returns 0.8482 on any machine, while sweeping seeds
moves it across 0.82–0.87.

The same configuration and seed, run four times in two different environments:

| Run | Environment | test_roc_auc |
|---|---|---|
| host 1 | Python 3.14, packages from `make setup` | 0.8482378549 |
| host 2 | Python 3.14, packages from `make setup` | 0.8482378549 |
| image 1 | Python 3.11, the training image | 0.8482378549 |
| image 2 | Python 3.11, the training image | 0.8482378549 |

Measured spread: **0**, to ten decimal places, across two interpreters and two resolved dependency
sets. The ± 0.001 leaves room for floating-point differences on another CPU or architecture, and it
is still much smaller than the effect of any hyperparameter change in the runs below.

---

## The problem

240 machines, 25 readings each, 6 sensor features, binary target `failed_within_7d` with a
positive rate near 12%.

Machines have persistent characteristics — a hot-running machine reads hot in every row. So the
train/validation/test split is **grouped by `machine_id`**: every reading from one machine lands
in exactly one partition. Splitting row-wise instead lets the model memorise the machine and
reports a validation score that will never survive production. `tests/test_data.py` asserts this
property holds, and Lab 4 turns it into a CI gate.

## The data

Synthetic, generated deterministically from the seed by `scripts/make_dataset.py`:
`data/raw/sensors.csv`, 6,000 rows.

| Identifier | Value |
|---|---|
| Data fingerprint (sha256 prefix, logged with every run) | `422cccb9136e8140` |
| DVC hash of `data/raw` (from `data/raw.dvc`) | `1c886b512c8a5c9bf723da1cd119fc80.dir` |

DVC tracks `data/raw`; Git tracks only `data/raw.dvc`. See *Notes for the grader* for the remote.

---

## Tracked runs

Five runs on the same data and seed, varying `max_depth` to find where the forest stops
generalising. Everything else is the default (`n_estimators=200`, `min_samples_leaf=5`,
seed `20260101`).

| Run | max_depth | val_roc_auc | test_roc_auc |
|---|---|---|---|
| depth-3 | 3 | 0.8386 | 0.8524 |
| depth-5 | 5 | 0.8423 | 0.8530 |
| depth-8 | 8 | 0.8364 | 0.8482 |
| depth-12 | 12 | 0.8352 | 0.8424 |
| depth-16 | 16 | 0.8362 | 0.8415 |

Shallow trees do best. Validation ROC AUC peaks at depth 5, and test ROC AUC falls steadily past
depth 8 as the trees grow deep enough to fit individual machines rather than the failure pattern.

`make reproduce` keeps the repository default, depth 8. Switching it to depth 5 *because* depth 5
scored best on the test set would be selecting on the test set, and the reported test number would
stop being an honest estimate.

Each run is recorded in MLflow at `MLFLOW_TRACKING_URI` (a local SQLite store, `mlflow.db`, which is
not committed) and carries: every hyperparameter including the seed, `val_*` and `test_*` metrics
separately, the Git commit (`git_commit`), the data fingerprint (`data_fingerprint`), the DVC hash
(`data_dvc_md5`), and the trained model as an artifact.

---

## What is pinned, and where

| What | How | Where |
|---|---|---|
| Dependencies | `pip-compile --generate-hashes`: 95 packages, every one hash-pinned, compiled inside the pinned base image so the resolution matches the interpreter that runs it | `requirements.txt` (from `requirements.in`) |
| Dependency install | `pip install --require-hashes`: a substituted package fails the build | `Dockerfile` |
| Base image | `python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534` | `Dockerfile`, both stages |
| Seeds | Python `random`, NumPy, `PYTHONHASHSEED`, and the forest's `random_state`; the seed is logged as a parameter | `src/seeds.py`, `src/train.py` |
| Split | grouped by `machine_id`, built in the pipeline, deterministic given the seed | `src/data.py` |
| Leakage | fails if any machine appears in more than one split | `tests/test_data.py::test_no_machine_leaks_across_splits` |
| Data version | DVC | `data/raw.dvc` |

---

## Layout

```
src/          Layer 1 — provider-neutral. No SDKs, no bucket names, no absolute paths.
cloudlayer/   Layer 3 — the only place a provider SDK may be imported.
scripts/      Dataset generation, cloud check, portability audit, metric verification.
tests/        Data contract tests and split property tests.
```

`src/config.py` is the single point of environment knowledge. Everything else reads from it.
`make portability-audit` enforces the rule; it fails the build if a provider string appears in
`src/` or `tests/`.

---

## Setup (for development — not needed to reproduce)

```bash
cp cloud.env.example cloud.env      # fill in, never commit
make setup
make cloud-check                    # eight slots, all PASS
make data                           # generate the dataset (inside the training image)
make test                           # all tests passing
```

Tools used outside the training image, installed into the same virtual environment:

```bash
pip install "dvc[azure]==3.67.1"                                  # Task 4 (dvc-azure 3.1.1)
pip install azure-storage-blob==12.30.1 azure-identity==1.25.3   # only for `make image-push`
```

The Azure SDK is kept out of `requirements.txt` on purpose: the training image never talks to
Azure, and the course guide installs it separately. It is imported lazily inside
`cloudlayer/azure.py`, so nothing else needs it.

---

## Container image

`make image-push` builds the image, pushes it through `cloudlayer/azure.py`, and prints the
digest-pinned reference.

Pushed, pinned by digest:

```
itcs3556688067.azurecr.io/itcs355/itcs355-lab1@sha256:652ba7c6b54bbe2a9aa2a265c7eccbac9db2fc022c0ff9a0c14820c4e726794c
```

Built for `linux/amd64` from commit `70c551a`. The registry also carries the tag `70c551a`, but
the digest is the reference to use: a tag can be moved, a digest cannot. `make reproduce` builds
the image locally and does not pull this one.

---

## Reproducibility trade-off

**Drop the hashes first.** Every dependency stays pinned to an exact version with `==`, so what I
lose is protection against different bytes arriving under the same version — a replaced file on a
mirror or a compromised index — which pip would then install silently instead of failing the build.
PyPI forbids re-uploading a file under the same name, so this is the least likely failure. The
digest pin guards the likeliest one: `python:3.11-slim` moves regularly with no commit from me. The
seed is what makes the claimed metric checkable at all.

---

## Notes for the grader

**The one command needs only Docker.** As provided, `make reproduce` first ran
`scripts/make_dataset.py` on the host, which needs numpy and pandas there. `make data` now runs the
same script inside the training image, and the regenerated file is byte-identical (same sha256,
fingerprint `422cccb9136e8140`). This was tested from a fresh clone on a host with no Python
packages installed and no `cloud.env`.

**Two permission fixes to the provided build.** The image runs as the non-root user `runner`
(uid 10001), which could not write into the bind-mounted `reports/` (owned by whoever cloned the
repository) or create MLflow's `mlruns/` inside the root-owned `/app`, so the provided
`make reproduce` failed on Linux. `make reproduce` now opens `reports/` before mounting it, and the
`Dockerfile` gives `runner` ownership of `/app`. The container still runs as non-root.

**Two DVC remotes, one location.** Both point at `${BLOB_URI}/dvc`:

| Remote | URL | Used for |
|---|---|---|
| `public` (default) | `https://itcs3556688067.blob.core.windows.net/itcs355/itcs355/dvc` | `dvc pull` — anonymous read, no Azure account |
| `storage` | `azure://itcs355/itcs355/dvc` with `account_name = itcs3556688067` | `dvc push -r storage` — writes, with the owner's Azure login |

DVC reads an `https://` URL as a plain HTTP remote that sends no Azure credentials: fine for reading
objects the account exposes, unable to write. DVC's Azure form carries credentials but does not fall
back to anonymous access, so it is kept for pushing only.

**`dvc pull` needs no Azure account.** From a fresh clone:

```bash
pip install "dvc[azure]==3.67.1"
dvc pull
```

Tested from a fresh clone with no Azure login and no `cloud.env`: 2 files fetched in about
7 seconds, and `data/raw/sensors.csv` is byte-identical to the tracked version. An anonymous visitor
can read an object only by its exact path; listing the container returns `404`. `make reproduce`
does not need `dvc pull` at all, because `make data` regenerates the same file from the seed.

**The Git commit inside the container reads `unknown`.** `.dockerignore` excludes `.git` from the
image, as provided. The tracked runs above were made on the host and carry the real commit SHA.

**Cloud:** Azure, region `eastasia`. Azure for Students subscriptions cannot deploy to
`southeastasia`.

---

## Checklist before you submit

- [x] `make reproduce` works from a fresh clone — tested in a clean clone on a host with no Python packages and no `cloud.env`; not yet on a second physical machine
- [x] `make verify` passes against the claim line
- [x] `make test` — all tests pass
- [x] `make portability-audit` — clean
- [x] Image builds for `linux/amd64` and is pushed, digest-pinned (see *Container image*)
- [x] `dvc push` completed; a grader can `dvc pull` without an Azure account
- [x] Five or more tracked runs with params, metrics, data fingerprint, and commit SHA
- [x] Every instruction block from the template is gone (the course-materials block at the top stays)
- [x] `git log -p | grep -i -E "secret|password|AKIA|BEGIN PRIVATE"` — the only matches are words in the course's own documentation and comments; no credential appears in history

---

# Lab 2 — Tracking and registry

Everything below is Lab 2. The Lab 1 sections above are unchanged.

## What runs where

| Piece | Where | Why |
|---|---|---|
| Training job (Task 1), the 12-trial study (Task 2), the seed-variance runs (Task 3) | Azure ML command jobs on the cluster `ded-ds2`: `Standard_DS2_v2`, dedicated, 0–1 nodes, scales to zero after 120 s idle | the handout's managed compute |
| Tracking server and model registry | self-hosted MLflow 3.16 on a `Standard_B2ats_v2` VM, PostgreSQL backend, HTTPS and a password in front | reference §5, first recommendation |
| Model artifacts | `${BLOB_URI}/mlruns`, written by the server with its managed identity | reference §5 |

```bash
make train-remote                 # Task 1: one managed training job
make tune-remote                  # Task 2: the 12-trial study as one managed, resumable job
make seeds-remote                 # Task 3: seed variance of the study's top 3 trials
make compare                      # Task 3: reports/lab2-comparison.md
make register RUN_ID=<run id>     # Task 4: register with lineage on the version
make promote VERSION=<n>          # Task 4: promote to the alias `staging`
make reload-check VERSION=<n>     # Task 5
```

The submitting machine needs `TRAINING_TARGET`, `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and
`MLFLOW_TRACKING_PASSWORD` in `cloud.env`; `cloud.env.example` documents all four.

## Tracking server: self-hosted, and the trade-off

- **Chosen: self-hosted.** Portable across providers, and it keeps `mlflow==3.16.0` as pinned.
- **Not chosen: Azure ML's managed MLflow.** It needs the `azureml-mlflow` plugin, whose latest
  release (1.62.0.post6) requires `mlflow-skinny<=3.15.0`, below the course pin. Downgrading would
  also leave the Lab 1 tracking database unreadable: its schema is newer than 3.15 understands.
- **Cost of the choice.** A VM billed by the hour, plus a static IP and a disk billed all the time
  (see *Compute and cost*). The VM is deallocated when not in use and shuts down daily at 23:00.

Three differences from the §5 example, each forced by hosting it for a managed job:

1. **Not `localhost`.** A job running in Azure cannot reach a laptop, so the server is on a VM.
2. **PostgreSQL instead of SQLite.** `cloud.env.example`: "use sqlite locally and a database
   backend on a server".
3. **Artifacts served through the server** (`--serve-artifacts --artifacts-destination`) rather
   than `--default-artifact-root`. Otherwise every client, the grader's `reload_check.py` included,
   would need its own Blob Storage credentials. Artifacts still land in `${BLOB_URI}/mlruns`.

Configuration: `infra/tracking-server/` (`provision_azure.sh`, `docker-compose.yml`, `Caddyfile`,
`make_secrets.sh`). Secrets live in files that are gitignored and never committed.

**How a job reaches the password-protected server.** The job receives `MLFLOW_TRACKING_USERNAME` and
`MLFLOW_TRACKING_PASSWORD` as job environment variables at submission time, from the submitter's
`cloud.env`. They are never in Git or in an image layer. Limitation: anyone with access to the Azure
ML workspace can read them in the job's definition. Lab 4 moves them to Key Vault.

## Search space

| Hyperparameter | Values | Question it asks |
|---|---|---|
| `max_depth` | 4, 8, 12 | how complex may one tree be? |
| `class_weight` | None, balanced | how should the rare class (~12% failures) be weighted? |
| `max_features` | sqrt, 1.0 | how different should the trees be from one another? |

3 × 2 × 2 = 12 trials. `min_samples_leaf` (5) is held fixed because it asks the same question as
`max_depth`. `n_estimators` (200) is held fixed because it mostly moves cost, not behaviour.

## First submission: the permissions that failed (Drill 2)

The job's submit-time identity (my `az login`) could do everything. Its run-time identity, the
compute cluster, had no identity and no permissions at all. Two failures, one after the other:

1. **Pulling the image.** Job `itcs355-715240070e82`:
   `Failed to pull Docker image itcs3556688067.azurecr.io/itcs355/itcs355-lab1@sha256:4f7f0eff… This
   error may occur because the compute could not authenticate with the Docker registry to pull the
   image. If using ACR please ensure the ACR has Admin user enabled or a Managed Identity with
   AcrPull access to the ACR is assigned to the compute.`
   **Fix:** a system-assigned managed identity on `ded-ds2`, granted **`AcrPull` on the registry
   only**. The ACR admin user was not enabled: it is a shared password with push rights.
2. **Reading the data.** Job `itcs355-4559eb762148`, with the image now pulled: the data mount
   failed with `ScriptExecution.StreamAccess.Authentication … This request is not authorized to
   perform this operation using this permission` (HTTP 403 on `…/itcs355/itcs355/data`). The node
   had obtained a storage token for its managed identity, and the identity had no data-plane role.
   **Fix:** **`Storage Blob Data Contributor` on the container `itcs355` only**, not the storage
   account. Contributor rather than Reader, because the job also writes its outputs (the study's
   checkpoint).

Granted one at a time, reading what broke next, as Session 5 describes. Final roles of the
cluster identity: `AcrPull` (registry) and `Storage Blob Data Contributor` (container `itcs355`).

Other failures on the way, not permissions:

- `NotSupportedAssetOutputUri`: Azure ML treats `https://` storage URIs as public and read-only,
  so job inputs and outputs are passed in `wasbs://` form (`cloudlayer/azure.py`, `_job_uri`).
- `UnsupportedModelRegistryStoreURIException … 'azureml://…'`: Azure ML replaces
  `MLFLOW_TRACKING_URI` in every job with its own store. The adapter passes the server as
  `ITCS355_MLFLOW_TRACKING_URI` and restores it inside the job.
- `AzureCliCredential: Failed to invoke the Azure CLI`: from WSL the CLI can take longer than the
  SDK's 10-second default to return a token; the adapter allows 60 s.

## Study, interruption and resume (Task 2)

The study ran as two managed jobs writing one checkpoint, `${BLOB_URI}/jobs/lab2-tune/tune_checkpoint.json`.

| Job | What happened | Trials finished |
|---|---|---|
| `itcs355-7b68899ff665` | cancelled on purpose once 3 trials had finished; the 4th was cut off mid-trial | 4 (the cut-off run is marked `KILLED`) |
| `itcs355-a78b2fedff1c` | resubmitted with the same output folder; skipped the 4 trials in the checkpoint | 8 |

12 finished trials in total, none repeated. **Dedicated, not low-priority, compute.** This Azure for
Students subscription has an Azure ML low-priority vCPU quota of 0
(`ClusterMinNodesExceedCoreQuota … total vCPU quota of 0`), so the study runs on dedicated nodes,
which are not pre-empted. The interruption is therefore simulated by cancelling the job.
`compare_runs.py` lists `FINISHED` runs only, so the cut-off trial is not counted twice.

## Seed variance (Task 3)

`make seeds-remote` (job `itcs355-7ef754377692`) refits the study's top 3 configurations with 5 model
seeds each. Only the model's `random_state` changes; the train/val/test split keeps the study's
seed, so every refit is scored on the same validation rows.

| Configuration | Study val ROC-AUC | 5-seed mean | std | range |
|---|---|---|---|---|
| depth 4, class_weight None, sqrt | 0.8405 | 0.8408 | 0.0013 | 0.8392–0.8425 |
| depth 4, class_weight balanced, sqrt | 0.8398 | 0.8411 | 0.0017 | 0.8398–0.8439 |
| depth 8, class_weight balanced, sqrt | 0.8394 | 0.8387 | 0.0020 | 0.8360–0.8413 |

The top three trials are 0.0011 apart, less than one seed standard deviation. The study's ranking
among them is noise. `max_features = sqrt` against `1.0` (about 0.01 apart) is not.

## Compute and cost

Prices: Azure Retail Prices API, eastasia, Linux, 2026-09-18, 1 USD = 33.234 THB (`src/costs.py`).

| Resource | Rate | Billed when |
|---|---|---|
| `Standard_DS2_v2` node | 7.11 THB/h | from node allocation until it scales to zero, 120 s after the last job |
| `Standard_B2ats_v2` VM (tracking server) | 0.44 THB/h | while running |
| Static public IP + 30 GB HDD of the VM | ~0.24 THB/h | always |
| Container registry, Basic (from Lab 1) | ~5.5 THB/day | always |

**Per trial.** `cost_thb` in the comparison table is fit time × node rate: 0.0014–0.0039 THB per
trial, 0.0296 THB for all 12. **It undercounts the bill.** A job also pays for node start-up and
for the idle minutes before scale-down:

| Job | Queued → start (node start-up) | Running |
|---|---|---|
| Task 1 `itcs355-d0a6afb1b9e6` | 4.1 min | 2.4 min |
| Study part 1 `itcs355-7b68899ff665` | 5.7 min | 2.7 min |
| Study part 2 `itcs355-a78b2fedff1c` | 0.5 min (node still warm) | 1.8 min |
| Seeds `itcs355-7ef754377692` | 2.1 min | 2.4 min |

About 10 minutes of node time per job from a cold start, about 1.2 THB, against 0.03 THB of fitting.

**Monthly retraining.** One study a month on a cold node: about 1.2 THB of compute, plus about an
hour of tracking server (0.44 THB). The standing costs dominate: static IP and disk about 175 THB a
month, the container registry about 165 THB a month. Retraining is under 1% of the monthly bill.

**Total spend for Lab 2.** «fill: from Azure Cost Management once posted»

## Registered model and promotion (Task 4)

«fill: registered name/version, run ID, and the eight lineage fields from `make register`»

**Promotion.** The chosen version is promoted with the MLflow alias `staging` (`make promote`);
MLflow 3 uses aliases in place of the old stages.

**Who may promote staging → production.** Not the person who trained the model: promotion needs
a second pair of eyes. In a real organisation it belongs to the owner of the production service,
the ML platform or MLOps lead, with sign-off from the maintenance operations manager, because the
scores decide where technicians are sent. Before promoting they should require:

1. **Lineage complete on the version** (commit, data version, image digest, seed), and a rebuild
   from those that reproduces the metric within tolerance.
2. **Evaluation on data newer than the tuning set.** ROC-AUC no worse than the current production
   model by more than seed noise (about 0.002), and calibration checked: Brier score, and the
   predicted against the actual failure rate.
3. **The operating threshold agreed with maintenance**, with the inspections per week it implies
   and the recall at that threshold. Not the default 0.5, which here would catch 5% of failures.
4. **`reload_check.py` passing from the registry** inside the serving image.
5. **A rollback.** The previous production version stays registered, and moving the alias back is
   one command.

## Notes for the grader — Lab 2

- **Registry access.** `reload_check.py` needs `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and
  `MLFLOW_TRACKING_PASSWORD`; the credentials are shared privately, never in Git.
  «fill: when the server will be up»
- **Commands the handout names but the Makefile lacked.** `make train-remote` is added;
  `make tune-remote`, `make seeds-remote`, `make register` and `make promote` are added for the
  study, the seed runs and Task 4. `make cost-report` does not exist; spend is taken from Azure Cost
  Management.
- **`make reload-check` needed `MODEL_REGISTRY_NAME` exported in the shell.** It now falls back to
  `cloud.env`.
- **Teardown.** `make teardown` calls the adapter's `teardown()`, which `base.py` schedules for Lab 5
  ("do not implement ahead"). The cluster scales itself to zero nodes and the tracking VM is
  deallocated with the Azure CLI; the registered model and its artifacts stay for Lab 3, as the
  handout says.
- **Cloud:** Azure, region `eastasia`. The smallest general-purpose size, `Standard_B1s`, is not offered there;
  the server uses `Standard_B2ats_v2`, about the same price.
