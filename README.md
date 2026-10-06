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
  (see *Compute and cost*). The VM is deallocated when not in use; its daily 23:00 auto-shutdown is
  switched off while Lab 2 is being graded, so the registry stays reachable.

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

Evidence from the jobs' own logs (Azure ML `user_logs/std_log.txt`). The second job skips what the
checkpoint holds, and its running spend continues from the first job's 0.0076 THB:

```
itcs355-7b68899ff665 (interrupted)
trial 0: {'max_depth': 4, 'class_weight': None, 'max_features': 'sqrt'} -> val_roc_auc=0.8405 cost=0.0016 THB  cumulative=0.0016
...
trial 3: {'max_depth': 4, 'class_weight': 'balanced', 'max_features': 1.0} -> val_roc_auc=0.8307 cost=0.0023 THB  cumulative=0.0076

itcs355-a78b2fedff1c (resubmitted)
trial 0: already done, skipping (resumed from checkpoint)
trial 1: already done, skipping (resumed from checkpoint)
trial 2: already done, skipping (resumed from checkpoint)
trial 3: already done, skipping (resumed from checkpoint)
trial 4: {'max_depth': 8, 'class_weight': None, 'max_features': 'sqrt'} -> val_roc_auc=0.8364 cost=0.0022 THB  cumulative=0.0099
...
trial 11: {'max_depth': 12, 'class_weight': 'balanced', 'max_features': 1.0} -> val_roc_auc=0.8150 cost=0.0035 THB  cumulative=0.0296
spent 0.0296 of 150.0 THB
```

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

**Total spend for Lab 2.** Azure Cost Management, as posted on 19 September 2026: **9.8 THB** (virtual machines, which include the training nodes, 4.50; network and static IP 2.80; storage 2.47), against the 150 THB budget. The container registry from Lab 1 adds about 5.5 THB a day on top. Postings lag by up to a day, and the tracking server keeps running while Lab 2 is graded.

## Registered model and promotion (Task 4)

**`itcs355-6688067` version 1**, from run `a51929352ced426ea7f9dc6fb27841fc` (trial-00: max_depth 4,
class_weight None, max_features sqrt). Registered with `make register RUN_ID=a51929352ced426ea7f9dc6fb27841fc`.
The adapter copies the lineage onto the model **version**, not only the run, and refuses to register
if any field is missing:

| Field | Value |
|---|---|
| `git_commit` | `4bb88f97798789cdcb331adc4284d35ebb138d83` (the commit the study job ran) |
| `data_version` | `1c886b512c8a5c9bf723da1cd119fc80.dir` (DVC md5 of `data/raw`) |
| `mlflow_run_id` | `a51929352ced426ea7f9dc6fb27841fc` |
| `training_job_id` | `itcs355-7b68899ff665` (Azure ML job) |
| `image_digest` | `sha256:f0253d25002cc08736957b396dfce578e6a363ff55aabc3df0788f10782ef7db` |
| `seed` | `20260101` |
| `metric_val` | `0.8404552730949226` (val ROC-AUC) |
| `metric_test` | `0.8543112516622451` (test ROC-AUC) |

MLflow 3 logs models as logged-model objects, so the version's `source` is
`models:/m-604f180487784f819a6361ba22b7fa1e`; its `run_id` still points at the run above.

**Promotion.** Version 1 is promoted with the MLflow alias `staging` (`make promote VERSION=1`);
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
  **The server stays up continuously from Saturday 19 September 2026 until at least the end of
  Session 3 (Monday 21 September).** After that it is started on request, in about two minutes.
  Tested from a fresh clone with no `cloud.env` and only those three variables:

  ```
  $ python scripts/reload_check.py --name itcs355-6688067 --version 1
  loading models:/itcs355-6688067/1
    reading 125: p(failure)=0.0203
    reading 126: p(failure)=0.0658
    reading 127: p(failure)=0.0293
    reading 128: p(failure)=0.0262
    reading 129: p(failure)=0.0137

  PASS  model reloaded from the registry and scored rows
  ```

  Without the password the same command fails with HTTP 401. `data/raw/sensors.csv` must exist
  first (`make data` or `dvc pull`), because `reload_check.py` scores rows from the local test split.
- **Commands the handout names but the Makefile lacked.** `make train-remote` is added;
  `make tune-remote`, `make seeds-remote`, `make register` and `make promote` are added for the
  study, the seed runs and Task 4. `make cost-report` does not exist; spend is taken from Azure Cost
  Management.
- **Credential scan.** `python scripts/scan_secrets.py`, which replaced the old word-matching grep
  upstream, reports `CLEAN` on this repository's whole history. The tracking server's password was
  searched for explicitly and appears 0 times. Shell variable names such as `POSTGRES_PASSWORD` in
  `infra/tracking-server/make_secrets.sh` hold values generated at run time by `rand`; the old grep
  matched those names, the new scanner matches the shape of a value and does not.
- **`make reload-check` needed `MODEL_REGISTRY_NAME` exported in the shell.** It now falls back to
  `cloud.env`.
- **Teardown.** `make teardown` calls the adapter's `teardown()`, which `base.py` schedules for Lab 5
  ("do not implement ahead"). The cluster scales itself to zero nodes and the tracking VM is
  deallocated with the Azure CLI; the registered model and its artifacts stay for Lab 3, as the
  handout says.
- **Cloud:** Azure, region `eastasia`. The smallest general-purpose size, `Standard_B1s`, is not offered there;
  the server uses `Standard_B2ats_v2`, about the same price.

## Checklist before you submit — Lab 2

- [x] `submit_training()` and `wait_training()` implemented and working — `cloudlayer/azure.py`; jobs listed above
- [ ] 12+ trials on discounted compute, checkpointed, all tracked — 12 trials, checkpointed and tracked, but on **dedicated** compute: the subscription's low-priority quota is 0 (see *Study, interruption and resume*)
- [x] Interruption survived and resumed — evidence in logs (simulated by cancelling; see above)
- [x] Comparison artifact in `reports/` — `reports/lab2-comparison.md`
- [x] 200-word justification covering all four required points — in the same file
- [x] Model registered with all eight lineage fields — `itcs355-6688067` version 1
- [x] Promotion step performed, with a note on who should own it — alias `staging`
- [x] `reload_check.py` runs from the registry and scores rows — tested from a fresh clone
- [x] Cost recorded per trial, total under 150 THB — 0.0296 THB of fitting; 9.8 THB billed for the whole lab
- [ ] `make teardown` run — not run: `teardown()` is Lab 5 work in `base.py`. The cluster scales itself to zero, the tracking VM is deallocated after grading, and the registered model stays for Lab 3

---

# Lab 3 — Serving, load testing and rollback

## What runs where

| Piece | Where | Why |
|---|---|---|
| Inference service | one container image, FastAPI, `service/` | the same image runs locally and on the platform |
| Endpoint | **Azure Container Apps**, consumption plan, `itcs355-predict` | `getting-started-azure.md`: an Azure ML managed online endpoint needs `ceil(1.2 × instances) × cores` of quota, so one `Standard_DS3_v2` asks for 8 vCPU against a student cap of about 3 that cannot be raised |
| Model | loaded once at startup from the MLflow registry by version (`models:/itcs355-6688067/1`) | Lab 2's registry, unchanged |
| Registry credentials | Container Apps **secrets**, referenced by the container as `secretref:` | better than Lab 2's job environment variables; Lab 4 moves them to Key Vault |
| Image pull | user-assigned managed identity `itcs355-serve-id` with **`AcrPull` on the registry only** | no registry password anywhere |

```bash
make serve                        # run locally on :8080 with a file-backed model
make serve-image                  # build the serving image
make deploy VERSION=1             # push it and create/update the Container Apps revision
make smoke                        # three known payloads against the live endpoint
make loadtest TARGET=<url>/predict  # k6 at 1, 10 and 50 users, summaries into reports/lab3/
make teardown LAB=3               # delete everything tagged lab=3
```

## Health and readiness are different questions

`/health` answers "the process is up"; `/ready` answers "the model is loaded and can score".
The Container Apps revision declares the first as its liveness probe and the second as its
readiness probe, so a replica that is still loading the model is not sent traffic. This is not
theoretical here: during the first deployment the model failed to load, `/health` returned 200
for twenty minutes while `/ready` returned 503, and the ingress correctly served 404 rather
than routing to a container that could not answer.

## Three failures worth recording

1. **The registered model would not load in the container.** `mlflow.sklearn.load_model`
   raised `Untrusted types found in the file: ['sklearn.tree._tree.Tree']`. MLflow 3 stores
   scikit-learn models with skops, and skops refuses to rebuild that type unless the caller
   names it: it holds raw node indices that scikit-learn reads without bounds checking. The
   list belongs in the MLmodel file, written at log time by `log_model(skops_trusted_types=…)`
   — the argument the course repository added after this model was registered. `service/app.py`
   therefore catches that one error, downloads the artifact by version and loads it with
   `skops.io.load(trusted=["sklearn.tree._tree.Tree"])`, which is a statement about provenance:
   the version's lineage names the commit, the data version and the training job that made it.
   This is exactly the serialization risk the handout warns about, met in practice.
2. **Two uvicorn workers do not fit 1 GiB.** The provided `Dockerfile.serve` starts two
   workers; each loads its own copy of the model and of scikit-learn. On a 0.5 vCPU / 1 GiB
   replica the platform killed the container with SIGTERM about a second after the second copy
   finished loading, eight times in a row. The image now starts **one** worker and concurrency
   comes from replicas.
3. **The CLI's warning line broke a lookup.** `_run` in the adapter returns stdout and stderr
   together, which is right for `docker push` (the digest is on stderr) and wrong for
   `--query` lookups: the `containerapp` extension prints "WARNING: The behavior of this
   command has been altered by the following extension" on stderr, so an emptiness test saw a
   non-empty string and the adapter took the update branch for an app that did not exist.
   Lookups now go through `_query`, which reads stdout only.

Two platform constraints shaped the deployment as well. An **express** Container Apps
environment refuses `activeRevisionsMode: Multiple` (`ExpressEnvironmentFeatureNotSupported`),
which Task 4's canary needs, so the environment is created with `--environment-mode
WorkloadProfiles`. And `az containerapp create --yaml` fails inside this preview extension
with a 400 from the API, so the adapter creates the app with CLI flags and then patches the
probes through the ARM API with a sanitised template — the read spec carries fields
(`imageType`, `targetPortHttpScheme`) the write API rejects.

## Load test, cold start, batch and cost

Everything is in **[`reports/lab3-load.md`](reports/lab3-load.md)**, with the raw k6 summaries
in `reports/lab3/`. The short version:

- **Target, committed and pushed before the endpoint existed** (`5cb7e21`): p95 under 300 ms
  at 10 concurrent users, errors under 1%, measured in-region on a warm instance.
- **Breaking concurrency on 0.5 vCPU: five users** (p95 276.7 ms at four, 372.6 ms at five).
- **The configuration that meets the target: 1 vCPU × 3 replicas** — p95 292.3 ms, p99 353.1 ms,
  99.4 req/s, no errors.
- **Cold start 35.5 s**, of which 4.7 s is loading the model; a warm request is 0.22 s.
- **100 rows in one batch call take 19.0 ms**, against about 1,790 ms for 100 single calls.
- **About 0.16 THB per 1,000 predictions** at a 20% utilisation assumption, stated explicitly.

## Canary and rollback

Model version 2 (val ROC-AUC 0.8261 against version 1's 0.8405) went out at 10% of traffic.
The monitor scores every response against its known label and compares cohorts by the
`x-model-version` header, without reading the traffic weights or which version is new. It
alerted after **13.0 minutes** at 3 sigma; traffic went back to 100% baseline and the canary
revision was deactivated. Timestamps, the traffic weights before and after, and every scored
response are in `reports/lab3/`.

## Notes for the grader — Lab 3

- **The endpoint is deleted.** `make teardown LAB=3` removes everything tagged `lab=3`: the
  container app, its environment and the serving identity. The Lab 2 registry, storage account
  and container registry carry other lab tags and are untouched, because Lab 3's own handout
  says the registered model stays for later labs. Output of the teardown run is in
  `reports/lab3/teardown.txt`.
- **`make teardown` as provided would have deleted Lab 1.** The target called
  `teardown(cfg.tags(1))` — lab 1 hardcoded — so running it during Lab 3, as this handout asks,
  would have removed the storage account and the container registry. It now requires an
  explicit `LAB=` and refuses without one; `scripts/teardown.py` and the adapter's `teardown()`
  delete by tag, never by resource group. Writing it in Lab 3 rather than Lab 5 is a deviation
  from `base.py`, forced by this handout's checklist.
- **`make deploy` and `make smoke` did not exist** in the provided Makefile; both are added,
  along with `INSTANCE` (`<cpu>/<memory>`) for the instance-size experiment.
- **Load generated from inside the region.** k6 runs on the tracking-server VM in `eastasia`
  through the Azure agent, because the NSG admits SSH only from the address the VM was created
  from and that address had changed. A laptop in Bangkok adds 40–60 ms to every measurement.
- **Model loading through the adapter.** The handout asks for the model to arrive through the
  adapter; the provided `service/app.py` loads it from the MLflow registry directly, and that
  code is kept. MLflow is the portable seam here — no provider name appears in `service/`, and
  `make portability-audit` passes.

---

# Lab 4 — CI/CD, observability and drift

## What runs where

| Piece | Where | Why |
|---|---|---|
| CI | GitHub Actions, `.github/workflows/ci.yml`, on every pull request and push to main | the handout's order: secret scan → lint → unit → data contract → behaviour → build → integration |
| CD | `.github/workflows/cd.yml`, after a green CI run of a **push to main** only | a pull request's CI run never deploys |
| CD identity | user-assigned identity `itcs355-gha-id`, **OIDC federated credential**, Contributor on the resource group | no client secret or stored key exists anywhere |
| Staging endpoint | Container App `itcs355-staging`, 0.5 vCPU, scale 0–1, Single revision mode, revision suffix = commit | scales to zero when unused, so staging costs nothing between deploys |
| Served model | the version the registry alias **`staging`** (set in Lab 2) points to, resolved once by CD and deployed as a number | the alias says what is approved; the revision must say what is running, and keep saying it after a restart |
| Dashboard | Prometheus + Grafana in Docker on the laptop, scraping `/metrics`; `monitoring/dashboard.json` is the dashboard | dashboard as code, provisioned on start |
| Drift detector | `monitoring/drift_job.py` on an **Azure ML schedule**, every 15 minutes, on the scale-to-zero `ded-ds2` cluster | the handout names Azure ML schedules for Azure |
| Production inputs | the service logs one line per scored row; the Container Apps environment ships stdout to Log Analytics `itcs355-logs-l4`; the job reads the last 500 back | the service stays provider-neutral; the platform does the collecting |
| Alert | the job writes `drift_threshold_ratio` to Azure Monitor custom metrics; metric alert `itcs355-drift-alert` → action group → **email** to the student address | a real channel; Line Notify no longer exists |

```bash
make test                                   # unit, data contract, behaviour, service
make threshold-study                        # reports/lab4/threshold-study.md
make inject-drift && make drift             # the local version of Task 6
make monitor ENDPOINT=$(cat reports/lab4-staging-endpoint.txt)   # Grafana on http://localhost:3000
make drift-schedule / make drift-unschedule # the Azure ML schedule
python scripts/send_traffic.py --rows 100 --repeat 90 --switch-to data/current.csv --switch-after 30
make teardown LAB=4                         # resources tagged lab=4, and the schedule by name
```

## Tests, and the incident each data contract test would have caught (Task 1)

| Category | Where | Fails when |
|---|---|---|
| Unit | `tests/test_drift.py` (PSI and KS arithmetic), `tests/test_service.py` (validation, batch = singles, metrics split 4xx/5xx, rolling statistic) | our code changes |
| Data contract | `tests/test_data.py` | the producer of the data changes |
| Model behaviour | `tests/test_model_behaviour.py` (healthy machine scores low, risk rises with wear, not constant, latency under the 300 ms target) | the model changes |
| Integration | `ci.yml` build job: build the serving image, start it, `POST /predict`, assert the response shape **and that `model_version` is the commit SHA** | the image does not serve what was built |

| Data contract test | The production incident it would have caught |
|---|---|
| `test_schema_columns_present_and_typed` | The sensor gateway renames `ambient_humidity` to `humidity` in an update, or starts writing `reading_id` as text. Without the test, `pd.DataFrame(rows)[FEATURES]` fails at serving time, or a type silently changes under the model. |
| `test_no_nulls_in_required_columns` | A gateway outage writes empty fields for an afternoon; the rows reach training as NaN and scikit-learn either refuses them in the scheduled retrain or, after someone adds an imputer, learns from invented values. |
| `test_features_within_plausible_ranges` | **Task 3's bad commit:** a firmware update makes a batch of sensors report Fahrenheit under the name `temp_c` (231 °C). Same for pressure arriving in psi instead of kPa. The model would score impossible machines with full confidence. |
| `test_target_is_binary_and_not_degenerate` | The join to the maintenance log breaks and every label comes through as 0. A model trained on it reaches 100% accuracy by never predicting a failure. |
| `test_identifier_is_unique` | An export job re-runs and appends instead of replacing, so every reading appears twice and duplicates can land on both sides of a split. |
| `test_no_machine_leaks_across_splits` | Someone "simplifies" the split to random rows: readings of one machine appear in training and test, and the validation score stops predicting production (the Lab 1 leakage test, still in CI). |

## The blocked bad commit (Task 3)

Branch `lab4-bad-contract` changes the data producer, `scripts/make_dataset.py`: machines 200 and
up report temp_c in Fahrenheit. CI on that pull request fails in the **Data contract tests** step:

```
FAILED tests/test_data.py::test_features_within_plausible_ranges - AssertionError: temp_c above plausible ceiling: 231.396
```

The unit tests in the step before it pass, so the contract test is the one that stops it. The
`build` job (image build, integration test) was skipped, so nothing was built, pushed or
deployed, and the pull request was closed without merging.

Evidence: [pull request #1](https://github.com/Guysiravich/public_teaching_mlaiops/pull/1) and its
[failing CI run](https://github.com/Guysiravich/public_teaching_mlaiops/actions/runs/37481561827)
— `test` job: Lint ✓, Portability audit ✓, Generate dataset ✓, Unit tests ✓, **Data contract
tests ✗**, Model behaviour tests skipped; `secrets` ✓; `build` skipped. The assertion message
above is from that step's log, captured in
[`reports/lab4/ci-blocked-commit.png`](reports/lab4/ci-blocked-commit.png): `1 failed, 9 passed`.

## Dashboard and SLO (Task 4)

`monitoring/dashboard.json` is a real Grafana dashboard, provisioned by `monitoring/compose.yaml`:
request rate; error rate split into 4xx and 5xx; p50/p95/p99 of `/predict` with the 300 ms target
drawn on it; **temp_c rolling mean and standard deviation over the last 500 inputs** (the
feature-distribution statistic — mean for a shift, standard deviation for a change of scale); the
model version in production; and the number of rows in the rolling window. All of it comes from
`/metrics` on the service (`prometheus-client`, added to both hash-locked requirement files).

The SLOs are in `monitoring/slo.yaml`, one sentence each for target, window and the response when
the budget is spent: **availability 99.5% over 30 days** (deploy freeze except rollbacks and the
fix), **p95 under 300 ms over 7 days** (a ticket for capacity, not a page — a 7-day maintenance
decision tolerates seconds), **model age under 35 days** (page the model owner; never retrain by
hand on unchecked data). The reasoning for each number is in the file.

## Drift threshold, and why (Task 5)

`reports/lab4/threshold-study.md` measures the two numbers a threshold has to sit between, against
the training reference, with windows of the held-out machines:

- **Noise.** With no drift at all, five features stay under PSI 0.057 at p99 in a 500-row window.
  **load_pct does not**: it differs between machines by design, so the held-out pool already scores
  0.134 against training and drift-free windows reach 0.213 at p99 (0.252 at worst).
- **Harm.** A temp_c offset of +3 °C gives PSI 0.116, +4 °C gives 0.185, +6 °C 0.41.

So the thresholds are **0.10 for five features and 0.30 for load_pct**, over the **last 500
inputs**. The library default of 0.25 would have missed a +4 °C sensor offset and would still
have fired on load_pct's normal machine-to-machine variation; a single 0.10 would fire on load_pct
on every run. A window shorter than 500 rows charts sampling noise as drift (p99 of 0.10 at 200
rows), so the job skips scoring and reports the shortfall when it has fewer.

Which statistic caught which fault, run locally on all three `inject_drift` modes:

| Mode | temp_c PSI | KS | Caught at 0.10 |
|---|---|---|---|
| `shift` +6 | 0.383 | 0.246 | yes |
| `scale` ×1.5 (mean unchanged at 79.58) | 0.206 | 0.114 | yes — PSI sees shape, the mean does not move at all |
| `mix` (reweighted machines) | 0.009 | 0.034 | **no** — the realistic one stays under every threshold |

## Injected drift (Task 6)

| Time (UTC, 6 Oct) | Event |
|---|---|
| 13:35 | normal traffic, 100 held-out rows every 30 s |
| 13:50 | scheduled run on normal inputs: ratio **0.57** (temp_c PSI 0.057) — no alert |
| **13:50:23** | **injection**: inputs switch to `data/current.csv`, temp_c +6 °C |
| 14:00 | scheduled run "completed" in **zero seconds** — Azure ML reused the cached result |
| 14:20 | 14:15 run (after the fix): temp_c PSI **0.3505**, ratio **3.505** |
| **14:22:39** | **alert fired** — but no email: the address had not completed Azure's new receiver verification (failure 4 below) |
| 14:44 | receiver re-added, verification code sent and confirmed |
| 14:51 | one more run of the same job on the same last 500 inputs: ratio 3.505 |
| 14:56:49 | alert fired again — still no "Fired" email; its **"Resolved" email arrived** at 15:22, a minute after the resolve |
| 15:27 | one more run, same inputs, ratio 3.505 (the Gmail address added as a second receiver) |
| **15:31:51** | **alert fired; "Azure: Activated Severity: 2 itcs355-drift-alert" arrived at 15:32** (Value 3.5052, Threshold 1) |

**Detection time: 32 min 16 s**, of which one 15-minute cycle was lost to the cached run; without
it the 14:00 run would have alerted at about 14:07. The dashboard shows the rolling mean moving from
79.6 to 85.5 °C within three minutes of the injection with the standard deviation flat
(`reports/lab4/dashboard-shift.png`). Post-mortem: **[`reports/lab4-postmortem.md`](reports/lab4-postmortem.md)**
— the decision is neither retrain nor roll back, because the cause is a broken sensor, not a
changed world.

## Failures worth recording

1. **GitHub's OIDC subject carries IDs.** The federated credential written as
   `repo:Guysiravich/public_teaching_mlaiops:environment:staging` was refused (`AADSTS700213`); the
   token says `repo:Guysiravich@52388254/public_teaching_mlaiops@1367593417:environment:staging`.
   The IDs stop a deleted-and-recreated repository of the same name inheriting the trust.
2. **`deploy()` failed three ways on its first CD run**: `get-shared-keys` does not take `--ids`;
   `containerapp create --yaml` still returns the Lab 3 400, so the ARM body now goes to ARM with
   `az rest --method put`; and the CLI's default express environment refuses a revision suffix, so
   the environment is created with `--environment-mode WorkloadProfiles` (the Lab 3 README's
   description of "CLI flags and an ARM patch" is superseded by the PUT).
3. **An Azure ML schedule will not take a command job** ("Unsupported job type 'CommandJob'"), so
   the job runs as the single step of a pipeline — which then rejected the `wasbs://` output folder
   ("DataStore name is missing"; the drift job has no output, so it has none now) and then
   **reused the cached result** of an identical step, so a run "completed" without reading the new
   inputs. `force_rerun` fixes it. A detector that silently stops detecting is the failure the
   post-mortem's heartbeat alert is for.
4. **An alert that fires is not an alert that arrives.** Azure Monitor now sends an email
   receiver a one-time code and delivers nothing to it until the code is entered; the code
   expires in 30 minutes. The first alert (14:22:39) fired in Azure and reached nobody; the
   second (14:56:49), fired just after verification, was not delivered either although its
   history says the action group executed — only its "Resolved" notice arrived. The third
   (15:31:51) arrived within a minute. Every one of them showed `ActionsTriggered` in Azure,
   so checking the alert state would have passed on the first. The handout's "an alert that
   actually arrived somewhere" is the right test, and the screenshot of the delivered email is
   kept with the submission (not committed: it prints the subscription ID).

## Notes for the grader — Lab 4

- **Deviations from the provided files, each deliberate:**
  - `monitoring/dashboard.json` is now an importable Grafana dashboard. The provided error-rate
    query returns nothing: the numerator has a `status_class` label the denominator lacks, so
    the division needs `ignoring(status_class) group_left`.
  - `cloudlayer/base.py` gains two methods beyond the eleven, `recent_inputs()` and
    `schedule()`. Reading the platform's logs and creating a schedule are provider calls, and
    provider calls belong in `cloudlayer/`.
  - CD rebuilds the serving image from the commit CI tested instead of carrying CI's image
    across. The provided `cd.yml` runs on another runner, where CI's image does not exist. The
    base image is pinned by digest and every package by hash.
  - `azure-identity` is added to `requirements.txt`. The drift job authenticates as the
    cluster's managed identity, as the adapter docstring suggests.
- **Secrets.** GitHub holds `AZURE_CLIENT_ID`, `AZURE_TENANT_ID` and `AZURE_SUBSCRIPTION_ID`
  (identifiers), and `STAGING_ENV`. `STAGING_ENV` carries the cloud.env lines CD needs,
  including the tracking server's password. Nothing secret is committed, and
  `make scan-secrets` is clean on the full history.
- **The schedule is deleted by name.** Azure ML schedules carry no resource tags, so
  `make teardown LAB=4` deletes `itcs355-drift` explicitly and then lists what is left.

## Checklist before you submit — Lab 4

- [x] Unit tests, 2+ data contract tests, 1+ model behaviour test, 1 integration test
- [x] README naming the incident each data contract test would have caught — above
- [x] CI pipeline running the full sequence, secrets via OIDC and the repository secret store
- [x] Images tagged by commit SHA (CI builds `itcs355-serve:${{ github.sha }}`; staging runs `--g<sha>` revisions)
- [x] CD to staging on green, main only
- [x] Evidence of the blocked bad commit — failing run and the test that caught it
- [x] Dashboard with the five required signals
- [x] SLO: target, window, and error-budget response
- [x] Scheduled drift detector with a justified threshold, alerting to a real channel
- [x] Injected drift: alert evidence, timestamps, detection time
- [x] Five-line post-mortem
- [ ] `make teardown` run — TEARDOWN_PLACEHOLDER
