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

**TODO (before submitting):** run `make image-push` and paste the digest reference it prints here,
in the form `itcs3556688067.azurecr.io/itcs355/itcs355-lab1@sha256:…`.

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

**The DVC remote is written in DVC's Azure form.** `.dvc/config` points at
`azure://itcs355/itcs355/dvc` with `account_name = itcs3556688067` — the same location as
`${BLOB_URI}/dvc`. The literal `https://…blob.core.windows.net/…` form is read by DVC as a plain
HTTP remote that sends no Azure credentials, so `dvc push` to a private account cannot work with it.

**The DVC remote is private, and `make reproduce` does not need it.** `dvc pull` requires an Azure
identity with *Storage Blob Data Reader* on the storage account `itcs3556688067`; ask me and I will
grant it. Without that access, `make reproduce` still works, because `make data` regenerates the
exact dataset from the seed — the data fingerprint above, and `dvc status` reporting
*Data and pipelines are up to date*, confirm it matches the tracked version.

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
- [ ] Image builds for `linux/amd64` ✔ — **push pending** (`make image-push`, then fill in *Container image*)
- [ ] `dvc push` completed — **pending**; remote configured and reachable
- [x] Five or more tracked runs with params, metrics, data fingerprint, and commit SHA
- [x] Every instruction block from the template is gone (the course-materials block at the top stays)
- [x] `git log -p | grep -i -E "secret|password|AKIA|BEGIN PRIVATE"` — the only matches are words in the course's own documentation and comments; no credential appears in history
