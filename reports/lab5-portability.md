# Lab 5 Task 4 — the portability test

## Part A — audit

`make portability-audit` passes: no provider string in `src/`, `service/`, `monitoring/` or
`tests/`. Every provider call — Azure and now AWS — is in `cloudlayer/`.

## Part B — the swap, demonstrated

**Deviation, stated first.** No AWS or GCP account with billing was available; both need a
credit card. The second provider is therefore **AWS emulated by LocalStack** (S3 and Lambda,
free edition) in Docker on the laptop. Nothing in `cloudlayer/aws.py` refers to the emulator.
boto3 reads the standard `AWS_ENDPOINT_URL`, so the emulator is three lines in
[`portability/aws.env`](../portability/aws.env). With those lines removed, the same adapter
talks to AWS. What the emulator cannot show is AWS's own permissions (IAM), its quotas or its
bill. Those are the parts most likely to differ.

```bash
make swap-demo      # emulator up, bucket + function provisioned, then upload/download/invoke
```

Result ([`reports/lab5/portability-demo.txt`](lab5/portability-demo.txt)):

| Method | What happened |
|---|---|
| `upload` | the training data, 321,437 bytes, to `s3://itcs355-portability/lab5/data/raw/sensors.csv` |
| `download` | back into a directory that did not exist; sha256 identical |
| `invoke` | the serving function with the Lab 3 smoke rows. The batch probabilities equal the same model scored locally (largest difference 4.3 × 10⁻¹⁹). Cold start 9.0 s, warm 77 ms. An out-of-range `load_pct` returns 422, as the Azure endpoint does |

`invoke` calls a Lambda function. On AWS proper the model would sit behind a SageMaker
endpoint, but SageMaker is not in the emulator's free edition. The function runs
[`cloudlayer/lambda_handler.py`](../cloudlayer/lambda_handler.py): the same request schema,
feature list and model file as `service/app.py`, behind a different transport.

**The course's own check, run as shipped** ([`reports/lab5/swap-check.txt`](lab5/swap-check.txt)).
`scripts/portability_swap_check.py` swaps only `provider` inside an otherwise Azure `Config`:

- **All three fail.** `BLOB_URI` is still an Azure `https://…blob.core.windows.net` address,
  and there are no AWS credentials.
- **With the AWS storage settings supplied,** upload and download pass. `invoke` returns 422,
  because the check sends `{"rows": []}` and the service's schema requires at least one row.
  That is the check's assumption, not the adapter's fault, and it is left as found.

## Part C — the write-up

**The hardest method was `invoke`, by a distance.** Upload and download were about 15 lines
each and passed first time; object storage really is the same product under two names.
`invoke` needed three things the Azure version did not:

- **Somewhere to run the model.** That meant a new handler, a Linux bundle of scikit-learn
  built for Python 3.12, and a provisioning step.
- **A new way to read errors.** Lambda answers HTTP 200 when the function itself failed,
  putting the failure in a `FunctionError` field. The service's real status, headers and body
  come back wrapped one level deeper. The Azure adapter read the status straight from HTTP.
- **Accepting a cold start.** It was 9 s here; the Container App took 35 s in Lab 3.

**Where the abstraction genuinely leaked: configuration.** The adapter seam held, and `src/`
needed no change. But `cloud.env` is not provider-neutral, and the course's swap check
assumes it is:

- `BLOB_URI` has a provider-specific shape.
- `REGION` holds an Azure region name that boto3 would happily build an endpoint from.
- Identity is a different kind of thing on each side. Azure uses a managed identity or a CLI
  login with no secret. AWS uses an access key pair, or a role that only exists inside AWS.

Swapping the adapter means swapping the configuration with it, and no interface hides that.

**A full migration: about 15–20 engineering days.** Three methods took one day. The rest is
where the work is:

| Work | Days |
|---|---|
| Training jobs and schedules on SageMaker (`submit_training`, `wait_training`, `schedule`) | 3 |
| A real endpoint with probes, revisions and the canary (`deploy`) | 3 |
| Monitoring in CloudWatch (`emit_metric`, `recent_inputs`, alerts) | 2 |
| The LLM step on a different model family (`generate`) — re-baselining the golden set, and expecting regressions | 2 |
| `push_image` and `teardown` | 1 |
| IAM roles and GitHub OIDC trust | 2 |
| Moving the MLflow server | 1 |
| Repeating the load test, canary and drift exercise as evidence | 3 |
| **Total** | **≈ 17** |

The adapter saves the code changes in `src/`, perhaps a quarter of that. It does not save
the identity, quota and operational work, which is most of it.

**Was it worth building? Yes — but not for portability.** Nobody here is about to change
cloud, and paying for that option alone would be poor value. What paid for the seam was
something else. Every provider call sits in one directory, so:

- the portability audit can prove `src/` is clean;
- the LLM gate, the drift statistics and the guard are tested without a cloud account;
- this swap took a day instead of a rewrite.

On a system of this size the seam costs maybe 10% more code. I would build it again for
those reasons. I would not defend it to a budget holder as insurance against changing cloud.
