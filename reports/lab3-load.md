# Lab 3 — load test

## Target, stated before measuring

| | Target |
|---|---|
| p95 latency, `POST /predict`, 10 concurrent users | **under 300 ms** |
| Error rate | **under 1%** |
| Where measured from | a client in the same Azure region as the endpoint (not a laptop in Bangkok) |
| Instance | 0.5 vCPU / 1 GiB, one replica, warm |
| Cold start | reported separately, not inside the p95 above |

Why 300 ms: the caller is a dispatch screen a person reads, not a service inside a request
chain. A third of a second is below what a person notices, and it leaves room for one Azure
hop. The threshold lives in `loadtest/k6.js`, so the load test fails the build if it is missed.

This section, and the threshold in `loadtest/k6.js`, were committed and pushed **before the
endpoint existed** (commit `5cb7e21`). Every table below was filled in afterwards.

**Method.** k6 2.3.0 runs on the tracking-server VM in `eastasia`, the same region as the
endpoint, so the numbers are the service's and not the 40–60 ms to a home connection in
Bangkok. Each level runs 45–60 s. Raw k6 summaries are committed in `reports/lab3/`.
Requests come from the VM through the Azure agent, because the NSG only admits SSH from the
address the VM was created from and that address had changed.

## Results

Single-row `POST /predict`, one replica of 0.5 vCPU / 1 GiB unless stated.

| Concurrency | Throughput (req/s) | p50 | p95 | p99 | Error rate | File |
|---|---|---|---|---|---|---|
| 1 | 29.0 | 20.1 ms | **65.8 ms** | 71.4 ms | 0% | `k6-single-1.json` |
| 2 | 25.2 | 88.1 ms | 110.9 ms | 167.4 ms | 0% | `k6-half-2.json` |
| 3 | 24.3 | 110.1 ms | 191.2 ms | 207.8 ms | 0% | `k6-half-3.json` |
| 4 | 21.7 | 191.8 ms | 276.7 ms | 301.0 ms | 0% | `k6-half-4.json` |
| 5 | 20.7 | 221.9 ms | **372.6 ms** | 402.3 ms | 0% | `k6-half-5.json` |
| 10 | 15.5 | 666.8 ms | 896.6 ms | 997.0 ms | 0% | `k6-single-10.json` |
| 50 | 21.3 | 410.0 ms | 3,786 ms | 28,698 ms | **63.1%** | `k6-single-50.json` |

### Breaking point

**Five concurrent users.** p95 is 276.7 ms at four and 372.6 ms at five, so the target is
crossed between them. Throughput does not rise with concurrency — 29 req/s at one user, 21
at five, 15 at ten — which says the replica is saturated at one request in flight and the
rest of the time is queueing. At 50 users the queue collapses: 63% of requests fail and p99
reaches 29 s.

### The configuration that meets the target

| Configuration | 10 users: p95 | Throughput | Meets the 300 ms target |
|---|---|---|---|
| 0.5 vCPU × 1 replica | 896.6 ms | 15.5 req/s | no |
| 1 vCPU × 1 replica | 434.3 ms | 32.2 req/s | no |
| 0.5 vCPU × 3 replicas | 655.0 ms | 50.2 req/s | no |
| **1 vCPU × 3 replicas** | **292.3 ms** | **99.4 req/s** | **yes** (p99 353 ms, 0% errors) |

Doubling the CPU of one replica is not enough on its own, and widening to three small
replicas is not either: this service needs both. The shape is consistent with one request
at a time per worker — the single uvicorn worker means concurrency comes from replicas.

### Cold start

The app would not scale to zero on its own after 25 minutes of idling, so the replica was
removed deliberately (revision deactivated, then reactivated) and the first request timed.

| | |
|---|---|
| First successful `POST /predict`, run 1 | **35.7 s** |
| First successful `POST /predict`, run 2 | **35.5 s** |
| Warm request immediately after | 0.24 s / 0.22 s (from Bangkok) |
| Of which, loading the model from the registry | **4.7 s** (container log: server process at 14:40:34, model loaded at 14:40:38.9) |

So roughly 31 s is the platform starting a container and pulling a 274 MB image, and 5 s is
the model. This is why the deployed configuration keeps `minReplicas` at 1: on a scale-to-zero
service the first user after an idle period would wait half a minute.

### Batch against single calls

One user, 1 vCPU × 3 replicas, `POST /predict/batch`:

| Rows per call | p50 | p95 | p99 | Per-row p50 |
|---|---|---|---|---|
| 1 | 18.1 ms | 22.4 ms | 26.4 ms | 18.1 ms |
| 10 | 17.8 ms | 22.0 ms | 25.4 ms | 1.78 ms |
| 50 | 18.5 ms | 22.8 ms | 27.3 ms | 0.37 ms |
| 100 | 19.0 ms | 24.0 ms | 31.0 ms | **0.19 ms** |
| single `POST /predict` | 17.9 ms | 22.0 ms | 25.6 ms | 17.9 ms |

**100 rows in one call take 19.0 ms; the same 100 rows one at a time take about 1,790 ms.**
That is 94× faster, and the advantage is entirely per-call overhead: TLS, ingress, FastAPI
and one pandas frame are paid once instead of a hundred times. The advantage disappears when
the call becomes the queue: at 10 concurrent users on the small instance the server is already
the bottleneck, so batching only changes how many requests are waiting.

### Payload size

The same table read the other way: the request grows from about 130 bytes to 13 KB between 1
and 100 rows, and p50 moves 17.8 → 19.0 ms while p99 moves 25.4 → 31.0 ms. Serialisation is
not dominating at this size; it is visible only in the tail. The schema caps a batch at 100
rows (`BatchRequest`, `max_length=100`), so the first thing that breaks beyond this point is
validation, with 422, not latency.

### One instance size up

| | 0.5 vCPU / 1 GiB | 1 vCPU / 2 GiB | Change |
|---|---|---|---|
| p95 at 5 users | 372.6 ms | 185.0 ms | −50% |
| Throughput at 5 users | 20.7 req/s | 38.8 req/s | +87% |
| p95 at 10 users | 896.6 ms | 434.3 ms | −52% |
| Price while active | $0.000015/s | $0.000030/s | ×2 |
| Cost per 1,000 predictions at 100% utilisation | 0.031 THB | 0.039 THB | +26% |

Doubling the size halves the latency and nearly doubles the throughput, so the extra money
buys latency rather than throughput per baht — which is the trade-off worth stating out loud.

## Cost per 1,000 predictions

Azure Container Apps, consumption plan, `eastasia`, Retail Prices API checked 2026-09-22:
**active vCPU $0.000024 per second**, **memory $0.000003 per GiB-second**, **requests $0.40
per million**. The first 180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests
each month are free; the figures below ignore that grant, because a free tier is not a cost
model.

```
cost per 1,000 = replicas × (vCPU × $0.000024 + GiB × $0.000003) ÷ (throughput × utilisation) × 1,000
                 + 1,000 × $0.40/1,000,000
```

| Configuration | Throughput | Utilisation assumed | Cost per 1,000 |
|---|---|---|---|
| 1 vCPU × 3 replicas (meets the target) | 99.4 req/s | 100% | $0.00131 = **0.043 THB** |
| 1 vCPU × 3 replicas | 99.4 req/s | **20%** | $0.00493 = **0.164 THB** |
| 0.5 vCPU × 1 replica | 29.0 req/s | 20% | $0.00299 = 0.099 THB |
| Batch, 100 rows per call, 1 replica | 5,260 rows/s | 20% | $0.00003 = **0.001 THB** |

**The utilisation assumption is where this number is most fragile.** At 100% the endpoint is
saturated and the latency target is already missed; at 20% — a busy hour and a quiet night —
the cost per prediction quadruples, because an idle replica is billed at the idle rate and
still occupies memory. The honest sentence is: this service costs about **0.16 THB per 1,000
predictions** while it is provisioned to meet the latency target at realistic load.

**When is batch cheaper than keeping this endpoint warm?** Three warm replicas cost about
$7.8 a day whatever the traffic. Scoring in batches of 100 costs about $5.7 × 10⁻⁹ per
prediction, so the batch path is cheaper up to roughly **1.4 billion predictions a day**. In
other words: at any volume this course will ever see, batch is cheaper, and the online
endpoint buys latency, not economy. The endpoint is worth its price only when a person or a
system is waiting for the answer.

## Canary and rollback (Task 4)

| | |
|---|---|
| Baseline revision | `itcs355-predict--base1`, model version 1 (val ROC-AUC 0.8405) |
| Canary revision | `itcs355-predict--canary2`, model version 2 (val ROC-AUC 0.8261 — worse by a small margin, not broken) |
| Split | 90 / 10 at 14:43:34 UTC |
| Traffic actually observed | 11.2%, 11.4%, … 10.4% of responses per minute carried `x-model-version: 2` |
| Detection | **13.0 minutes** after the split, at 3.0 sigma, 23,262 responses |
| Rollback | 14:57:00 UTC, traffic back to 100% baseline; the next response carried `x-model-version: 1` |
| Evidence | `reports/lab3/canary-log.txt` (timestamps, traffic before/after), `reports/lab3/canary-raw.jsonl` (every response with its version, probability, label and latency) |

**The rule, declared before the canary was deployed** (`scripts/canary_monitor.py`): score every
response against the row's known label with the Brier score, group responses by the
`x-model-version` header relabelled A and B in first-seen order, and alert when one group's
cumulative mean exceeds the other's by more than 3 standard errors of the difference, on at
least 200 responses per group, twice in a row. The monitor never reads the traffic weights and
never reads which version is new; the mapping from cohort to version is printed only at the end.

Measured: cohort A mean Brier **0.0808** over 20,925 responses, cohort B **0.0918** over 2,342.
Cohort B was version 2.

**Five lines, as the handout asks.**

1. *What metric revealed it:* the Brier score against known labels, per cohort — not latency
   and not error rate, both of which stayed flat (0% errors throughout). A worse model is not
   an unhealthy one: every probe passed for the whole 13 minutes.
2. *How long detection took:* 13.0 minutes, 23,262 requests, of which only 2,342 reached the
   canary. The difference is real but small — 0.0110 in mean Brier — so the wait is a sample
   size problem, not a monitoring one.
3. *What would have made it faster:* more traffic on the canary, a metric with a larger effect
   size (log loss separates these two models by 0.076 against 0.018 for Brier), or accepting a
   weaker rule than 3 sigma and paying for it in false alarms.
4. *What would have happened at 50/50:* the canary would have collected responses about nine
   times faster, so the same 3-sigma rule would have fired in roughly 1.5–2 minutes — at the
   cost of sending half the users to the worse model for those minutes instead of a tenth.
5. *What it cost to be careful:* 13 minutes × 10% of traffic ≈ 2,300 predictions served by a
   model whose probabilities are about 14% worse in Brier terms. That is the price of the
   evidence, and it is the number to put in front of whoever asks why the rollout was slow.
