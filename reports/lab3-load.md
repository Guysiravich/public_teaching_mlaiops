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

This section, and the threshold in `loadtest/k6.js`, were committed before the endpoint
existed. Every table below was filled in afterwards.

## Results

_Not measured yet._

| Concurrency | Throughput (req/s) | p50 | p95 | p99 | Error rate |
|---|---|---|---|---|---|
| 1 | | | | | |
| 10 | | | | | |
| 50 | | | | | |

### Breaking point

_Not measured yet._

### Cold start

_Not measured yet._

### Batch against single calls

_Not measured yet._

### Payload size

_Not measured yet._

### One instance size up

_Not measured yet._

### Cost per 1,000 predictions

_Not measured yet._
