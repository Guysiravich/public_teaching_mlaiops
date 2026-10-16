# Lab 5 — cost estimate, made before running anything

Committed on 16 October 2026, before Task 1 (the handout requires the estimate to predate the
work). `reports/lab5-cost.md` compares billing against this file; the git history shows which
came first.

**Prices** checked on 16 October 2026 at prices.azure.com (Linux, pay-as-you-go), converted at
**33.526 THB/USD** (open.er-api.com, same day — the source Lab 2 used).

| Item | List price | THB |
|---|---|---|
| `Standard_B2s` VM, East Asia (pipeline compute) | 0.0584 USD/h | 1.96 /h |
| `Standard_B2ats_v2` VM (MLflow tracking server) | 0.0131 USD/h | 0.44 /h |
| gpt-4.1-mini, Global, Southeast Asia — input | 0.40 USD / 1M tokens | 13.4 / 1M |
| gpt-4.1-mini — cached input | 0.10 USD / 1M tokens | 3.4 / 1M |
| gpt-4.1-mini — output | 1.60 USD / 1M tokens | 53.6 / 1M |
| Container Registry, Basic | 0.1666 USD/day | 5.59 /day |

## What is estimated

Everything created for Lab 5 carries the tag `lab=5`, so billing filtered by that tag is the
"actual" this estimate is judged against. The shared resources from earlier labs keep their own
tags and are estimated separately below, because a cost report that left them out would
understate what this lab really costs to run.

### Tagged `lab=5` — the figure to compare with billing

| Component | Assumption | THB |
|---|---|---|
| Pipeline compute (training, validate, evaluate) | new `Standard_B2s` cluster, 0–1 nodes, as the handout's student note advises (2 vCPU fits the 3-vCPU cap). One run ≈ 4 min node start + 5 steps × 2 min + 2 min idle ≈ 16 min ≈ 0.52 THB. 12 runs: 6 while building, 1 failing-gate demonstration, 2 for the permission-removal test, 3 scheduled | 6.3 |
| Serving (staging) | Container Apps, 0.5 vCPU, scale to zero; smoke tests only; inside the monthly free grant (Lab 3: a full load test used 2% of it) | 0.0 |
| LLM step, Part B | gpt-4.1-mini: about 400 calls × (800 input + 250 output tokens) = 320k in + 100k out → 4.3 + 5.4. A smaller comparison with gpt-5-mini, about 100 calls whose hidden reasoning tokens bill as output (≈ 1,000 out each) → 7.7 | 17.4 |
| Monitoring | Log Analytics, well under the monthly free ingestion | 0.0 |
| Storage | pipeline artifacts and data, under 1 GB for 10 days | 0.1 |
| **Subtotal** | | **23.8** |
| Debugging margin, ×1.5 | Lab 4's first CD run failed three ways before it worked; Lab 2's first job failed twice on permissions. Assume the same here | 11.9 |
| **Estimate, tagged `lab=5`** | | **≈ 36 THB** |

### Shared resources, not tagged `lab=5`, over a 10-day Lab 5 window (16–26 October)

| Component | Assumption | THB |
|---|---|---|
| MLflow tracking server VM | on about 10 hours in total; deallocated otherwise | 4.4 |
| Container Registry, public IP, storage and disk | fixed daily cost, measured from October billing: 11.4 THB/day × 10 days | 114 |
| **Shared, estimated** | | **≈ 118 THB** |

### Second provider (Task 4)

0 THB. No AWS or GCP account with billing exists; the plan is a free route (an education lab
account if the course has one, otherwise an emulator), recorded as a deviation in the write-up.

## The term budget

Actual spend so far from Cost Management: **11.06 USD ≈ 371 THB** (September 5.72, October
1–16 5.34). With this estimate the term reaches about **525 THB**, under the 800 THB cap — **if
the shared resources are deleted when Lab 5 is torn down.** Left running to the end of
November, the fixed 11.4 THB/day alone would add about 400 THB and break the cap. Task 6's
teardown therefore covers the shared resources too, once Lab 5 is graded.

## Where this estimate is most likely to be wrong

- **Debugging time on pipeline compute.** A step that fails after the node has started still
  pays for the start; ten failed attempts cost as much as the whole planned set.
- **LLM output length.** Output is four times the input price. If answers run to 600 tokens
  instead of 250, the gpt-4.1-mini line more than doubles — which is exactly the lever Task 9
  measures.
- **Untagged spend.** Anything created without `lab=5` (a disk, an IP, a default workspace
  created by a CLI command) lands in "shared" or nowhere, and the tag filter will not see it.
