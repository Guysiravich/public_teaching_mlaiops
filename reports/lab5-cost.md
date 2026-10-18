# Lab 5 — Cost report

Billing is Azure Cost Management, ActualCost, pulled on 18 October 2026 at 16:08 UTC, more
than 24 hours after `make teardown LAB=5` finished (17 Oct 08:51 UTC). Amounts are in USD
from billing, converted at 33.526 THB/USD, the rate the estimate used. The Lab 5 window is
the three billing days it touched: 16–18 October.

## 1. Estimate, made before running

[`reports/lab5-cost-estimate.md`](lab5-cost-estimate.md), committed in `5bb8018` before any
Lab 5 resource existed:

- **≈ 36 THB** for resources tagged `lab=5`: pipeline compute 6.3, LLM 17.4, storage 0.1,
  plus a ×1.5 debugging margin;
- **≈ 118 THB** of shared fixed cost over an assumed 10-day window: the tracking-server VM
  4.4, and the registry, IP and disks at 11.4 THB a day.

## 2. Actual, from billing filtered by tag

**Filtered by `lab=5`, billing shows 0.0422 USD = 1.41 THB**: the Azure OpenAI tokens and
nothing else. That figure is real and badly incomplete, which is the main finding of this
report (section 3).

Everything billed in the Lab 5 window, by resource:

| Resource | Meter | USD | THB | Tag it is billed under |
|---|---|---|---|---|
| Container Registry | Basic Registry Unit | 0.3887 | 13.03 | lab=1 |
| Tracking server's public IP | Standard IPv4 Static | 0.2850 | 9.55 | lab=2 |
| Tracking server's OS disk | S4 LRS | 0.1156 | 3.88 | lab=2 |
| **Pipeline cluster `l5-f2s`** | **F2s v2** | **0.0918** | **3.08** | **lab=2 (the workspace)** |
| Azure Monitor `observabilityagents/default` | Observability Agent credit | 0.0694 | 2.33 | **none** |
| Azure OpenAI `itcs355-aoai-l5` | gpt-4.1-mini input | 0.0322 | 1.08 | lab=5 |
| Pipeline cluster node disk | P10 LRS | 0.0302 | 1.01 | lab=2 (the workspace) |
| Pipeline cluster node IP | Standard IPv4 Static | 0.0225 | 0.75 | lab=2 (the workspace) |
| Azure OpenAI | gpt-4.1-mini output | 0.0100 | 0.34 | lab=5 |
| Storage account, Key Vault | operations | 0.0006 | 0.02 | lab=1 / lab=2 |
| Tracking server VM | **B2ats v2 - Free** (10.8 h in October) | 0.0000 | 0.00 | lab=2 |
| Staging container app | — (inside the monthly free grant) | 0.0000 | 0.00 | lab=5 |
| **Total, 16–18 October** | | **1.0460** | **35.07** | |

The daily totals agree: 0.5928 + 0.3362 + 0.1170 USD. **The term so far: 11.99 USD ≈ 402
THB**, under the 800 THB cap.

## 3. The gap

| | Estimated | Billed | Gap |
|---|---|---|---|
| Tagged `lab=5` | 36 THB | 1.41 THB | −96% |
| Lab 5's own work, attributed by hand (LLM + pipeline + the untagged agent) | 36 THB | 8.58 THB | −76% |
| Shared fixed cost, per day | 11.4 + VM | 8.8 + VM 0 | −23% |
| Everything in the window | — | 35.07 THB | |

**Why the tag filter misses most of it.** I tagged the compute cluster `lab=5` at creation,
as the handout advises. It did not help. **Azure ML bills a cluster's VM, disk and IP to the
workspace's resource ID**, and the workspace carries `lab=2`. So 4.84 THB of pipeline compute
is invisible to the filter, whatever the cluster's own tags say. Two more resources escaped it:

- The **Azure Monitor observability agent** (2.33 THB) carries no tag at all. Nothing in this
  repository created it; it most likely came with the Container Apps environment that Lab 5's
  staging app ran in.
- Two capstone resources were retagged `lab=5` with `student=6688067` instead of
  `itcs355-6688067`. Teardown's own tag filter found them only after the tag was corrected.

Tagging from creation is necessary but not sufficient. The bill has to be read **by resource,
not by tag**.

**Why the estimate was high:**

1. **The LLM line was 12× too high** (17.4 against 1.41). I assumed about 400 calls at 800
   input and 250 output tokens. The real step averaged **434 input and 32 output** tokens over
   about 200 calls: the answer is a 40-word JSON object, not a paragraph. The planned
   gpt-5-mini comparison never ran.
2. **The tracking-server VM costs nothing to run.** Its meter is *B2ats v2 - Free*: the
   student subscription includes those hours. Every estimate this term priced it at 0.44 THB
   an hour, and what it really costs is its disk and its static IP, running whether the VM is
   on or not. Deallocating it overnight saved nothing.
3. **The debugging margin went unspent in the cloud.** Most Lab 5 failures happened before
   submission, at no cost: SDK wiring, a regex, a quota. They were caught locally or rejected
   by the API. One cloud run stopped early, at validate.
4. **Pipeline compute was cheaper even though the instance cost more.** Azure ML refused
   `Standard_B2s`, so it ran on `Standard_F2s_v2` (0.108 against 0.058 USD/h). Seven runs, two
   of them stopping early, used about 0.85 node-hours: 3.08 THB against 6.3 estimated.

**What the estimate missed entirely:** the untagged observability agent (+2.33 THB), and that
the window was 3 billing days, not 10.

## 4. Breakdown by component, Lab 5 window (THB)

| Component | THB | What |
|---|---|---|
| Training | 3.08 | F2s v2 node-hours across seven pipeline runs |
| Pipeline (other than compute) | 1.76 | the node's disk and public IP, billed per run |
| Storage | 3.90 | the tracking server's OS disk; blob and Key Vault operations |
| Serving | 0.00 | staging Container App: scale to zero, inside the free grant |
| LLM (Part B) | 1.41 | gpt-4.1-mini tokens |
| Monitoring | 2.33 | the untagged observability agent; Log Analytics within the free tier |
| Registry | 13.03 | Container Registry Basic, a fixed daily unit |
| Network | 9.55 | the tracking server's static public IP, charged even while deallocated |
| **Total** | **35.07** | |

**75% of the window is the registry, an IP address and a disk** — 26.5 THB that would have
been spent if Lab 5 had not run at all. The lab's actual work came to 8.6 THB.

## 5. Cost per 1,000 predictions

The configuration that meets the latency SLO (Lab 3: p95 292 ms at 10 users) is **1 vCPU ×
3 replicas, 99.4 req/s**. Container Apps list price is 0.000024 USD per vCPU-second and
0.000003 per GiB-second, so 3 × (1 × 0.000024 + 2 × 0.000003) × 3600 = 0.324 USD/h =
**10.86 THB/h**. `src/costs.cost_per_1k_predictions`:

| Utilisation | THB per 1,000 |
|---|---|
| 5% | 0.607 |
| 25% | 0.121 |
| 80% | 0.038 |

**Which I believe: none of the three.** This system scores a fleet of 240 machines, once a
day: 240 predictions a day, a utilisation of about 0.0003%. A warm endpoint at 10.86 THB/h
would cost 260 THB a day, **about 1,090 THB per 1,000 predictions.** That is the Lab 3
conclusion again. At this volume, scheduled batch scoring, or a scale-to-zero endpoint, wins
at any realistic rate.

For comparison, the LLM triage step costs **7.54 THB per 1,000** whatever the utilisation
(Part B). That is 200 times the classifier's endpoint at 80%. The language model, not the
classifier, would be this system's serving bill.

## 6. One optimisation applied: scale to zero for staging

| | Before (Lab 3: `minReplicas` 1) | After (Labs 4–5 staging: `minReplicas` 0) |
|---|---|---|
| Configuration | 0.5 vCPU / 1 GiB, always one replica | the same, no replica while idle |
| List price while idle | 0.054 USD/h = 1.81 THB/h = **43.4 THB/day** | **0** |
| Billed in the Lab 5 window | — | **0.00 THB** (16–18 Oct: smoke tests, two rollouts, the CD runs) |
| THB per 1,000 at staging's traffic (≈ 50 requests a day) | ≈ 870 | ≈ 0 |
| Latency, first request | 0.22 s warm | **35.5 s cold start** (Lab 3); 35 s again on the Lab 5 rollback probe |
| Latency, p95 once warm | 66 ms (1 user) | unchanged |

**The trade, plainly.** A replica that waits costs 43 THB a day at list price. The free grant
hid that cost for about four days a month, and would have stopped hiding it after that.
Removing the waiting replica costs the first caller after a quiet spell 35 seconds. For
staging, where the caller is a smoke test that already waits for readiness (`wait_ready`),
that is free. For production it is not: one 35-second answer breaks a 300 ms SLO on its own.
So production keeps `minReplicas` 1, and this optimisation applies to staging only.

**Not an optimisation, recorded because it looked like one:** deallocating the tracking
server overnight. Its compute was already free, and its disk and static IP bill either way.
The saving that would be real is moving it to a dynamic IP: 9.55 THB in this window, a third
of the total.
