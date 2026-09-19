# Lab 2 — Run comparison

Experiment `itcs355-lab2` · 12 trials · total spend 0.0296 THB

`thb_per_point` is cost per percentage point of val_roc_auc above the worst trial. Cheap improvements rank low; expensive improvements rank high, however good the headline number is.

| run_id   |   val_roc_auc |   cost_thb |   n_estimators |   max_depth |   min_samples_leaf | class_weight   | max_features   |   thb_per_point |
|:---------|--------------:|-----------:|---------------:|------------:|-------------------:|:---------------|:---------------|----------------:|
| a5192935 |        0.8405 |     0.0016 |            200 |           4 |                  5 | None           | sqrt           |          0.0006 |
| f9e0834d |        0.8398 |     0.0014 |            200 |           4 |                  5 | balanced       | sqrt           |          0.0006 |
| d46b03e6 |        0.8394 |     0.0017 |            200 |           8 |                  5 | balanced       | sqrt           |          0.0007 |
| 52f06450 |        0.8364 |     0.0022 |            200 |           8 |                  5 | None           | sqrt           |          0.001  |
| 1728ce57 |        0.8352 |     0.0019 |            200 |          12 |                  5 | None           | sqrt           |          0.0009 |
| cbc73c3a |        0.8317 |     0.0023 |            200 |           4 |                  5 | None           | 1.0            |          0.0014 |
| c5ca9a68 |        0.8307 |     0.0023 |            200 |           4 |                  5 | balanced       | 1.0            |          0.0015 |
| 251b52cb |        0.8261 |     0.0019 |            200 |          12 |                  5 | balanced       | sqrt           |          0.0017 |
| c3d560f8 |        0.8244 |     0.0033 |            200 |           8 |                  5 | balanced       | 1.0            |          0.0035 |
| 226b9b6d |        0.8239 |     0.0036 |            200 |           8 |                  5 | None           | 1.0            |          0.004  |
| b7d45350 |        0.8192 |     0.0039 |            200 |          12 |                  5 | None           | 1.0            |          0.0093 |
| 2f377acc |        0.815  |     0.0035 |            200 |          12 |                  5 | balanced       | 1.0            |      35000      |

## Which model did you register, and why?

**Registered: run `a5192935`** (max_depth 4, class_weight None, max_features sqrt).

The top three trials sit within 0.0011 val ROC-AUC; refitting each with five model seeds gives std
0.0013–0.0020 (this config: 0.8408 ± 0.0013), so their ranking is noise. I chose between the two
depth-4 leaders on how the output would be used. Maintenance planners would read p(failure) as a
probability: inspect when p × breakdown cost exceeds inspection cost. Both rank machines
identically (inspecting the top 100 catches 44 of 121 failures either way), but
class_weight=balanced inflates probabilities threefold: mean predicted 33% against an actual 10.1%,
Brier 0.149 versus 0.075. This model predicts 10.9%. More recall comes from lowering the threshold,
not reweighting. It is also the shallowest, simplest candidate.

Cost: each trial's fit costs about 0.002 THB, but a job pays for node start-up; a cold-start study
costs about 1.2 THB. Monthly retraining is about 1.6 THB including an hour of tracking server, under
1% of the standing costs (registry, IP, disk: ~340 THB/month).

Could be wrong: deployed with the default 0.5 threshold, it flags 13 machines and catches 5% of
failures; and if the failure rate drifts, calibration drifts with it.

*Evidence: seed variance from `make seeds-remote` (README, Seed variance); ranking and calibration
from `python scripts/calibration_check.py`. `make compare` rewrites this file; this section was
added after the last run.*
