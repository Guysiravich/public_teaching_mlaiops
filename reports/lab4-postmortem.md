# Post-mortem — temp_c read 6 °C high on the staging endpoint (Lab 4, Task 6)

Injected on purpose: `make inject-drift` (`--feature temp_c --mode shift --magnitude 6`), sent to
`itcs355-staging` by `scripts/send_traffic.py` from 13:50:23 UTC on 6 October 2026. Evidence in
`reports/lab4/`: `alerts.json`, `drift-metrics.json`, `dashboard-shift.png`, `prometheus.csv`, and
the request log `reports/lab4-traffic.log`.

**What fired:**
Azure Monitor alert `itcs355-drift-alert` (Sev2) at **14:22:39 UTC** — fired, but the email
receiver had not completed Azure's new verification, so it reached nobody. Once verified, the same
check on the same inputs fired again and the email "Activated Severity: 2 itcs355-drift-alert"
arrived at 15:32 (README, Lab 4, failure 4). The first firing:
`drift_threshold_ratio` 3.505 ≥ 1, written by the 14:15 scheduled drift run — temp_c PSI 0.3505
against its 0.10 threshold over the last 500 inputs; every other feature under its limit (load_pct
0.17 of 0.30). The run before the injection scored 0.57. Injection to alert: **32 min 16 s**.

**True cause:**
A **broken input, not a changed world**: every reading of temp_c arrived 6 °C high while the
machines, and so the failure labels, were unchanged — a sensor offset (calibration or firmware
fault at the producer). Schema and null rate were intact, so the contract tests passed and only the
distribution check could see it. It looks exactly like data drift on the dashboard (rolling mean
79.6 → 85.5 °C in about 3 minutes, standard deviation flat), which is why the cause has to be
established from outside the model: no change of season, workload or fleet explains 6 °C on every
machine at the same minute.

**Retrain, roll back, or no action — and why:**
**Neither retrain nor roll back the model; fix the producer and quarantine the readings.** Rolling
back is pointless: the model did not change and the previous version would read the same wrong
numbers. Retraining is harmful: it would teach the model that 6 °C hotter is normal, bake the
fault into the next registered version and leave no clean model to return to once the sensor is
fixed. The response is to stop trusting temp_c from the affected source (hold its predictions for
manual review), get the sensor recalibrated, and drop the affected window from any future training
set. What would change my mind: technicians confirming the machines really run hotter — a new duty
cycle, a hot season — and failure labels moving with it over the following weeks. Then it is real
data drift, and retraining on recent, verified data is right.

**What this would have cost if unnoticed for a week:**
Measured on the held-out machines with the 6 °C offset (labels unchanged): the share of readings
flagged as "will fail within 7 days" rises from 3.75% to 4.38% (90 → 105 per 2,400 readings), and
8 of the 15 extra flags are machines that do not fail. Assuming the 240-machine fleet is scored once
a day (1,680 readings a week) and an unnecessary technician visit costs about 2,000 THB, that is
about 11 extra dispatches and roughly 6 wasted ones — **about 12,000 THB a week**, before the larger
cost: anyone who retrained on that week's data would have shipped the offset into the model.

**How to prevent or detect it faster:**
**Alert on the detector's own heartbeat**: a metric alert that fires when `drift_window_rows` has
not been written for 20 minutes. Half of this incident's 32 minutes was one scheduled run that
"completed" in zero seconds because Azure ML reused the previous run's cached output and never read
the new inputs (fixed: `force_rerun` in `cloudlayer/azure.py`). A detector that silently stops
detecting is the failure no drift threshold can catch; with the heartbeat and the fix, the 14:00
run would have alerted at about 14:07, roughly 17 minutes after injection.
