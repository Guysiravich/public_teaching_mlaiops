# Drift threshold study

Reference: training split, 3,600 rows. Held-out pool: validation + test, 2,400 rows from machines never seen in training.

## 1. Noise: largest PSI over the six features, no drift

1,000 windows per size, drawn without replacement from the held-out pool.

| Window (rows) | median | p99 | max | windows above 0.10 | above 0.15 |
|---|---|---|---|---|---|
| 250 | 0.155 | 0.278 | 0.322 | 91.1% | 55.1% |
| 500 | 0.144 | 0.213 | 0.252 | 95.8% | 41.4% |
| 1000 | 0.136 | 0.179 | 0.194 | 98.8% | 22.7% |

Per feature, p99 of PSI with no drift:

| Feature | 250 rows | 500 rows | 1000 rows |
|---|---|---|---|
| temp_c | 0.101 | 0.057 | 0.033 |
| vibration_mm_s | 0.079 | 0.038 | 0.018 |
| pressure_kpa | 0.079 | 0.039 | 0.020 |
| hours_since_service | 0.085 | 0.052 | 0.027 |
| load_pct | 0.278 | 0.213 | 0.179 |
| ambient_humidity | 0.088 | 0.042 | 0.019 |

Whole held-out pool against the training reference (no sampling noise left; what remains is the difference between machines):

| temp_c | vibration_mm_s | pressure_kpa | hours_since_service | load_pct | ambient_humidity |
|---|---|---|---|---|---|
| 0.012 | 0.002 | 0.003 | 0.009 | 0.134 | 0.004 |

## 2. Harm: a `temp_c` sensor fault, labels unchanged

Unshifted held-out Brier 0.0772; share flagged (p >= 0.5) 3.8%. PSI is the mean over 200 windows of 500 rows.

| Fault | PSI (500 rows) | Brier | Brier change | Share flagged |
|---|---|---|---|---|
| offset +0.5 °C | 0.029 | 0.0772 | +0.0% | 3.8% |
| offset +1 °C | 0.040 | 0.0773 | +0.2% | 3.9% |
| offset +2 °C | 0.064 | 0.0776 | +0.5% | 4.0% |
| offset +3 °C | 0.116 | 0.0779 | +0.9% | 4.0% |
| offset +4 °C | 0.185 | 0.0781 | +1.2% | 4.2% |
| offset +6 °C | 0.411 | 0.0782 | +1.3% | 4.4% |
| spread x1.1 | 0.025 | 0.0772 | +0.0% | 3.8% |
| spread x1.25 | 0.061 | 0.0776 | +0.5% | 3.9% |
| spread x1.5 | 0.172 | 0.0778 | +0.8% | 4.2% |
| spread x2 | 0.451 | 0.0786 | +1.8% | 5.1% |

## 3. Harm per feature: an offset of one training standard deviation

| Feature | 1 SD | PSI (500 rows) | Brier change |
|---|---|---|---|
| temp_c | 10.15 | 1.079 | +4.3% |
| vibration_mm_s | 1.51 | 0.995 | +22.7% |
| pressure_kpa | 14.25 | 0.936 | +0.8% |
| hours_since_service | 2609.14 | 1.367 | +3.9% |
| load_pct | 21.76 | 1.146 | +1.1% |
| ambient_humidity | 12.05 | 0.925 | -0.4% |
