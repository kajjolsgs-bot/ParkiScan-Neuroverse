# ParkiScanML — training report

Generated 2026-10-12T07:07:55.232994+00:00

> Research benchmark on a small public dataset with participant-grouped cross-validation. Not clinically validated; not for diagnosis or screening decisions.

## oxford_voice

- Task: classification — PD (1) vs healthy (0)
- Unit: sustained vowel recording
- Rows: 195, participants: 32 (24 positive)
- CV: StratifiedGroupKFold x5 repeats (participant-grouped)
- Best non-baseline model: **neural_net_mlp**
- Usable for research scores: **yes**
- ⚠ SMALL_SAMPLE: only 32 participants
- ⚠ TINY_TEST_SET: independent test has only 8 participants; its metrics are unstable

Metrics at **participant level** (pooled out-of-fold):

| model | ROC AUC | 95% CI (participant bootstrap) | AUC mean±sd over repeats | balanced acc | sens | spec |
|---|---|---|---|---|---|---|
| dummy_most_frequent | 0.406 | 0.189–0.622 | 0.406±0.001 | 0.500 | 1.000 | 0.000 |
| logistic_regression | 0.724 | 0.469–0.942 | 0.740±0.026 | 0.688 | 0.875 | 0.500 |
| random_forest | 0.677 | 0.460–0.867 | 0.744±0.022 | 0.542 | 0.833 | 0.250 |
| hist_gradient_boosting | 0.625 | 0.359–0.891 | 0.678±0.037 | 0.667 | 0.958 | 0.375 |
| neural_net_mlp | 0.828 | 0.647–0.971 | 0.777±0.047 | 0.646 | 0.917 | 0.375 |

**Independent test** — independent held-out participants (1/4, stratified & grouped, seed 42); model + Youden threshold chosen on training participants only; model **random_forest**

| n | threshold | TP | FP | TN | FN | sensitivity (95% CI) | specificity (95% CI) | precision | F1 | ROC AUC | indeterminate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.765 | 3 | 0 | 2 | 3 | 0.500 (0.188–0.812) | 1.000 (0.342–1.000) | 1.000 | 0.667 | 1.000 | 0.12 |

## telemonitoring

- Task: regression — motor UPDRS (all participants have early PD)
- Unit: home voice recording
- Rows: 5875, participants: 42
- CV: GroupKFold(5) (participant-grouped)
- Best non-baseline model: **ridge**
- Usable for research scores: **no**
- ⚠ NO_BETTER_THAN_BASELINE: best model does not beat predicting the training mean
- Note: age/sex/test_time removed so the model cannot shortcut via demographics or time

| model | MAE | R² |
|---|---|---|
| dummy_mean | 7.15 | -0.042 |
| ridge | 7.36 | -0.133 |
| random_forest | 7.59 | -0.251 |
| hist_gradient_boosting | 7.61 | -0.233 |
| neural_net_mlp | 8.03 | -0.587 |

## speech_multi

- Task: classification — PD (1) vs healthy (0)
- Unit: one of 26 speech recordings per participant
- Rows: 1040, participants: 40 (20 positive)
- CV: StratifiedGroupKFold x5 repeats (participant-grouped)
- Best non-baseline model: **neural_net_mlp**
- Usable for research scores: **no**
- ⚠ SMALL_SAMPLE: only 40 participants
- ⚠ INDEPENDENT_TEST_FAILED: held-out ROC AUC 0.44 — cross-validation result did not generalise
- ⚠ TINY_TEST_SET: independent test has only 10 participants; its metrics are unstable
- Note: UPDRS column removed (it is only defined for PD and would leak the label)

Metrics at **participant level** (pooled out-of-fold):

| model | ROC AUC | 95% CI (participant bootstrap) | AUC mean±sd over repeats | balanced acc | sens | spec |
|---|---|---|---|---|---|---|
| dummy_most_frequent | 0.500 | 0.500–0.500 | 0.500±0.000 | 0.500 | 1.000 | 0.000 |
| logistic_regression | 0.670 | 0.486–0.821 | 0.565±0.047 | 0.525 | 0.550 | 0.500 |
| random_forest | 0.630 | 0.448–0.800 | 0.547±0.032 | 0.625 | 0.600 | 0.650 |
| hist_gradient_boosting | 0.648 | 0.470–0.807 | 0.552±0.027 | 0.600 | 0.550 | 0.650 |
| neural_net_mlp | 0.702 | 0.529–0.855 | 0.598±0.019 | 0.750 | 0.800 | 0.700 |

**Independent test** — independent held-out participants (1/4, stratified & grouped, seed 42); model + Youden threshold chosen on training participants only; model **neural_net_mlp**

| n | threshold | TP | FP | TN | FN | sensitivity (95% CI) | specificity (95% CI) | precision | F1 | ROC AUC | indeterminate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.362 | 5 | 4 | 1 | 0 | 1.000 (0.566–1.000) | 0.200 (0.036–0.624) | 0.556 | 0.714 | 0.440 | 0.20 |

## spiral

- Task: classification — PD (1) vs control (0)
- Unit: spiral test 0, one file per participant
- Rows: 40, participants: 40 (25 positive)
- CV: StratifiedGroupKFold x5 repeats (participant-grouped)
- Best non-baseline model: **random_forest**
- Usable for research scores: **yes**
- ⚠ SMALL_SAMPLE: only 40 participants
- ⚠ TINY_TEST_SET: independent test has only 10 participants; its metrics are unstable
- Note: features computed by parkiscan.features.spiral (same code used for in-game traces)

Metrics at **recording level** (pooled out-of-fold):

| model | ROC AUC | 95% CI (participant bootstrap) | AUC mean±sd over repeats | balanced acc | sens | spec |
|---|---|---|---|---|---|---|
| dummy_most_frequent | 0.500 | 0.500–0.500 | 0.500±0.000 | 0.500 | 1.000 | 0.000 |
| logistic_regression | 0.744 | 0.573–0.883 | 0.748±0.019 | 0.667 | 0.600 | 0.733 |
| random_forest | 0.829 | 0.682–0.945 | 0.843±0.014 | 0.747 | 0.760 | 0.733 |
| hist_gradient_boosting | 0.757 | 0.586–0.898 | 0.782±0.014 | 0.700 | 0.800 | 0.600 |
| neural_net_mlp | 0.723 | 0.542–0.872 | 0.780±0.039 | 0.680 | 0.760 | 0.600 |

**Independent test** — independent held-out participants (1/4, stratified & grouped, seed 42); model + Youden threshold chosen on training participants only; model **random_forest**

| n | threshold | TP | FP | TN | FN | sensitivity (95% CI) | specificity (95% CI) | precision | F1 | ROC AUC | indeterminate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.529 | 5 | 0 | 4 | 1 | 0.833 (0.436–0.970) | 1.000 (0.510–1.000) | 1.000 | 0.909 | 1.000 | 0.10 |

## gaitndd

- Task: classification — PD (1) vs control (0)
- Unit: one ~5 min walk per participant (force-sensitive insoles)
- Rows: 31, participants: 31 (15 positive)
- CV: StratifiedGroupKFold x5 repeats (participant-grouped)
- Best non-baseline model: **random_forest**
- Usable for research scores: **yes**
- ⚠ SMALL_SAMPLE: only 31 participants
- ⚠ TINY_TEST_SET: independent test has only 8 participants; its metrics are unstable

Metrics at **recording level** (pooled out-of-fold):

| model | ROC AUC | 95% CI (participant bootstrap) | AUC mean±sd over repeats | balanced acc | sens | spec |
|---|---|---|---|---|---|---|
| dummy_most_frequent | 0.475 | 0.331–0.636 | 0.475±0.000 | 0.475 | 0.200 | 0.750 |
| logistic_regression | 0.850 | 0.671–1.000 | 0.858±0.012 | 0.869 | 0.800 | 0.938 |
| random_forest | 0.900 | 0.754–1.000 | 0.888±0.013 | 0.838 | 0.800 | 0.875 |
| hist_gradient_boosting | 0.892 | 0.729–1.000 | 0.893±0.016 | 0.902 | 0.867 | 0.938 |
| neural_net_mlp | 0.867 | 0.709–1.000 | 0.843±0.018 | 0.902 | 0.867 | 0.938 |

**Independent test** — independent held-out participants (1/4, stratified & grouped, seed 42); model + Youden threshold chosen on training participants only; model **logistic_regression**

| n | threshold | TP | FP | TN | FN | sensitivity (95% CI) | specificity (95% CI) | precision | F1 | ROC AUC | indeterminate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.652 | 4 | 0 | 4 | 0 | 1.000 (0.510–1.000) | 1.000 (0.510–1.000) | 1.000 | 1.000 | 1.000 | 0.00 |

## daphnet

- Task: classification (event detection) — freezing-of-gait window (1) vs walking/standing (0)
- Unit: 4.0s window, 1.0s hop, ankle accelerometer
- Rows: 17720, participants: 10 (8 positive)
- CV: leave-one-subject-out (participant-grouped)
- Best non-baseline model: **hist_gradient_boosting**
- Usable for research scores: **yes**
- ⚠ SMALL_SAMPLE: only 10 participants
- Note: detects FoG episodes in people already diagnosed; not a diagnostic model

Metrics at **recording level** (pooled out-of-fold):

| model | ROC AUC | 95% CI (participant bootstrap) | AUC mean±sd over repeats | balanced acc | sens | spec |
|---|---|---|---|---|---|---|
| dummy_most_frequent | 0.237 | 0.149–0.355 | 0.237±0.000 | 0.500 | 0.000 | 1.000 |
| logistic_regression | 0.850 | 0.805–0.901 | 0.850±0.000 | 0.778 | 0.729 | 0.827 |
| random_forest | 0.867 | 0.829–0.904 | 0.867±0.000 | 0.721 | 0.512 | 0.929 |
| hist_gradient_boosting | 0.884 | 0.846–0.919 | 0.884±0.000 | 0.797 | 0.759 | 0.834 |
| neural_net_mlp | 0.828 | 0.764–0.894 | 0.828±0.000 | 0.649 | 0.337 | 0.960 |

**Independent test** — leave-one-subject-out; threshold tuned per fold on the other subjects (inner grouped CV)

| n | threshold | TP | FP | TN | FN | sensitivity (95% CI) | specificity (95% CI) | precision | F1 | ROC AUC | indeterminate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 17720 | — | 1433 | 3631 | 12391 | 265 | 0.844 (0.826–0.860) | 0.773 (0.767–0.780) | 0.283 | 0.424 | 0.884 | — |

## etpdvis

- Task: classification — PD (1) vs healthy control (0)
- Unit: one participant = 6 visualisation images (TMT A/B/C x heatmap + scanpath)
- Rows: 40, participants: 40 (28 positive)
- CV: StratifiedGroupKFold x5 repeats (participant-grouped)
- Best non-baseline model: **logistic_regression**
- Usable for research scores: **no**
- ⚠ SUSPICIOUSLY_HIGH: near-perfect separation on a tiny sample usually means a confound or label leakage (recording setup, rendering, task duration); do not rely on this model
- ⚠ SMALL_SAMPLE: only 40 participants
- ⚠ TINY_TEST_SET: independent test has only 10 participants; its metrics are unstable
- Note: dataset ships rendered images, not raw gaze samples; image statistics are a weak proxy and the sample is very small (12 HC / 28 PD). Ink coverage mostly tracks how long the participant looked (task duration), so high scores here are not evidence of a gaze biomarker

Metrics at **recording level** (pooled out-of-fold):

| model | ROC AUC | 95% CI (participant bootstrap) | AUC mean±sd over repeats | balanced acc | sens | spec |
|---|---|---|---|---|---|---|
| dummy_most_frequent | 0.429 | 0.260–0.593 | 0.429±0.000 | 0.500 | 1.000 | 0.000 |
| logistic_regression | 1.000 | 1.000–1.000 | 1.000±0.000 | 1.000 | 1.000 | 1.000 |
| random_forest | 0.997 | 0.980–1.000 | 0.997±0.005 | 0.958 | 1.000 | 0.917 |
| hist_gradient_boosting | 0.979 | 0.924–1.000 | 0.950±0.027 | 0.899 | 0.964 | 0.833 |
| neural_net_mlp | 1.000 | 1.000–1.000 | 0.995±0.007 | 1.000 | 1.000 | 1.000 |

**Independent test** — independent held-out participants (1/4, stratified & grouped, seed 42); model + Youden threshold chosen on training participants only; model **logistic_regression**

| n | threshold | TP | FP | TN | FN | sensitivity (95% CI) | specificity (95% CI) | precision | F1 | ROC AUC | indeterminate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.704 | 7 | 0 | 3 | 0 | 1.000 (0.646–1.000) | 1.000 (0.438–1.000) | 1.000 | 1.000 | 1.000 | 0.00 |

## Deep learning — 1-D CNN on raw accelerometer windows (Daphnet freezing of gait)

- Model: 1-D CNN (NumPy): Conv(3→16,k9)-ReLU-MaxPool4-Conv(16→32,k7)-ReLU-GAP-Dense(2)
- Input: raw ankle acceleration, 3 × 256 samples (4 s @ 64 Hz), per-window mean removed
- Protocol: leave-one-subject-out; decision threshold tuned on a held-out TRAINING subject

| windows | TP | FP | TN | FN | sensitivity (95% CI) | specificity (95% CI) | precision | F1 | ROC AUC |
|---|---|---|---|---|---|---|---|---|---|
| 17720 | 1197 | 4375 | 11647 | 501 | 0.705 (0.683–0.726) | 0.727 (0.720–0.734) | 0.215 | 0.329 | 0.850 |

Per-subject ROC AUC: S01 0.95, S02 0.83, S03 0.90, S05 0.82, S06 0.92, S07 0.94, S08 0.76, S09 0.91

## Differential gait sub-stratification (control / Parkinson's / Huntington's / ALS; research)

- Participants: 64 (control 16, parkinsons 15, huntingtons 20, als 13); CV: stratified 5-fold (one recording per participant); best: **logistic_regression**

| model | macro F1 | balanced acc | macro ROC AUC (OvR) | recall control / PD / HD / ALS |
|---|---|---|---|---|
| dummy_most_frequent | 0.119 | 0.250 | 0.473 | 0.00 / 0.00 / 1.00 / 0.00 |
| logistic_regression | 0.663 | 0.678 | 0.858 | 0.88 / 0.47 / 0.60 / 0.77 |
| random_forest | 0.556 | 0.566 | 0.824 | 0.75 / 0.40 / 0.50 / 0.62 |
| hist_gradient_boosting | 0.530 | 0.530 | 0.803 | 0.81 / 0.40 / 0.60 / 0.31 |
| neural_net_mlp | 0.651 | 0.656 | 0.878 | 0.75 / 0.53 / 0.65 / 0.69 |

Confusion matrix of the best model (rows = true, columns = predicted; control, PD, HD, ALS):

    [14, 1, 0, 1]
    [2, 7, 3, 3]
    [3, 3, 12, 2]
    [1, 2, 0, 10]

## Severity sub-stratification (people with PD only; research)

### gaitndd:hoehn_yahr
- Hoehn & Yahr stage (PD participants only) — one walk per participant; participants: 15; best: **hist_gradient_boosting** → BEATS_BASELINE

| model | MAE | Spearman ρ |
|---|---|---|
| dummy_mean | 0.76 | -1.000 |
| ridge | 0.80 | 0.691 |
| random_forest | 0.63 | 0.455 |
| hist_gradient_boosting | 0.48 | 0.607 |
| neural_net_mlp | 1.77 | 0.249 |

### gaitndd:early_vs_advanced
- H&Y >= 3 (moderate-advanced) vs <= 2.5 (early) — one walk per participant; participants: 15; best: **logistic_regression** → ABOVE_CHANCE

| model | ROC AUC |
|---|---|
| dummy_most_frequent | 0.000 |
| logistic_regression | 0.870 |
| random_forest | 0.741 |
| hist_gradient_boosting | 0.833 |
| neural_net_mlp | 0.611 |

### telemonitoring:motor_updrs_tier
- motor UPDRS >= 20 vs < 20 (early-PD cohort) — home voice recording; participants: 42; best: **logistic_regression** → CHANCE_LEVEL

| model | ROC AUC |
|---|---|
| dummy_most_frequent | 0.589 |
| logistic_regression | 0.602 |
| random_forest | 0.530 |
| hist_gradient_boosting | 0.458 |
| neural_net_mlp | 0.569 |

### speech_multi:updrs
- UPDRS (PD participants only) — speech recording; participants: 20; best: **random_forest** → NO_BETTER_THAN_BASELINE

| model | MAE | Spearman ρ |
|---|---|---|
| dummy_mean | 13.47 | -0.513 |
| ridge | 15.87 | -0.386 |
| random_forest | 15.31 | -0.320 |
| hist_gradient_boosting | 15.69 | -0.262 |
| neural_net_mlp | 17.77 | -0.254 |
