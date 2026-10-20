# Freezing-of-gait forecasting — Daphnet, leave-one-subject-out

Data: Daphnet Freezing of Gait (10 participants with PD, 64 Hz, 3 accelerometers). 35405 windows (4 s, 0.5 s hop, causal features). Validation: leave-one-subject-out (10 folds).

Target: a freeze **starts within H seconds** of the current moment. Windows that already contain freezing are
excluded (that is detection, reported separately in `training_report.md`).

## Window-level ROC AUC (pooled out-of-fold, 95% CI by subject bootstrap)

| horizon | freeze-index baseline | ankle_imu | ankle_imu+ehr | all_imu | all_imu+ehr |
|---|---|---|---|---|---|
| 2 s | 0.63 | 0.76 (0.71–0.83) | 0.74 (0.62–0.85) | – | – |
| 3 s | 0.60 | 0.76 (0.72–0.83) | 0.75 (0.65–0.85) | 0.79 (0.75–0.85) | 0.76 (0.68–0.84) |
| 5 s | 0.56 | 0.73 (0.70–0.79) | 0.70 (0.59–0.80) | – | – |

## Deployed configuration: `ankle_imu+ehr`, horizon 3 s

Alarm rule: risk ≥ fold threshold (training-subject specificity 0.85) for 2 consecutive windows.

* Window level: sensitivity 0.52, specificity 0.84 (TP 543, FP 4573, TN 24589, FN 499).
* Event level: **148 of 237 freezing episodes (62%) were preceded by an alarm in the 5 s before onset**; median lead time 4.6 s (IQR 3.4–4.8 s).
* False alarms: 139.3 per hour of non-freezing walking (4.44 h).

| subject | episodes | warned | median lead (s) | false alarms | hours |
|---|---|---|---|---|---|
| S01 | 23 | 18 | 4.2 | 124 | 0.50 |
| S02 | 24 | 20 | 4.6 | 59 | 0.34 |
| S03 | 49 | 0 | – | 0 | 0.47 |
| S04 | 0 | 0 | – | 95 | 0.57 |
| S05 | 66 | 64 | 4.7 | 77 | 0.44 |
| S06 | 10 | 7 | 3.5 | 86 | 0.51 |
| S07 | 24 | 2 | 1.5 | 0 | 0.42 |
| S08 | 14 | 12 | 4.7 | 47 | 0.15 |
| S09 | 27 | 25 | 4.7 | 54 | 0.40 |
| S10 | 0 | 0 | – | 76 | 0.62 |

### As a wearable cue (10 s refractory period)

A cue (vibration / rhythmic beat) lasts several seconds, so an alarm that re-triggers while a cue is active is
not a new alarm. Rule: per-fold threshold (training-subject specificity 0.85), 2 consecutive windows, 10 s cue refractory period.

* **148 of 237 freezes (62%) cued before onset**, median lead 4.6 s; 96 cues per hour of non-freezing walking.
* For context, this lab protocol provoked about 53 freezes per hour of walking.

### Operating curve (pooled threshold, descriptive)

Each row uses one threshold for everyone; a clinician picks the trade-off. Out-of-fold scores only.

| window specificity | freezes warned | false cues / h | median lead (s) |
|---|---|---|---|
| 0.85 | 61% (144/237) | 94 | 4.6 |
| 0.90 | 54% (128/237) | 73 | 4.3 |
| 0.95 | 43% (103/237) | 40 | 3.8 |
| 0.97 | 35% (83/237) | 26 | 3.2 |
| 0.99 | 19% (44/237) | 9 | 2.7 |

## Limitations

* S03 (H&Y 2.5, OFF) received no warnings at the deployed threshold and S07 only 2 of 24 — their pre-freeze
  signal differs from the other participants; per-patient calibration in the twin is meant to address this.
* 10 participants, recorded in one laboratory protocol designed to provoke freezing; prevalence of freezing is
  far higher than in daily life, so false-alarm rates at home will differ.
* S04 and S10 had no freezing episodes; they contribute only to specificity / false alarms.
* Clinical header (age, sex, disease duration, H&Y, medication state) is real but has only 10 distinct values,
  so its contribution cannot be estimated precisely.
* Research prototype, not a medical device. Bächlin M, Plotnik M, Roggen D, et al. Wearable Assistant for Parkinson's Disease Patients With the Freezing of Gait Symptom. IEEE Trans Inf Technol Biomed. 2010;14(2):436-446. Dataset: UCI Machine Learning Repository, Daphnet Freezing of Gait (CC BY 4.0).
