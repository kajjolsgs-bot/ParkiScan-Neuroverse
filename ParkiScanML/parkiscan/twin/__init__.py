"""ParkiScan Neuroverse · Game + Digital Twin — a per-patient virtual replica that fuses static clinical records (EHR) with streaming
wearable / game data, keeps a running physiological state (levodopa pharmacokinetics → motor ON/OFF state), and
predicts one adverse event: freezing of gait (FoG).

Two prediction horizons, two evidence levels — kept separate on purpose:

* **Seconds ahead (real data).**  `fog_forecast` — a model trained and evaluated leave-one-subject-out on the real
  Daphnet ankle/thigh/trunk accelerometer recordings, fused with the real per-patient clinical header published with
  that dataset (age, sex, disease duration, Hoehn & Yahr, medication state).
* **Hours ahead (synthetic data).**  `cohort` + `risk` — a population model trained on a synthetic cohort
  (Synthea-style EHR + simulated 7-day wearables + levodopa PK). It demonstrates the fusion pipeline end to end; its
  accuracy is a property of the simulator, not clinical evidence.

Research prototype. Not a medical device. Not for diagnosis or treatment decisions.
"""
