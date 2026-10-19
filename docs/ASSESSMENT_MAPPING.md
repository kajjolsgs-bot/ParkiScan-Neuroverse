# Task → features → model → output

Every ParkiScan task, what it measures, where the features are computed, which model (if any) may use them,
and what the assessment endpoint (`GET /session/{id}/assessment`) outputs. Source of truth in code:
`ParkiScanML/parkiscan/assessment.py` (`TASK_MAP`).

> Research prototype. Nothing here is a diagnosis. The overall Parkinson's / not-Parkinson's classification is
> **withheld** because no model has been trained and validated on ParkiScan sessions with clinician-confirmed labels.

| Level · task | Domain | Features (Unity `TaskRecord` + service) | Model | Output |
|---|---|---|---|---|
| L1 · Follow the memory lantern | Visual pursuit & attention | pointer error mean/SD, % on target, re-acquisition latency, flash reaction time, jitter | none — no labelled dataset for this task | metrics + data-quality status |
| L1 · Memory Core A — sustained “Ahh” | Voice | voiced duration, F0 mean/SD (Unity autocorrelation + librosa pYIN), jitter/shimmer approximations, HNR, intensity, SNR, clipping | none — public voice sets use Praat/MDVP features from other microphones (not comparable) | metrics; recording kept locally (`Telemetry/<session>/recordings/`) |
| L1 · Memory Core B — “pa-ta-ka” | Speech motor (diadochokinesis) | onset rate, inter-onset CV, pauses, voiced segments | none | metrics |
| L1 · Memory Core C — clap to the beat (90 BPM) | Rhythm / motor timing | clap count, inter-clap interval mean & CV, tempo error, mean (abs) asynchrony to the metronome, beats matched; service clap detector | none | metrics |
| L1 · Memory Core D — repeat a sentence (selected language) | Connected speech | speech rate (syllables / voiced s), pauses, latency; WER/CER **only if** local ASR (faster-whisper) is installed | none | metrics; ASR marked unavailable otherwise |
| L1 · Recover the lost memory | Visual memory | hits, omissions, false recognitions, order match, recall time | none | metrics |
| L2 · Trace the neural circuit (spiral) | Fine motor | radial deviation, within-tolerance fraction, tremor-band proxy, LDLJ smoothness, velocity, pen lifts; service spiral kinematics (same extractor as training) | **spiral** (UCI tablet spirals; random forest; experimental, opt-in) | metrics + experimental research probability with abstention band |
| L2 · Connect the memory nodes | Processing speed & sequencing | completion time, sequencing accuracy, wrong connections, inter-selection time | none | metrics |
| L2 · Repair the memory component | Visuospatial planning | parts fitted, placement error, rotation accuracy, misalignments | none | metrics |
| L2 · Reactor signal | Reaction time & response control | RT mean/median/SD, accuracy, false & missed responses | none | metrics |
| L3 · Mirror the Memory Avatar | Upper-limb movement (webcam) | moves completed, peak arm elevation, similarity, tracking confidence, missing-frame fraction | none | metrics |
| L3 · Glowing path (standing) | Stepping & stopping (webcam) | steps, cadence, step-interval SD, alternation, stop latency at red, go latency at green, steps during red | none — PhysioNet gait models use insole stride intervals over 5 min | metrics |
| L3 · Memory vehicle (seated) | Hand reactions to traffic signals (webcam) | right/left raises, brake latency (yellow → left hand), go latency (green → right hand), red-light stops, ambiguous both-hand events, calibrated thresholds | none | metrics |
| L3 · Six check-ins | Self-report | answer per domain (0–3 frequency / difficulty), shown language, voice played | none — unvalidated questions; summarised, never scored | per-domain answers |
| Wearable ankle IMU (future) | Freezing of gait | 4-s windows: per-axis loco/freeze band power, freeze index | **daphnet** (HGB on features; 1-D CNN on raw windows also trained) | only when real, non-simulated hardware data exists |
| Wearable wrist IMU (future) | Tremor | 4–7 Hz band ratio, peak frequency, gyro RMS | none yet | only when hardware exists |

## Data-quality states

`valid`, `valid_with_caveats` (degraded quality flags), `insufficient` (missing key features / incomplete),
`invalid` (QA test run or simulated input), `not_collected` (skipped / sensor absent). Only `valid*` domains are
listed as usable evidence; everything else carries its reason.

## Why most tasks have no model

A model may only score a feature vector produced by the same extractor, protocol and kind of device it was
trained on. Of the seven public datasets available, that holds only for spiral traces (same kinematic
extractor, though tablet vs. mouse/touch is still a domain shift) and ankle-accelerometer windows (Daphnet,
which needs the future ankle IMU). Applying the voice, gait-insole or eye-tracking-image models to game data
would produce numbers that look precise and mean nothing, so the pipeline refuses to.

## What would be needed for a validated classification and sub-stratification

1. ParkiScan sessions from participants with clinician-confirmed diagnoses (PD, other parkinsonism, healthy
   controls) and MDS-UPDRS / Hoehn & Yahr ratings.
2. Training on those sessions with participant-grouped CV, then an independent external test set.
3. Reported sensitivity, specificity, PPV/NPV, calibration and abstention rate, per language and per device.
4. Ethics approval and clinician review of every individual result.
