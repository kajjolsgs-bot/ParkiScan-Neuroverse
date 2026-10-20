# Third-party notices

ParkiScan's own source code is licensed under Apache-2.0 (see `LICENSE`). The components below keep their own licences.

## Machine-learning models (run on-device in Unity, ONNX)

| component | file(s) | source | licence |
|---|---|---|---|
| MediaPipe BlazePose (pose detector + full landmark model) | `Assets/ParkiScan/ML/BlazePose/*.onnx`, `anchors.csv` | Google MediaPipe, ONNX conversion by Unity Technologies (inference-engine-samples) | Apache-2.0 |
| MediaPipe BlazeFace short-range face detector | `Assets/ParkiScan/Resources/FaceMesh/blaze_face_short_range.onnx`, `face_anchors.csv` | Google MediaPipe, ONNX by Unity Technologies (huggingface.co/unity/inference-engine-blaze-face) | Apache-2.0 |
| MediaPipe Face Landmarker v2 face mesh (478 landmarks incl. iris) | `Assets/ParkiScan/Resources/FaceMesh/face_landmarks_detector.onnx` | extracted from Google's `face_landmarker.task` (float16 v1) and converted to ONNX with tf2onnx 1.17 | Apache-2.0 |
| BlazePose / BlazeFace post-processing helpers | `Assets/ParkiScan/Scripts/Tracking/BlazeMath.cs` | ported from Unity-Technologies/inference-engine-samples | Apache-2.0 |

## Libraries

| library | use | licence |
|---|---|---|
| Chart.js 4.4.1 (`ParkiScanML/app/dashboard/static/chart.umd.min.js`) | dashboard charts | MIT |
| FastAPI, Starlette, pydantic, uvicorn | twin service | MIT / BSD-3-Clause |
| NumPy, SciPy, pandas, scikit-learn, joblib | models | BSD-3-Clause |
| librosa, soundfile | voice features | ISC / BSD-3-Clause |
| Unity packages (URP, Inference Engine, Input System, TextMeshPro) | game | Unity Companion License / Unity terms |
| Unity glTFast | 3-D model import | Apache-2.0 |

## Datasets (used for training and evaluation only; raw data is not redistributed in this repository)

| dataset | licence / terms | citation |
|---|---|---|
| Daphnet Freezing of Gait (UCI ML Repository, doi:10.24432/C56K78) | CC BY 4.0 | Bächlin M, et al. IEEE Trans Inf Technol Biomed 2010;14(2):436-446 · Roggen D, Plotnik M, Hausdorff J. Daphnet Freezing of Gait [Dataset], UCI, 2010 |
| PhysioNet Gait in Parkinson's Disease / Gait in Neurodegenerative Disease | PhysioNet open-access licences (see each project page) | Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 2000 · original dataset papers |
| UCI Parkinson Disease Spiral Drawings, Oxford Parkinson's (voice), Parkinson's Telemonitoring, Parkinson Speech (multiple recordings) | CC BY 4.0 (UCI) | as listed in `ParkiScanML/data/raw/MANIFEST.json` after `tools/prepare_datasets.py` |

The ten Daphnet participants' clinical header used by the digital twins (age, sex, disease duration, Hoehn & Yahr, medication state) is transcribed from Table I of Bächlin et al. 2010.

## Fonts

Rajdhani and Noto Sans (Indic scripts) — SIL Open Font License 1.1.

`Assets/ParkiScan/Art/Fonts/ParkiScanSymbols.ttf` — a renamed subset of DejaVu Sans 2.37 (symbols such as ✓ ▸ ◆ ▲ →), used as a TextMeshPro fallback font. DejaVu changes are public domain; the underlying Bitstream Vera licence (permissive; modified versions must be renamed) is included in `ParkiScanSymbols_LICENSE.txt` next to the font.

## Game art

3-D models, textures, music and sound in `Assets/` were generated or supplied for this project. Check the licence of any asset you replace or redistribute separately.
