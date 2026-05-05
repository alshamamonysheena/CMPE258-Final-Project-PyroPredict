# PyroPredict — Real-Time Wildfire Smoke Localisation

[![Open EDA In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alshamamonysheena/CMPE258-Final-Project-PyroPredict/blob/main/notebooks/01_data_and_eda.ipynb)
[![Open Training In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alshamamonysheena/CMPE258-Final-Project-PyroPredict/blob/main/notebooks/02_train_baselines.ipynb)

**CMPE 258 Deep Learning Final Project** | Spring 2026 | San Jose State University

---

## Team Members

| Name | SJSU ID | Email |
|------|---------|-------|
| Alshama Mony Sheena | 018214560 | alshama.monysheena@sjsu.edu |
| Gautam Santhanu Thampy | 018222477 | gautamsanthanu.thampy@sjsu.edu |

**Project Track:** Option 1 — Modern Deep Learning Pipeline (Training + Deployment)

---

## Project Overview

PyroPredict is an end-to-end deep-learning pipeline for early wildfire detection
through smoke localisation in fixed-camera imagery. The system trains, evaluates,
and deploys two state-of-the-art object detectors — Ultralytics **YOLO11** and
**YOLO26** — and demonstrates meaningful improvements through training-strategy
changes and efficiency optimisation.

## Dataset

| Property | Details |
|----------|---------|
| **Primary dataset** | [D-Fire](https://github.com/gaiasd/DFireDataset) — 21,527 images, 26,557 bounding boxes |
| **Classes** | `fire` (14,692 boxes) · `smoke` (11,865 boxes) |
| **Negative samples** | ~9,800 images (no fire/smoke) for false-positive reduction |
| **Splits** | Train 14,122 · Val 3,099 · Test 4,306 images |
| **Supplementary** | [HPWREN/FIgLib](https://www.hpwren.ucsd.edu/FIgLib/) camera imagery (737 images via AI for Mankind) for out-of-distribution evaluation |
| **Format** | YOLO bounding-box format (`class x_center y_center width height`) |

## Approach

### A. Problem Formulation
Localise wildfire smoke in camera imagery for early detection. Success criteria:
mAP@50 ≥ 0.85, low false-positive rate on cloud/fog/haze scenes.

### B. SOTA Model Survey & Baselines
Two recent open-source detectors trained under identical conditions:
- **YOLO11m** (Ultralytics, Sep 2024) — improved backbone/neck architecture
- **YOLO26m** (Ultralytics, Jan 2026) — NMS-free inference, MuSGD optimiser, up to 43% faster CPU inference

### C. Meaningful Model Improvements (2 axes)

1. **Training-strategy change** — domain-specific augmentations (synthetic haze/fog,
   dawn/dusk lighting shifts, copy-paste smoke regions) and hard-negative mining
   (clouds, steam, dust) with full ablation study.

2. **Efficiency improvement** — ONNX export with INT8 post-training quantisation,
   benchmarked for accuracy drop, latency (ms/frame), throughput (FPS), and model size.

### D. Deployment
Streamlit dashboard with model comparison toggle (YOLO11 vs YOLO26, FP32 vs INT8),
bounding-box visualisation, and live inference metrics. Dockerised for reproducibility.

## Project Structure

```
├── configs/            Training & dataset YAML configs
├── src/
│   ├── data/           Download, prepare, and split scripts
│   │   ├── download.py     Multi-source data download (Roboflow, FIgLib, local)
│   │   ├── prepare.py      VOC→YOLO conversion, validation, statistics
│   │   └── split.py        Event-aware train/val/test splitting
│   ├── train/          Training scripts (Phase 3–4)
│   ├── export/         ONNX export & quantisation (Phase 5)
│   └── utils/
│       └── viz.py      Annotated-sample grids, distribution plots
├── notebooks/
│   ├── 01_data_and_eda.ipynb     Data download & EDA (Colab, CPU)
│   ├── 02_train_baselines.ipynb  YOLO11/26 baseline training (Colab, GPU)
│   ├── 03_ablations.ipynb        Augmentation ablation experiments (upcoming)
│   └── 04_quantize.ipynb         ONNX export & INT8 benchmarking (upcoming)
├── app/                Streamlit demo application
├── docker/             Dockerfile & compose (upcoming)
├── data/               Downloaded & processed data (git-ignored)
├── models/             Saved weights & ONNX files (git-ignored)
├── requirements.txt    Pinned Python dependencies
└── .gitignore
```

## Progress

### Completed
- [x] Project proposal and problem formulation
- [x] Data pipeline: multi-source download script (Roboflow, FIgLib/HPWREN, local)
- [x] Annotation conversion (VOC XML → YOLO format) with validation
- [x] Event-aware dataset splitting (prevents data leakage across fire events)
- [x] EDA notebook: dataset statistics, sample visualisation, class/box distributions
- [x] Successfully ran EDA on D-Fire (14,122 train / 3,099 val / 4,306 test images)
- [x] Baseline training notebook for YOLO11m and YOLO26m (ready to execute)
- [x] Streamlit demo app with upload/sample image flow, model selector, annotations, and inference metrics

### In Progress
- [ ] Train YOLO11m and YOLO26m baselines on D-Fire (Colab GPU)
- [ ] Baseline evaluation on test set (mAP, precision, recall, F1)
- [ ] Copy trained weights into local `models/` for the live demo

### Next Steps
- [ ] Phase 4: Augmentation ablation experiments (domain augmentations, hard negatives, training recipe)
- [ ] Phase 5: ONNX export + INT8 quantisation + efficiency benchmarking
- [ ] Phase 6: Streamlit demo polish, screenshots, and demo rehearsal
- [ ] Phase 7: Dockerised deployment artifact
- [ ] Final report and presentation

## How to Run

### 1. Colab data and training

Use Colab for anything that needs the D-Fire dataset or GPU training:

- [Open the EDA notebook in Colab](https://colab.research.google.com/github/alshamamonysheena/CMPE258-Final-Project-PyroPredict/blob/main/notebooks/01_data_and_eda.ipynb)
- [Open the training notebook in Colab](https://colab.research.google.com/github/alshamamonysheena/CMPE258-Final-Project-PyroPredict/blob/main/notebooks/02_train_baselines.ipynb)

In Colab, choose `Runtime -> Change runtime type -> T4 GPU` or any available GPU. Keep the training notebook's default `RUN_MODE = 'smoke'` for the first 2-epoch validation run, then switch to `RUN_MODE = 'full'` for the real YOLO11m run. Set `TRAIN_SECOND_MODEL = True` only after YOLO11m finishes and there is time left.

### 2. Local Streamlit demo

After Colab saves outputs to Google Drive, copy only the small demo artifacts back locally:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

mkdir models
mkdir data\samples

# Copy from Google Drive manually:
# MyDrive/PyroPredict/runs/yolo11m_baseline/weights/best.pt -> models/yolo11m_baseline.pt
# MyDrive/PyroPredict/exports/yolo11m_baseline.onnx -> models/yolo11m_baseline.onnx
# 5-10 demo images -> data/samples/

streamlit run app/app.py
```

Large files under `data/`, `models/`, `runs/`, `*.pt`, and `*.onnx` are intentionally git-ignored.

## References

For the current 48-hour demo workflow, see [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md).

- D-Fire Dataset: [github.com/gaiasd/DFireDataset](https://github.com/gaiasd/DFireDataset)
- HPWREN FIgLib: [hpwren.ucsd.edu/FIgLib](https://www.hpwren.ucsd.edu/FIgLib/)
- Dewangan et al. (2022). FIgLib & SmokeyNet. *Remote Sensing* 14(4):1007
- Ultralytics YOLO11: [docs.ultralytics.com/models/yolo11](https://docs.ultralytics.com/models/yolo11/)
- Ultralytics YOLO26: [docs.ultralytics.com/models/yolo26](https://docs.ultralytics.com/models/yolo26/)
- ONNX Runtime Quantisation: [onnxruntime.ai/docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

## Licence

Academic use only — CMPE 258 course project.
