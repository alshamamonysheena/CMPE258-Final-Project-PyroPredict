# PyroPredict 48-Hour Demo Runbook

Use this as the handoff between Colab training and the local Streamlit demo.

## Quick Links

- [Open EDA notebook in Colab](https://colab.research.google.com/github/alshamamonysheena/CMPE258-Final-Project-PyroPredict/blob/main/notebooks/01_data_and_eda.ipynb)
- [Open training notebook in Colab](https://colab.research.google.com/github/alshamamonysheena/CMPE258-Final-Project-PyroPredict/blob/main/notebooks/02_train_baselines.ipynb)

In Colab, use `Runtime -> Change runtime type -> T4 GPU` or any available GPU. Use GPU, not TPU, for the Ultralytics YOLO workflow.

## Phase 1: Colab Data Check

Run `notebooks/01_data_and_eda.ipynb` in Google Colab.

Expected outputs in Google Drive under `MyDrive/PyroPredict/`:

- `dataset.yaml`
- `eda_report.json`
- `eda_samples.png`
- `eda_class_dist.png`
- `eda_distributions.png`
- `eda_box_scatter.png`
- `eda_split_pie.png`

Before moving on, confirm:

- The dataset has `train`, `val`, and `test` splits.
- The class mapping is `0: fire`, `1: smoke`.
- At least a few annotated sample images look correctly labeled.

## Phase 2: Colab Training

Run `notebooks/02_train_baselines.ipynb`.

Priority order:

1. Keep `RUN_MODE = 'smoke'` and `TRAIN_SECOND_MODEL = False`; run YOLO11m for the 2-epoch smoke test.
2. Change `RUN_MODE = 'full'`; run full YOLO11m baseline training.
3. Evaluate YOLO11m on the test split.
4. Export YOLO11m to ONNX.
5. If there is time left, set `TRAIN_SECOND_MODEL = True`; start YOLO26m if available, otherwise use the notebook fallback and label it honestly.

Expected outputs in Google Drive:

- `runs/yolo11m_baseline/weights/best.pt`
- `exports/yolo11m_baseline.onnx`
- `metrics/baseline_comparison.csv`
- `metrics/perclass_ap.csv`
- `metrics/baseline_comparison.png`
- `metrics/training_curves.png`

If time is tight, one complete YOLO11m baseline is better than two incomplete models.

For a smoke-test-only run, outputs will use the `yolo11m_smoke_test` suffix. Do not use those smoke-test weights as the final demo result unless full training fails.

## Phase 3: Local Demo Setup

Create these local ignored folders:

```powershell
mkdir models
mkdir data\samples
```

Copy artifacts back from Drive:

- `runs/yolo11m_baseline/weights/best.pt` -> `models/yolo11m_baseline.pt`
- `exports/yolo11m_baseline.onnx` -> `models/yolo11m_baseline.onnx` if available
- 5-10 test/demo images -> `data/samples/`

Run the app:

```powershell
streamlit run app/app.py
```

Demo checklist:

- Model appears in the sidebar.
- Upload image works.
- Sample image tab works.
- Boxes and labels display as `fire` or `smoke`.
- Detection table, latency, FPS, and model size render.

## Phase 4: Backup Materials

Save screenshots of:

- App before inference with model selected.
- App after a clear positive detection.
- EDA class distribution.
- Baseline metrics comparison or YOLO11 result table.
- Training curves if available.

Use screenshots as backup if live model loading fails.
