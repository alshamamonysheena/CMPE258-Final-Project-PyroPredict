# PyroPredict — Local Streamlit Demo

Run a local interactive UI for wildfire smoke and fire detection using your trained models.

---

## 1. Download required models from Google Drive

From `MyDrive/PyroPredict/`, download these into the project's `models/` folder:

| Source on Drive | Local destination |
|---|---|
| `runs/ablation_3_multiscale/weights/best.pt` | `models/best.pt` |
| `exports/ablation_4_combined.onnx` | `models/ablation_4_combined.onnx` |
| `exports/ablation_4_combined_int8.onnx` | `models/ablation_4_combined_int8.onnx` |

You only need at least **one** model for the demo to work. The `.pt` file is fastest and most accurate.

## 2. Install dependencies

From the project root:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install streamlit ultralytics onnxruntime opencv-python-headless pillow numpy
```

## 3. (Optional) Add sample images for the demo

Drop a few wildfire / smoke test images here:

```
data/samples/
```

These will appear under the **Sample** tab in the UI.

## 4. Launch the app

From the project root:

```bash
streamlit run app/app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`).

## 5. What you can do in the demo

- Upload an image **or** pick a sample.
- Pick a model from the sidebar (auto-discovered from `models/`).
- Adjust confidence and IoU thresholds.
- Toggle **Compare two models side-by-side** (e.g. FP32 vs INT8).
- See bounding boxes, confidence, latency, FPS, model size.

## Recording the demo video

1. Open the Streamlit app in your browser.
2. Start screen recording (macOS: `Cmd + Shift + 5`).
3. Walk through:
   - Upload a smoke image -> show detections + metrics.
   - Switch to a different model -> show difference.
   - Toggle compare mode -> FP32 vs INT8 side-by-side.
4. Stop and trim in QuickTime if needed.
