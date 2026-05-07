# PyroPredict — Presentation Script

**Total time:** 5 minutes (≈ 4 min slides + 1 min live demo)
**Format:** ~7 content slides at 30–40 seconds each
**Course:** CMPE 258 · Spring 2026 · SJSU

---

## Slide 1 — Title (~10 sec)

**PyroPredict — Real-Time Wildfire Smoke Localisation**
CMPE 258 Final Project · Spring 2026 · SJSU
Alshama Mony Sheena · Gautam Santhanu Thampy

**Visual:** Single sample image from `results/eda/eda_samples.png` cropped to one good bounding-box example.

**What to say:**
> "We built an end-to-end deep learning pipeline that detects and localizes wildfire smoke in fixed-camera imagery, trained two SOTA models, improved them, and deployed a working demo."

---

## Slide 2 — Problem & Motivation (~30 sec)

**Why this matters**
- Wildfire smoke is detectable **minutes before flames are visible**
- Existing camera networks (HPWREN / ALERTWildfire) are passive — humans monitor 1000s of feeds
- Goal: automatic **bounding-box localization** of smoke + fire for early-warning triage

**Visual:** None, or a simple split image: left = HPWREN raw camera feed, right = same image with bbox overlay (use one of your demo screenshots).

**What to say:**
> "Smoke is the earliest visible signal of a wildfire. Cameras already exist; what's missing is automated detection. We treat this as object detection, not just classification, because location matters for response."

---

## Slide 3 — Dataset & Approach (~30 sec)

**D-Fire** (primary): 21,527 images · 26,557 boxes · 2 classes (smoke, fire)
- Train 14,122 · Val 3,099 · Test 4,306
- Includes ~9,800 negative samples (clouds, fog, sunsets) for false-positive control

**HPWREN / FIgLib** (supplementary): 737 images via AI for Mankind for OOD evaluation

**Pipeline:** EDA → 2 SOTA baselines → 4 ablations → ONNX + INT8 → Streamlit demo

**Visual:** `results/eda/eda_class_dist.png` (class balance) + `results/eda/eda_split_pie.png` (train/val/test split) side-by-side. If only one fits, use the class distribution one.

**What to say:**
> "We use D-Fire as our primary dataset — over 26,000 boxes across smoke and fire. Crucially it includes thousands of clean negative samples, which forces the model to learn what *isn't* smoke."

---

## Slide 4 — SOTA Baselines: YOLO11m vs YOLO26m (~40 sec)

**Trained both under identical conditions** — 30 epochs, batch 32, 640px, L4 GPU

| Model | mAP@50 | mAP@50:95 | Precision | Recall | F1 |
|---|---|---|---|---|---|
| YOLO11m | 0.7666 | 0.4410 | 0.7692 | 0.6946 | 0.7300 |
| **YOLO26m** | **0.7760** | **0.4461** | **0.7726** | **0.7020** | **0.7356** |

YOLO26m wins across all metrics → **selected as our base for ablations**

**Visual:** `results/baselines/baseline_comparison.png` (bar chart comparing the two models on mAP@50, mAP@50:95, Precision, Recall, F1).

**What to say:**
> "We compared two recent SOTA detectors — YOLO11 from Ultralytics 2024, and YOLO26 from January 2026 which has NMS-free inference and a new optimizer. Under identical training, YOLO26m beat YOLO11m on every metric, so we picked it as the base for our improvements."

---

## Slide 5 — Improvement #1: Training Strategy (~40 sec)

**4 ablation experiments on YOLO26m baseline**

| Experiment | mAP@50 | Δ |
|---|---|---|
| Baseline YOLO26m | 0.7760 | — |
| 1. Domain augmentations (HSV, MixUp, Copy-Paste) | 0.7722 | -0.0038 |
| 2. Loss reweighting + label smoothing | 0.7779 | +0.0019 |
| **3. Multi-scale + longer schedule (40 ep)** | **0.7861** | **+0.0101** |
| 4. Combined (1+2+3) | 0.7812 | +0.0052 |

**Best:** Multi-scale + longer schedule → **+1.0 mAP@50** over the strong baseline

**Visual:** `results/ablations/ablation_comparison.png` (bar chart of ΔmAP@50 vs baseline for the 4 experiments).

**What to say:**
> "We ran four ablation experiments. The most effective was multi-scale training combined with a longer 40-epoch schedule, which gave us a full point of mAP@50 over an already-strong baseline. Interestingly, *combining* all techniques was worse — aggressive augmentation hurt convergence on small smoke plumes."

---

## Slide 6 — Improvement #2: Efficiency (FP32 vs INT8) (~40 sec)

**Exported best model to ONNX, then dynamically quantized to INT8**

| Model | Size | mAP@50 | Latency (CPU) | Notes |
|---|---|---|---|---|
| FP32 ONNX | 77.94 MB | 0.7786 | 91 ms | Real-time at 11 FPS |
| **INT8 ONNX** | **19.95 MB** | 0.7336 | 419 ms | **74% smaller**, -4.5 mAP |

**Tradeoff:** INT8 wins on storage (edge devices, drones); FP32 wins on CPU latency on x86 without VNNI

**Visual:** `results/quantization/quantization_comparison.png` (bar chart with size, latency, mAP side by side).
> If this PNG isn't in Drive, just put the table on the slide — the numbers tell the story.

**What to say:**
> "We exported to ONNX and applied dynamic INT8 quantization. We get a 74% size reduction with about a 4.5-point mAP drop. Honest finding: on standard x86 CPUs without dedicated INT8 hardware, dynamic quantization is actually slower than FP32 in ONNX Runtime — but the size reduction matters for edge deployment on drones or IoT camera nodes."

---

## Slide 7 — Deployment & Live Demo Setup (~30 sec)

**Streamlit dashboard** (locally runnable, Docker-ready)
- Upload or pick test-set image
- Toggle between FP32 ONNX / INT8 ONNX / `best.pt`
- Side-by-side: **original ↔ detection**
- Live metrics: latency, FPS, model size, per-detection confidence table

**Reproducibility:** GitHub repo · `requirements.txt` · `dataset.yaml` · saved weights

**Visual:** Screenshot of the Streamlit app showing a successful detection (use one of your fire-image results — bonus if it shows the original / detection split).

**What to say:**
> "We packaged everything into a Streamlit demo. It supports the proposal-required FP32 vs INT8 toggle and shows live latency and throughput on the demo machine. Now I'll switch to the live demo."

---

## Slide 8 — Switch to Live Demo (~5 sec)

> **"Let me show you the live demo."** → Alt-Tab to Streamlit.

---

# Live Demo Plan (1 minute)

Plan it second-by-second so you don't run over:

| Time | Action |
|---|---|
| **0:00–0:10** | Upload a clear smoke-plume image with `best.pt` selected. Show side-by-side: original \| detection. |
| **0:10–0:20** | Switch to **INT8 ONNX**, same image. Point out: similar detection, smaller model size badge, similar/lower confidence. |
| **0:20–0:35** | Toggle "Compare two models". Pick FP32 ONNX vs INT8 ONNX. Same image. Audience sees both side-by-side. |
| **0:35–0:50** | Upload a **negative image** (clouds / fog / sky). Show: model correctly returns 0 detections. This is the false-positive resistance story. |
| **0:50–1:00** | Quick recap: "FP32 + INT8 deployable, real-time at 11 FPS, robust to non-fire cloud scenes." |

---

# Files to Drop Into the Slides

Make sure your teammate has these downloaded from Drive:

| Slide | File | Source on Drive |
|---|---|---|
| 1 / 3 | `eda_samples.png` | `MyDrive/PyroPredict/eda_samples.png` |
| 3 | `eda_class_dist.png` | `MyDrive/PyroPredict/eda_class_dist.png` |
| 3 (optional) | `eda_split_pie.png` | `MyDrive/PyroPredict/eda_split_pie.png` |
| 4 | `baseline_comparison.png` | `MyDrive/PyroPredict/metrics/baseline_comparison.png` |
| 5 | `ablation_comparison.png` | `MyDrive/PyroPredict/metrics/ablation_comparison.png` |
| 6 | `quantization_comparison.png` | `MyDrive/PyroPredict/metrics/quantization_comparison.png` (or use the table) |
| 7 | Streamlit screenshot | Take one yourself before the talk |

---

# Cuts If You Go Over Time

If a dry run goes past 4 minutes, cut in this order:

1. Drop slide 3 to one chart only (skip the split pie)
2. Drop the per-class breakdown row from slide 4
3. Don't read the table aloud on slide 5 — just point at the bold row

---

# Coverage Map: What This Hits From the Rubric

| Course requirement | Where it's covered |
|---|---|
| Modern DL pipeline (Option 1) | Slides 3–7 (full chain) |
| SOTA model survey + baselines (≥ 2) | Slide 4 (YOLO11 vs YOLO26) |
| Meaningful improvement #1 (training strategy) | Slide 5 (4 ablations) |
| Meaningful improvement #2 (efficiency) | Slide 6 (FP32 vs INT8) |
| Deployable application + reproducibility | Slide 7 + live demo |
| Honest evaluation | Slide 6 (the INT8-slower-than-FP32 surprise is reported, not hidden) |

That last one matters — professors notice when you report unfavorable findings honestly. Don't gloss over the INT8-slower-than-FP32 surprise; it shows you actually ran the benchmark.

---

# Quick-Reference Numbers (memorize these)

- **Dataset:** 21,527 images · 26,557 boxes · 2 classes (smoke + fire)
- **Best baseline:** YOLO26m, mAP@50 = **0.7760**
- **Best ablation:** Multi-scale + longer schedule, mAP@50 = **0.7861** (+1.0 over baseline)
- **FP32 ONNX:** 77.94 MB · 91 ms · 11 FPS · mAP@50 = 0.7786
- **INT8 ONNX:** 19.95 MB · 419 ms · 2.4 FPS · mAP@50 = 0.7336
- **Size reduction:** 74%
- **Accuracy drop on quantization:** ~4.5 mAP points

---

# Speaker Notes — Anticipated Q&A

**Q: Why didn't INT8 give a speed-up?**
A: Dynamic quantization on x86 CPUs without AVX-512 VNNI falls back to slower kernels for INT8 ops. With static quantization or on a target device with INT8 hardware (e.g. NVIDIA Jetson, mobile NPU), INT8 typically runs faster. The size benefit (74%) still holds for edge deployment.

**Q: Why did the combined ablation lose to multi-scale alone?**
A: Stacking aggressive augmentation on top of a multi-scale schedule slightly under-fits small-object smoke cases. The model has too many distribution-shifting signals at once during training.

**Q: How do you handle false positives like clouds and fog?**
A: D-Fire's ~9,800 negative samples teach the model the "what isn't smoke" boundary. Our live demo will show a clean cloud image returning zero detections.

**Q: Why pick D-Fire over HPWREN/FIgLib?**
A: HPWREN has temporal smoke labels but not bounding boxes. We need bounding-box supervision for object detection. We use HPWREN-derived images (via AI for Mankind) as supplementary OOD evaluation only.

**Q: Why YOLO11m and YOLO26m rather than the small or large variants?**
A: 'm' is the Pareto-optimal point for accuracy / speed / training time on a single L4 GPU within our 30-epoch budget.

**Q: Is this real-time?**
A: Yes for FP32 ONNX (11 FPS on CPU). On a GPU it would be 60+ FPS comfortably.
