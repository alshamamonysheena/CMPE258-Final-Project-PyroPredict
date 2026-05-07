# PyroPredict — Results

This folder contains all evaluation artifacts for the project, organized by phase.

## Structure

```
results/
├── eda/             Phase 2 — Dataset analysis charts
├── baselines/       Phase 3 — YOLO11m vs YOLO26m baseline results
├── ablations/       Phase 4 — Training-strategy ablation experiments
└── quantization/    Phase 5 — FP32 vs INT8 efficiency benchmark
```

## Where to download files from

All files come from the project's Google Drive folder:

```
Google Drive: MyDrive/PyroPredict/
```

See each subfolder's `README.md` for the exact files to download and where to put them.

## Top-line results (for quick reference)

### Baselines (Phase 3, test set)
| Model | mAP@50 | mAP@50:95 | Precision | Recall | F1 |
|---|---|---|---|---|---|
| YOLO11m | 0.7666 | 0.4410 | 0.7692 | 0.6946 | 0.7300 |
| YOLO26m | 0.7760 | 0.4461 | 0.7726 | 0.7020 | 0.7356 |

### Ablations (Phase 4, on top of YOLO26m)
| Experiment | mAP@50 | Δ vs baseline |
|---|---|---|
| Baseline (YOLO26m) | 0.7760 | — |
| Ablation 1: Domain augmentations | 0.7722 | -0.0038 |
| Ablation 2: Loss + label smoothing | 0.7779 | +0.0019 |
| Ablation 3: Multi-scale + longer schedule | **0.7861** | **+0.0101** |
| Ablation 4: Combined | 0.7812 | +0.0052 |

**Best model: Ablation 3 (multi-scale + longer schedule)**

### Quantization (Phase 5, FP32 vs INT8 ONNX, test set, CPU)
| Model | Size | Latency | FPS | mAP@50 |
|---|---|---|---|---|
| FP32 ONNX | 77.94 MB | 91.4 ms | 10.94 | 0.7786 |
| INT8 ONNX | 19.95 MB | 419.4 ms | 2.38 | 0.7336 |

- Size reduction: **~74%**
- mAP@50 drop: **~4.5 points**
- Latency note: INT8 was slower than FP32 in ONNX Runtime CPU on this hardware.
  Tradeoff favors FP32 for real-time, INT8 for storage-constrained deployment.
