# Quantization Results — Phase 5 (FP32 vs INT8)

The best model from Phase 4 (`ablation_3_multiscale`) was exported to ONNX FP32
and then quantized to INT8 using ONNX Runtime dynamic quantization. Both models
were benchmarked on the same held-out test set.

## What to download here

From `Google Drive: MyDrive/PyroPredict/`:

| Drive file | Local path |
|---|---|
| `metrics/quantization_results.csv` | `results/quantization/quantization_results.csv` |
| `metrics/quantization_comparison.png` | `results/quantization/quantization_comparison.png` |

## Headline numbers

| Model | Size (MB) | Latency (ms) | FPS | mAP@50 | mAP@50:95 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| FP32 ONNX | 77.94 | 91.40 | 10.94 | 0.7786 | 0.4467 | 0.7604 | 0.7122 |
| INT8 ONNX | 19.95 | 419.37 | 2.38 | 0.7336 | 0.4035 | 0.7183 | 0.6796 |

## Tradeoff summary

- **Size:** ~74% reduction (77.94 MB → 19.95 MB)
- **Accuracy:** mAP@50 drops by ~4.5 points (0.7786 → 0.7336)
- **Latency on this CPU:** INT8 was *slower* than FP32 in ONNX Runtime CPU.
  This is a known property of dynamic quantization on x86 CPUs without VNNI/AVX-512
  optimizations — INT8 ops fall back to slower kernels. With static quantization or
  on a target device with proper INT8 hardware support, INT8 typically runs faster.

## Recommendation
- For the **real-time wildfire monitoring** use case → use **FP32 ONNX** (10.9 FPS, higher accuracy).
- For **storage-constrained edge deployment** (drones, IoT camera nodes) → INT8 is viable
  if the latency penalty on that device is acceptable.

The Streamlit demo lets the user toggle between FP32 and INT8 to visualize this tradeoff live.
