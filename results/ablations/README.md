# Ablation Results — Phase 4

Four ablation experiments were run on top of the YOLO26m baseline to identify
the most effective training-strategy improvements.

## What to download here

From `Google Drive: MyDrive/PyroPredict/`:

| Drive file | Local path |
|---|---|
| `metrics/ablation_results.csv` | `results/ablations/ablation_results.csv` |
| `metrics/ablation_comparison.png` | `results/ablations/ablation_comparison.png` |

### Per-experiment plots (optional but recommended)

From `runs/ablation_*/`:

| Drive file | Local path |
|---|---|
| `runs/ablation_1_augment/results.png` | `results/ablations/exp1_augment_results.png` |
| `runs/ablation_2_negatives/results.png` | `results/ablations/exp2_negatives_results.png` |
| `runs/ablation_3_multiscale/results.png` | `results/ablations/exp3_multiscale_results.png` |
| `runs/ablation_4_combined/results.png` | `results/ablations/exp4_combined_results.png` |

## What each experiment changed

| # | Name | Key change vs baseline |
|---|---|---|
| 1 | augment | Domain-specific augmentations: HSV shifts, MixUp, Copy-Paste |
| 2 | negatives | Loss reweighting (`box=8.0`, `cls=0.7`), label smoothing |
| 3 | multiscale | Longer schedule (40 epochs) + multi-scale training |
| 4 | combined | All of the above stacked together |

## Results (test set, mAP@50)

| Model | mAP@50 | ΔmAP@50 | mAP@50:95 | F1 | Train time (min) |
|---|---|---|---|---|---|
| YOLO26m (baseline) | 0.7760 | 0.0000 | 0.4461 | 0.7356 | — |
| Ablation 1: augment | 0.7722 | -0.0038 | 0.4465 | 0.7297 | 192.6 |
| Ablation 2: negatives | 0.7779 | +0.0019 | 0.4462 | 0.7438 | 192.0 |
| **Ablation 3: multiscale** | **0.7861** | **+0.0101** | **0.4579** | **0.7408** | 419.0 |
| Ablation 4: combined | 0.7812 | +0.0052 | 0.4521 | 0.7350 | 418.6 |

**Best model: Ablation 3 — multi-scale + longer schedule.**
This is the model used for Phase 5 (quantization) and the Streamlit demo.

## Why combined didn't beat multiscale

Stacking aggressive augmentation on top of multiscale schedule slightly under-fits
small-object smoke cases. Multiscale alone gave the best mAP@50 / mAP@50:95 tradeoff.
