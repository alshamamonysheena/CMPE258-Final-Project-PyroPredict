# Baseline Results — Phase 3 (YOLO11m vs YOLO26m)

## What to download here

From `Google Drive: MyDrive/PyroPredict/`, download:

| Drive file | Local path |
|---|---|
| `metrics/baseline_comparison.csv` | `results/baselines/baseline_comparison.csv` |
| `metrics/baseline_comparison.png` | `results/baselines/baseline_comparison.png` |
| `metrics/perclass_ap.csv` | `results/baselines/perclass_ap.csv` |
| `metrics/training_curves.png` | `results/baselines/training_curves.png` |

### Optional: per-run plots from each model

From `runs/yolo11m_baseline/` and `runs/yolo26m_baseline/`:

| Drive file | Local path |
|---|---|
| `runs/yolo11m_baseline/results.png` | `results/baselines/yolo11m_results.png` |
| `runs/yolo11m_baseline/confusion_matrix.png` | `results/baselines/yolo11m_confusion_matrix.png` |
| `runs/yolo11m_baseline/BoxPR_curve.png` | `results/baselines/yolo11m_PR_curve.png` |
| `runs/yolo26m_baseline/results.png` | `results/baselines/yolo26m_results.png` |
| `runs/yolo26m_baseline/confusion_matrix.png` | `results/baselines/yolo26m_confusion_matrix.png` |
| `runs/yolo26m_baseline/BoxPR_curve.png` | `results/baselines/yolo26m_PR_curve.png` |

## Headline numbers (test set)

| Model | mAP@50 | mAP@50:95 | Precision | Recall | F1 |
|---|---|---|---|---|---|
| YOLO11m | 0.7666 | 0.4410 | 0.7692 | 0.6946 | 0.7300 |
| YOLO26m | 0.7760 | 0.4461 | 0.7726 | 0.7020 | 0.7356 |

YOLO26m is the stronger baseline and is used as the starting point for Phase 4 ablations.
