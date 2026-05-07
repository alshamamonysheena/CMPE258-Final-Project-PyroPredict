# EDA Results — Phase 2

## What to download here

From `Google Drive: MyDrive/PyroPredict/`, download these files into this folder:

| Drive file | Local path |
|---|---|
| `eda_class_dist.png` | `results/eda/eda_class_dist.png` |
| `eda_distributions.png` | `results/eda/eda_distributions.png` |
| `eda_box_scatter.png` | `results/eda/eda_box_scatter.png` |
| `eda_split_pie.png` | `results/eda/eda_split_pie.png` |
| `eda_samples.png` | `results/eda/eda_samples.png` |
| `eda_report.json` | `results/eda/eda_report.json` |

## What each file shows

- **eda_samples.png** — 12 random training images with bounding boxes drawn (proves labels are valid)
- **eda_class_dist.png** — Bar chart: number of fire vs smoke boxes per split (train/val/test)
- **eda_distributions.png** — Histograms of image resolution and box width/height
- **eda_box_scatter.png** — Scatter of box width vs height by class (shows shape distribution)
- **eda_split_pie.png** — Pie chart of train/val/test image proportions
- **eda_report.json** — Raw stats (counts, classes, split sizes) used to produce the charts
