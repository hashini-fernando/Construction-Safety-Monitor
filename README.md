# Construction Safety Monitor
> Automated PPE compliance detection for construction sites using YOLOv8

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

Construction sites are among the most hazardous work environments in the world. This system automatically monitors construction site images and video to determine in real time whether workers are compliant with PPE requirements — answering the core question:

**Is this situation safe or unsafe?**

The pipeline detects workers, identifies PPE items (hard hats, safety vests), associates PPE with individual workers, applies per-worker compliance rules, and produces structured violation reports with confidence-based uncertainty flagging.

---

## Demo

```
python inference.py --source sample_images/ --weights best.pt
```

Or launch the interactive web app:
```
streamlit run app.py
```

> **Download trained weights:** [best.pt — Google Drive](#)  
> *(Replace `#` with your actual Google Drive link before submitting)*

---

## Safety Rules Definition

The following rules are enforced by the compliance engine. Rules were defined before model training to drive annotation decisions.

### Rule 1 — Hard Hat Required
| | |
|---|---|
| **Requirement** | Every worker present in a scene must wear a hard hat / safety helmet at all times |
| **Violation signal** | `no_hard_hat` class detected near a `person` bounding box, or no helmet detected near a worker at all |
| **Severity** | Critical |

### Rule 2 — High-Visibility Safety Vest Required
| | |
|---|---|
| **Requirement** | Every worker must wear a high-visibility (hi-vis) safety vest in any active work zone |
| **Violation signal** | `no_safety_vest` class detected near a `person` bounding box, or no vest detected near a worker |
| **Severity** | Critical |

### Rule 3 — Uncertainty Flagging
| | |
|---|---|
| **Requirement** | Predictions with confidence below 0.50 must not be treated as definitive — they are surfaced as uncertain and flagged for manual review |
| **Rationale** | Occluded workers, distant workers, or poor lighting can produce low-confidence detections. Silently classifying these as compliant or non-compliant introduces false safety assurances |
| **Output** | Verdict becomes `SAFE-UNCERTAIN` or `UNSAFE-UNCERTAIN` |

### Compliance Logic Summary

```
For each detected worker:
  IF no_hard_hat detected near worker  →  VIOLATION: "Missing hard hat"
  IF no_safety_vest detected near worker  →  VIOLATION: "Missing safety vest"
  IF any detection confidence < 0.50  →  flag as UNCERTAIN

Scene verdict:
  All workers compliant   →  SAFE
  Any worker violated     →  UNSAFE
  Any uncertain detection →  append -UNCERTAIN suffix
  No workers detected     →  NO_WORKERS
```

---

## Dataset

### Sources

| Source | Images | License | How used |
|---|---|---|---|
| [Roboflow Universe — Construction Site Safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) | ~2,800 | CC BY 4.0 | Primary base dataset |
| [PPE Combined Model — Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model) | ~500 subset | CC BY 4.0 | Diversity extension |
| Unsplash / Pexels (CC0) | ~80 | CC0 | Custom addition — varied lighting & environments |

**Total dataset size after cleaning:** ~3,400 images

### Class Distribution (after balancing)

| Class | Target Count | Strategy |
|---|---|---|
| `person` | ~2,200 | Undersampled (original: 3,522) |
| `hard_hat` | ~2,200 | Used as-is (original: 2,653) |
| `no_hard_hat` | ~2,200 | Augmented (original: 1,904) |
| `safety_vest` | ~2,200 | Augmented 2× (original: 1,404) |
| `no_safety_vest` | ~2,200 | Used as-is (original: 2,882) |

> `machinery` and `vehicle` classes present in the base dataset were **excluded** — only 41–44 instances each, insufficient for reliable detection, and outside the scope of PPE compliance.

### Train / Val / Test Split

| Split | Images |
|---|---|
| Train | 70% |
| Validation | 20% |
| Test | 10% |

### Dataset Diversity

The dataset intentionally includes:
- **Outdoor construction sites** — open lots, building frames, scaffolding
- **Indoor environments** — warehouses, basement construction
- **Lighting variation** — bright daylight, overcast, shadows, artificial lighting
- **Scale variation** — workers close to camera and distant (via mosaic augmentation)
- **Crowd variation** — single workers and multi-worker scenes

### Annotation Approach

- Annotated using **Roboflow** annotation tool
- Bounding box format: YOLO (normalised `cx cy w h`)
- Custom images (Unsplash/Pexels) annotated manually with Roboflow
- Class remapping applied at export:
  ```
  Hardhat        → hard_hat
  NO-Hardhat     → no_hard_hat
  Safety Vest    → safety_vest
  NO-Safety Vest → no_safety_vest
  Person         → person
  ```

### Dataset Cleaning

A dedicated cleaning pipeline (`data/dataset_cleaning.ipynb`) was run before training. It checked for and resolved:

- Corrupt or unreadable image files
- Missing or empty label files
- Invalid YOLO coordinates (out of `[0,1]` range) — auto-clamped where possible
- Wrong class IDs beyond defined range — rows dropped
- Tiny bounding boxes below 0.05% image area — rows dropped
- Near-duplicate frames (perceptual hash distance ≤ 8) — duplicates removed
- Extreme aspect ratios — logged but retained (640px resize handles these)

---

## Model & Training

### Architecture Choice

**YOLOv8m (medium)** was selected as the backbone.

| Consideration | Decision |
|---|---|
| Task type | Multi-class object detection — YOLOv8 is purpose-built for this |
| Model size | Medium (25M params) — best balance of accuracy and inference speed on T4 GPU |
| Transfer learning | Initialised from COCO-pretrained weights — leverages rich feature representations for person and object detection |
| Alternative considered | YOLOv9m — marginally higher COCO mAP but rougher tooling and less stable Colab support; the 1–2% gap disappears on domain-specific data |

### Training Configuration

```python
model.train(
    data       = 'data.yaml',
    epochs     = 100,
    imgsz      = 640,
    batch      = 16,           # T4 GPU
    optimizer  = 'AdamW',
    lr0        = 0.001,
    lrf        = 0.01,         # cosine LR decay
    weight_decay = 0.0005,
    warmup_epochs = 3,
    mosaic     = 1.0,          # multi-scale composite scenes
    mixup      = 0.1,
    copy_paste = 0.1,          # improves small/occluded object detection
    close_mosaic = 10,         # disable mosaic last 10 epochs
    patience   = 20,           # early stopping
)
```

**Key hyperparameter decisions:**
- `mosaic=1.0` — creates composite training images with workers at varying scales, directly improving detection of distant workers
- `close_mosaic=10` — disabling mosaic for the final 10 epochs allows the model to stabilise on clean, single-image inputs matching inference conditions
- `copy_paste=0.1` — copies and pastes object instances into scenes, improving detection of partially occluded workers
- `patience=20` — early stopping prevents overfitting on the relatively small dataset

### Training Environment

- **Platform:** Google Colab (T4 GPU, 16GB VRAM)
- **Training time:** ~2.5 hours for 100 epochs
- **Framework:** Ultralytics YOLOv8 8.x

> Full training notebook with loss curves and evaluation results:  
> [Google Colab Notebook](#) *(Replace with your Colab share link)*

---

## System Architecture

```
Input (image / video / webcam)
        │
        ▼
┌─────────────────────────┐
│   YOLOv8m Detection     │  → detects all instances of 5 classes
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Worker–PPE Association │  → IoU overlap + centroid fallback
│                         │    matches each PPE box to nearest worker
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Compliance Rule Engine │  → per-worker: checks hard_hat, safety_vest
│                         │    flags violations and uncertain predictions
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Violation Reporter     │  → structured JSON report + annotated frame
│                         │    verdict: SAFE / UNSAFE / *-UNCERTAIN
└─────────────────────────┘
        │
        ▼
Output: annotated image/video + JSON report + violations_log.csv
```

---

## Project Structure

```
construction-safety-monitor/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── dataset_cleaning.ipynb      # cleaning pipeline
│   └── README_dataset.md           # dataset documentation
│
├── train/
│   ├── construction_safety_training.ipynb   # full training notebook
│   └── data.yaml                            # class names + split paths
│
├── inference/
│   ├── inference.py                # CLI inference pipeline
│   └── zones.json                  # optional zone config
│
├── app.py                          # Streamlit web frontend
│
├── docs/
│   ├── safety_rules.md
│   ├── training_curves.png
│   ├── per_class_metrics.png
│   └── inference_results.png
│
└── sample_images/                  # ready-to-run test images
    ├── safe_scene_1.jpg
    ├── violation_no_helmet.jpg
    └── violation_no_vest.jpg
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- pip

### Install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/construction-safety-monitor.git
cd construction-safety-monitor
pip install -r requirements.txt
```

### Download trained weights

Download `best.pt` from [Google Drive](#) and place it in the project root:

```
construction-safety-monitor/
└── best.pt    ← place here
```

---

## Running Inference

### Single image

```bash
python inference/inference.py --source sample_images/violation_no_helmet.jpg
```

### Folder of images

```bash
python inference/inference.py --source sample_images/
```

### Video file

```bash
python inference/inference.py --source site_video.mp4 --show
```

### Webcam (live)

```bash
python inference/inference.py --source 0 --show
# Press S to save snapshot   |   Press Q to quit
```

### All options

```
--source    Image / folder / video path / webcam id (required)
--weights   Path to best.pt (default: best.pt)
--conf      Confidence threshold (default: 0.35)
--iou       NMS IoU threshold (default: 0.45)
--device    cuda / cpu / 0 (auto-detected if empty)
--output    Output directory root (default: runs/inference)
--show      Show live window during video/webcam inference
```

### Output structure

```
runs/inference/<timestamp>/
├── annotated/          annotated images or video
├── reports/            per-image JSON violation reports
├── summary.json        aggregate stats
└── violations_log.csv  flat log of every violation
```

### Example JSON report

```json
{
  "source": "site_image.jpg",
  "verdict": "UNSAFE",
  "workers_total": 3,
  "workers_compliant": 1,
  "workers_violated": 2,
  "workers_uncertain": 0,
  "inference_ms": 42.3,
  "workers": [
    {
      "id": 2,
      "status": "VIOLATION",
      "has_hardhat": false,
      "has_vest": true,
      "violations": ["Missing hard hat"],
      "confidence": 0.81,
      "uncertain": false
    }
  ]
}
```

---

## Streamlit App

A full web interface supporting image, batch, and video inference.

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

**Features:**
- Upload `best.pt` directly in the browser sidebar
- Live confidence / IoU / uncertainty threshold sliders
- Image tab — single image with annotated result + per-worker cards
- Folder tab — batch inference with progress log + results gallery
- Video tab — frame-by-frame processing with live preview + download
- Webcam tab — instructions for live CLI mode + snapshot testing

---

## Evaluation & Results

> Results below are on the held-out **test set** (10% of dataset, never seen during training).

| Metric | Value |
|---|---|
| mAP@50 | *(fill in from your results.csv)* |
| mAP@50-95 | *(fill in)* |
| Precision | *(fill in)* |
| Recall | *(fill in)* |

### Per-class AP@50

| Class | AP@50 |
|---|---|
| person | *(fill in)* |
| hard_hat | *(fill in)* |
| no_hard_hat | *(fill in)* |
| safety_vest | *(fill in)* |
| no_safety_vest | *(fill in)* |

> Training curves, confusion matrix, and per-class metrics are in `docs/`.

### Where the model performs well

- Workers clearly visible and facing the camera at mid-range distance
- High-contrast PPE (bright yellow/orange vests, white/yellow helmets) in daylight
- Scenes with 1–4 workers — association logic works reliably
- Clear hard hat violations where the head is visible and unobstructed

### Where the model struggles (honest failure cases)

- **Distant workers** — workers more than ~15 metres from camera appear very small; detection confidence drops significantly
- **Partial occlusion** — worker partially behind equipment or other workers; PPE association may fail or produce uncertain flags
- **Low-light scenes** — nighttime or heavy shadow conditions underrepresented in training data; model was not trained on true nighttime imagery
- **Rear-facing workers** — vest is often visible but hard hat detection is harder from behind; may produce false `no_hard_hat` flags
- **Helmet colours** — white hard hats perform well; non-standard colours (red, blue) are less common in training data

---

## Design Decisions & Trade-offs

### Why positive + negative classes?

Rather than inferring absence ("no hard hat detected near worker"), the dataset explicitly labels both `hard_hat` and `no_hard_hat`. This gives the model a positive detection target for violations rather than relying on the absence of a positive class — which fails for occluded or distant workers where the PPE simply isn't visible.

### Why IoU + centroid association?

Hard hat and safety vest boxes often don't fully overlap the person box — a helmet box sits at the very top of a person box with limited overlap. Using IoU alone with a strict threshold misses many valid associations. The centroid fallback catches cases where the PPE centroid falls inside the worker box even when IoU is low.

### Why uncertainty flagging instead of a binary output?

A binary safe/unsafe output silently hides model uncertainty. Low-confidence predictions on occluded or distant workers are worse than flagging them for manual review. The `UNCERTAIN` suffix surfaces this to the operator rather than producing a false safety assurance.

### Why YOLOv8m over YOLOv8s or YOLOv8l?

- `yolov8s` (11M params) — faster but noticeably lower recall on small objects (distant workers)
- `yolov8m` (25M params) — sweet spot for this dataset size and task
- `yolov8l` (43M params) — marginal accuracy gain, 2× memory cost, impractical on T4 without reducing batch size significantly

### Data balancing trade-off

Undersampling `person` and `no_safety_vest` (from 3,500 → 2,200) slightly reduces the absolute number of training examples for these classes. The alternative — oversampling all weaker classes up to 3,500 — would significantly increase training time and risk the augmented `safety_vest` class diverging from real distribution. The balanced ~2,200 target keeps training time manageable while preventing dominant classes from overwhelming gradients.

---

## Known Limitations & Failure Cases

| Limitation | Impact | Potential fix |
|---|---|---|
| No temporal analysis | Cannot detect patterns across frames (e.g. worker removes helmet momentarily) | Add tracking (ByteTrack) + per-ID compliance window |
| No depth estimation | Cannot reliably enforce height-based rules (fall protection) | Add monocular depth estimation or stereo camera input |
| Fixed zone config | Zone polygons in `zones.json` are per-scene and must be manually defined | Zone auto-detection from scene segmentation |
| No re-ID across frames | Worker IDs reset each frame in video mode | Integrate SORT/DeepSORT tracking |
| Binary PPE state | Cannot detect partial PPE use (unfastened helmet, open vest) | Keypoint-based PPE alignment or instance segmentation |

---

## Creativity & Innovation

Beyond the baseline detection task, this submission includes:

**1. Worker–PPE Association Engine**
Rather than scene-level flags, the system associates each PPE detection with its nearest worker using IoU + centroid proximity, enabling per-worker compliance reports rather than scene-level binary verdicts.

**2. Confidence-based Uncertainty Scoring**
Predictions below a configurable threshold are explicitly flagged as `UNCERTAIN` rather than silently classified. This surfaces model doubt to the operator — a critical requirement for safety-critical applications where false negatives have real consequences.

**3. Zone-based Rule System**
A configurable `zones.json` allows defining polygonal zones with different PPE requirements. Workers are tested against the zone they occupy, enabling stricter rules in high-risk areas (scaffolding, machinery zones) vs standard areas.

**4. Human-readable Violation Reports**
Every inference run produces structured JSON reports and a flat `violations_log.csv` — not just annotated images. Each report identifies the specific worker, the specific violation, confidence score, and uncertainty flag.

**5. Streamlit Web Frontend**
An industrial-themed web interface supporting image, batch, video, and webcam inference — enabling non-technical safety officers to use the system without touching the CLI.

---

## Requirements

```
ultralytics>=8.0.0
streamlit>=1.32.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
roboflow>=1.1.0
albumentations>=1.3.0
imagehash>=4.3.1
pyyaml>=6.0
matplotlib>=3.7.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## Reproducing Training

1. Open `train/construction_safety_training.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (`Runtime > Change runtime type`)
3. Fill in your Roboflow API key in Section 2
4. Run all cells — training takes ~2.5 hours on T4

The notebook includes:
- Dataset download and class distribution plot
- Albumentations augmentation pipeline for class balancing
- YOLOv8m training with full hyperparameter config
- Test set evaluation with per-class metrics
- Training curves and per-class precision/recall bar charts
- ONNX export



---
