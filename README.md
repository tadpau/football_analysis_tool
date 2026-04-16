# Football Analysis Tool

YOLOv8-based football match analysis pipeline: player/ball detection, tracking, team assignment, ball possession, and pitch-normalized distance/speed metrics.

## Project phases

| # | Phase | Status |
|---|---|---|
| 1 | Environment + repo scaffolding | **in progress** |
| 2 | Frame sampling + annotation (Label Studio) | pending |
| 3 | YOLOv8 fine-tuning (Google Colab) | pending |
| 4 | Tracker (ByteTrack) + kit-color team assignment | pending |
| 5 | Ball interpolation + possession logic | pending |
| 6 | Camera motion compensation | pending |
| 7 | Pitch keypoints + homography + speed/distance | pending |

## Hardware strategy

- **Laptop (Intel Iris Xe, 8 GB VRAM)** — inference, annotation, running the full pipeline on clips. OpenVINO optional for Iris Xe acceleration.
- **Google Colab / Kaggle (NVIDIA T4)** — all YOLO training runs. Notebook provided in `notebooks/`.

## Setup (one-time)

### 1. Install Python 3.12
Only Python 3.14 is currently on this machine; PyTorch/Ultralytics do not yet support it.

- Download Python 3.12 from https://www.python.org/downloads/windows/ (pick "Windows installer 64-bit")
- During install: tick **"Add python.exe to PATH"**
- Verify: `py -3.12 --version` should print `Python 3.12.x`

### 2. Create venv + install deps

```bash
cd "C:/Users/Tadas/Documents/Football_analysis_tool"
py -3.12 -m venv .venv
source .venv/Scripts/activate        # Git Bash
# or: .venv\Scripts\activate         # cmd/PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

## Folder layout

```
Football_analysis_tool/
├── video_clips/                # source match clips (10× ~2 min)
├── data/
│   ├── raw_frames/             # frames sampled for annotation
│   ├── annotations/            # Label Studio JSON exports
│   └── dataset/                # YOLO-format train/val/test (generated)
├── models/                     # fine-tuned .pt weights
├── output_videos/              # rendered analysis overlays
├── stubs/                      # cached tracker outputs (dev speedup)
├── notebooks/                  # Colab training notebook
├── src/
│   ├── utils/                  # video + bbox helpers
│   ├── sampling/               # frame extraction for annotation
│   ├── trackers/               # YOLO + ByteTrack
│   ├── team_assigner/          # shirt-color team clustering
│   ├── ball/                   # ball interpolation
│   ├── possession/             # possession assignment
│   ├── camera_movement/        # optical-flow camera compensation
│   ├── pitch/                  # keypoints + homography
│   └── speed_distance/         # world-coord metrics
└── main.py                     # pipeline orchestrator (TBD)
```
