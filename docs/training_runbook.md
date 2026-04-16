# Training runbook — Roboflow dataset → Colab → `best.pt`

Goal: produce `models/best.pt` (4 classes: player/GK/ref/ball) to plug into the pipeline in `--mode custom`. Training runs on Colab's free T4 GPU; your laptop is not involved.

## 1. Get a Roboflow API key (free)

1. Sign up at https://app.roboflow.com (free personal tier — no credit card)
2. Go to **Settings → API** and copy your private API key.
3. Open the dataset page: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
4. Click **Dataset → Download Dataset → YOLOv8** — note the **Version** number (typically 12 at time of writing).

You don't need to download anything on your laptop; Colab pulls it directly.

## 2. Open the Colab notebook

- Open https://colab.research.google.com → **File → Upload notebook** → pick `notebooks/train_yolo_colab.ipynb` from this repo
- `Runtime → Change runtime type → T4 GPU` → Save
- `Runtime → Run all`, but **stop before cell 3**.

## 3. Fill in cell 3 (Option A — Roboflow)

Comment out Option B (the `files.upload()` cell). Uncomment the Option A block and edit:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="PASTE_YOUR_KEY_HERE")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(12)           # confirm the number from step 1
dataset = version.download("yolov8")
DATA_YAML = f"{dataset.location}/data.yaml"
```

Also run: `!pip install -q roboflow` earlier in the notebook (before this cell).

## 4. Verify class order

After download, cell 4 prints `data.yaml`. It **must** show these classes in this exact order:

```yaml
names:
  - player
  - goalkeeper
  - referee
  - ball
```

If the order differs, stop and tell me — we'll fix the mapping rather than training a mis-labeled model.

*(Roboflow sometimes uses slightly different names like "ball" vs "soccer ball" — names can differ, but the **index order** is what matters for our ClassMap.custom().)*

## 5. Train

Cell 5 runs the training. Expected on free T4:
- `yolov8m` + `imgsz=1280` + `batch=8` + 50 epochs ≈ 30–50 min
- Watch `runs/football/yolov8m-ft/results.png` — val loss should be flattening by epoch 40
- If Colab disconnects mid-run, just re-run cell 5; ultralytics resumes from last checkpoint

## 6. Download and install weights

Cell 7 downloads `best.pt`. Save it into the repo at:

```
C:/Users/Tadas/Documents/Football_analysis_tool/models/best.pt
```

Then on your laptop:

```bash
cd "C:/Users/Tadas/Documents/Football_analysis_tool"
source .venv/Scripts/activate
python main.py --input "video_clips/<any clip>.mp4" \
               --output output_videos/custom.mp4 \
               --model models/best.pt --mode custom
```

## 7. (Later) Fine-tune on our own frames

When you want to train on Tadas's own clips:
1. Run `python -m src.sampling.extract_frames` → `data/raw_frames/`
2. Label in Label Studio (see `docs/annotation_runbook.md` — coming in Phase 2b)
3. Export YOLO format → zip → upload as Option B in the same notebook, but pass `model=YOLO("models/best.pt")` instead of `"yolov8m.pt"` as the starting point. That's a **fine-tune of your fine-tune**, preserving what the Roboflow dataset taught it.
