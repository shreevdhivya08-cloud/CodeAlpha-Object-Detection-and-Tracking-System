# 🎯 YOLOv8 Object Detection & Tracking

Real-time object detection and tracking from your webcam (or any video file) using **YOLOv8** and **OpenCV**.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📷 Webcam / video input | Any OpenCV-compatible source |
| 🟩 Bounding boxes | Colour-coded per class |
| 🏷️ Labels + confidence | e.g. `#3 person 94%` |
| 🔢 Tracking IDs | Persistent IDs via ByteTrack |
| ⚡ FPS display | Live frames-per-second overlay |
| 💾 Save output video | Writes an `output.mp4` automatically |

---

## 🗂️ Project Structure

```
├── detect_track.py   ← main script
├── requirements.txt  ← Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1 — Clone / download the project

```bash
git clone https://github.com/your-username/yolov8-tracker.git
cd yolov8-tracker
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** YOLOv8 weights (`yolov8n.pt`) are downloaded automatically on the first run (~6 MB).

### 4 — Run it!

```bash
python detect_track.py
```

Press **Q** to quit. The output video is saved as `output.mp4`.

---

## ⚙️ Command-line Options

```
usage: detect_track.py [-h] [--source SOURCE] [--model MODEL]
                       [--conf CONF] [--iou IOU] [--save SAVE]
                       [--no-show] [--no-track]

options:
  --source   Camera index (0, 1, …) or path/URL to video  [default: 0]
  --model    YOLOv8 weights file                           [default: yolov8n.pt]
  --conf     Confidence threshold 0–1                      [default: 0.40]
  --iou      NMS IoU threshold 0–1                         [default: 0.45]
  --save     Output video file path                        [default: output.mp4]
  --no-show  Disable the preview window (headless / SSH)
  --no-track Disable ByteTrack; run plain detection only
```

### Examples

```bash
# Use a video file instead of webcam
python detect_track.py --source my_video.mp4

# Use a larger (more accurate) model
python detect_track.py --model yolov8s.pt

# Higher confidence, no saving
python detect_track.py --conf 0.60 --save ""

# Headless mode (no window) — useful on servers
python detect_track.py --no-show

# Plain detection without tracking IDs
python detect_track.py --no-track
```

---

## 🤖 Model Sizes

Pick the right trade-off for your hardware:

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `yolov8n.pt` | 6 MB | ⚡⚡⚡ Fastest | ★★☆ |
| `yolov8s.pt` | 22 MB | ⚡⚡ Fast | ★★★ |
| `yolov8m.pt` | 52 MB | ⚡ Medium | ★★★★ |
| `yolov8l.pt` | 87 MB | 🐢 Slower | ★★★★★ |
| `yolov8x.pt` | 131 MB | 🐢🐢 Slowest | ★★★★★★ |

Start with `yolov8n.pt` (the default) — it runs well on a CPU.

---

## 🖥️ Requirements

- Python **3.8 – 3.12**
- A webcam **or** a video file
- No GPU required (CPU is fine for the `n` model)

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot open video source: 0` | Try `--source 1` (second camera) |
| Very low FPS on CPU | Use `--model yolov8n.pt` and/or lower resolution |
| Window doesn't open on server | Add `--no-show` |
| `ModuleNotFoundError` | Make sure your venv is activated and you ran `pip install -r requirements.txt` |

---

## 📄 License

MIT — free to use, modify, and distribute.
