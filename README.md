# CCTV Person Re-Identification Prototype (Single Camera)

A practical, end-to-end **single-camera** computer vision pipeline that performs **person detection**, **multi-object tracking**, and **re-identification (ReID)** using appearance embeddings to maintain identity continuity over time.

The system produces an **annotated output video** plus **structured run artifacts** (CSV/JSONL) designed for debugging, auditing, and downstream analytics.

> **Focus:** Single-camera identity continuity (not cross-camera identity linking).

---

## 🚀 Why This Project Matters

Recruiters often see computer vision work as "a notebook demo." This project is packaged as an engineering prototype:

* **CLI-driven execution** suitable for batch runs and reproducibility.
* **Structured logs** (`tracks.csv`, `events.jsonl`) for Detections + Tracks + Events.
* **Annotated video output** for quick visual verification.
* **Configuration-friendly** thresholds for detection, tracking, and ReID behavior.

---

## ✨ Key Features

* **Person Detection:** Powered by the YOLO-family of detectors.
* **Robust Tracking:** DeepSORT-style tracker for stable IDs across frames.
* **Re-identification (ReID):** Uses appearance embeddings + cosine similarity matching to reduce ID switches.
* **Structured Telemetry:**
* `tracks.csv`: Frame-level tracking table for Pandas/Excel analysis.
* `events.jsonl`: Identity lifecycle events (appear, disappear, merge).


* **Visualizations:** Annotated video export with bounding boxes and track IDs.
* **Developer Friendly:** Optional live preview (`--show`) and repeatable CLI runs.

---

## 🏗 High-Level Architecture

1. **Detector:** Finds people per frame (bounding boxes + confidence).
2. **Tracker:** Links detections over time using motion (Kalman Filter) and appearance cues.
3. **ReID:** Computes embeddings to mitigate ID switches during occlusions.
4. **Exporter:** Writes annotated video, track tables, and event streams.

---

## 📁 Repository Structure

```text
.
├── run.py                # Main entry point
├── requirements.txt      # Project dependencies
├── README.md
├── configs/
│   └── default.yaml      # Thresholds and model parameters
├── src/
│   ├── detection.py      # YOLO detector wrapper
│   ├── tracking.py       # Multi-object tracking logic
│   ├── reid.py           # Feature extraction & similarity
│   ├── io_utils.py       # CSV/JSONL logging helpers
│   └── viz.py            # OpenCV drawing utilities
└── outputs/              # Generated artifacts per run
    └── <run_name>/
        ├── annotated.mp4
        ├── tracks.csv
        └── events.jsonl

```

---

## 🛠 Getting Started

### Prerequisites

* Python 3.9+
* macOS, Linux, or Windows
* (Optional) CUDA-enabled GPU for faster inference
* [FFmpeg](https://ffmpeg.org/) for robust video encoding

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cctv-reid-prototype.git
cd cctv-reid-prototype

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

```

---

## 🚦 Usage

### Quick Start

Run the pipeline on a sample video with a live preview:

```bash
python run.py --input video.mp4 --output_dir outputs/run_01 --show

```

### Common CLI Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--input` | Path to input video file | None |
| `--output_dir` | Destination for artifacts | `outputs/` |
| `--conf` | Detector confidence threshold | `0.40` |
| `--reid_thresh` | Similarity threshold for ReID matching | `0.60` |
| `--max_age` | Frames to keep a "lost" track active | `30` |
| `--device` | Inference hardware (`cpu` or `cuda`) | `cpu` |

---

## 📊 Outputs & Telemetry

Inside your chosen output directory, the pipeline produces:

**1. `annotated.mp4**`
Rendered video with bounding boxes, track IDs, and state indicators.

**2. `tracks.csv**`
A frame-by-frame data table. Perfect for downstream data science tasks.

* **Columns:** `frame`, `track_id`, `x1`, `y1`, `x2`, `y2`, `det_conf`, `state`.

**3. `events.jsonl**`
A JSON Lines event stream for auditing identity lifecycle changes.

```json
{"t": 12.40, "event": "track_started", "track_id": 7, "bbox": [100, 200, 150, 300], "conf": 0.86}
{"t": 15.10, "event": "id_switch_mitigated", "old_id": 7, "new_id": 9}

```

---

## 🧠 How ReID Works

The ReID component utilizes a lightweight convolutional neural network to produce **appearance embeddings**. For each confirmed track, the system maintains a gallery of features. When a track is lost and a new detection appears nearby, the system calculates the **cosine similarity** between the new detection and the gallery to decide if it is a returning individual, significantly reducing fragmentation caused by short-term occlusions.

---

## 🛡 Privacy & Responsible Use

This prototype is intended for legitimate security and operations use cases.

* Avoid storing unnecessary personal data.
* Ensure compliance with local privacy and surveillance regulations (e.g., GDPR).
* Follow strict access controls for generated output artifacts.

---

## 🛠 Tech Stack

* **Language:** Python
* **CV Library:** OpenCV
* **Deep Learning:** PyTorch
* **Models:** YOLO (Detection), DeepSORT/ByteTrack (Tracking)
* **Data Science:** NumPy, Pandas

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
