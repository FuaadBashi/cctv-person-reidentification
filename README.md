# CCTV Person Tracking and Re-Identification

[![CI](https://github.com/FuaadBashi/cctv-person-reidentification/actions/workflows/ci.yml/badge.svg)](https://github.com/FuaadBashi/cctv-person-reidentification/actions/workflows/ci.yml)

A computer-vision pipeline that follows people through CCTV footage and keeps a **stable identity
for each person**, even when the tracker loses them behind an obstacle and picks them up again
under a new track ID. It writes an annotated video and structured logs explaining every identity
decision.

## Pipeline

```
frame ─▶ YOLOv8 person detection ─▶ DeepSORT tracking ─▶ face + body embeddings ─▶ identity gallery
            │ (small-person rescue:        (track IDs,        (InsightFace,            (global IDs,
            │  higher resolution and         frame to frame)     ResNet-50)               cosine matching)
            ▼  overlapping tiles + NMS)
```

| Stage | What it does | Source |
| --- | --- | --- |
| Detection | YOLOv8 finds people. When a frame yields too few, a rescue pass re-runs at higher resolution over overlapping tiles and merges the results with NMS, which catches distant, small figures. | [detector.py](cctv_reid/detector.py) |
| Tracking | DeepSORT links detections across frames into short-lived track IDs. | [tracker.py](cctv_reid/tracker.py) |
| Appearance | InsightFace face embeddings when a face is visible; a ResNet-50 body embedding otherwise. | [face.py](cctv_reid/face.py), [body_reid.py](cctv_reid/body_reid.py) |
| Identity | Each new track is matched against a gallery of known people by cosine distance: face first, then body. The body path uses a best-versus-second-best margin, so an ambiguous match creates a new identity instead of merging two people. | [identity.py](cctv_reid/identity.py) |
| Output | `annotated.mp4`, plus `tracks.csv`/`tracks.jsonl` with the reason and distance behind every assignment, and `events.jsonl`. | [pipeline.py](cctv_reid/pipeline.py) |

## Getting started

Requires Python 3.10+. A GPU helps but isn't required.

```bash
git clone https://github.com/FuaadBashi/cctv-person-reidentification.git
cd cctv-person-reidentification
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py --input footage.mp4 --output_dir results
```

The YOLOv8 weights (`yolov8n.pt` by default) download automatically on first run. Use
`--no_face` or `--no_body` to disable a modality, and `--show` for a live preview.
`python run.py --help` lists every option.

## Tuning

| Option | Default | Effect |
| --- | --- | --- |
| `--conf` | 0.35 | Detection confidence for the normal pass |
| `--small_conf` / `--small_imgsz` | 0.15 / 1536 | Rescue-pass sensitivity and resolution |
| `--reid_face_thresh` | 0.35 | Maximum cosine *distance* for a face match (lower = stricter) |
| `--reid_body_thresh` | 0.45 | Maximum cosine distance for a body match |
| `--face_every` / `--body_every` | 3 / 2 | Compute embeddings every N frames, trading accuracy for speed |

These defaults were tuned by eye on sample footage and have not been benchmarked. A proper
evaluation would score ID switches and IDF1 on a labelled clip.

## Tests

```bash
pip install pytest ruff numpy
pytest
ruff format --check . && ruff check .
```

The tests cover the parts with no model dependency: IoU, non-maximum suppression, full-frame
tile coverage, and the identity rules (new IDs, face-over-body priority, the ambiguity margin,
stable IDs per track, bounded galleries).

Use only footage you are authorised to process, and don't publish identifying video or logs.
