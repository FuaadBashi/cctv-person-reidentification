# Single-Camera Person Tracking and Re-Identification

A computer-vision prototype combining YOLO detection, DeepSORT tracking, and appearance-based identity association. It writes annotated video and structured track/event logs for inspection.

## Run locally

Install dependencies in a fresh Python environment and provide a video you are authorized to process:

```bash
git clone https://github.com/FuaadBashi/cctv-person-reidentification.git
cd cctv-person-reidentification
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --input footage.mp4 --output_dir results --no_face
```

The example disables the optional face module. See [run.py](run.py) for the complete argument list and [config.py](cctv_reid/config.py) for defaults. Model dependencies may download weights on first use.

## Architecture and source guide

| Stage | Source |
| --- | --- |
| Detection and small-person rescue | [detector.py](cctv_reid/detector.py) |
| Frame-to-frame tracking | [tracker.py](cctv_reid/tracker.py) |
| Body appearance features | [body_reid.py](cctv_reid/body_reid.py) |
| Identity gallery and assignment | [identity.py](cctv_reid/identity.py) |
| Orchestration and output | [pipeline.py](cctv_reid/pipeline.py) |

The pipeline produces `annotated.mp4`, `tracks.csv`, `tracks.jsonl`, and `events.jsonl`. Track rows include `track_id`, `global_id`, `assign_reason`, and `assign_dist`, connecting visible labels to assignment decisions.

## Configuration

`--conf` defaults to 0.35. `--iou` defaults to 0.5 and controls detector NMS, not the tracker association gate. Appearance thresholds are cosine **distances**: lower accepted distances require closer matches. Defaults are configuration choices, not validated operating points.

## Evaluation status

No labeled tracking benchmark, MOTA/IDF1 result, or measured throughput is established by this README. The repository does not currently include the evaluation script described in an earlier README. A reproducible evaluation needs a specified clip, ground-truth identities, matching protocol, hardware, and run configuration.

This is a single-camera prototype; occlusion, similar clothing, and scene changes can cause identity switches. Use consented or appropriately licensed footage and avoid publishing identifying footage or track logs.
