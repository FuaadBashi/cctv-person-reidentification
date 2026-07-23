# CCTV Person Re-Identification — Single-Camera Tracking Prototype

Maintains stable person identities across frames and through occlusions by combining YOLO detection, DeepSORT motion tracking, and a PyTorch appearance-embedding gallery. Exports annotated video plus structured audit logs so every identity decision is inspectable after the fact.

The hard part of single-camera re-ID is not detection. It is what happens when someone walks behind a pillar for two seconds: DeepSORT's Kalman filter loses the track, and a naive pipeline assigns a fresh ID on re-emergence. Appearance embeddings are what close that gap.

---

## Architecture

```
frame ──► YOLO detection ──► DeepSORT (Kalman + IoU association)
                                   │
                          track confirmed?
                              │        │
                            yes        no / occluded
                              │        │
                              │        ▼
                              │   appearance embedding ──► gallery cosine match
                              │        │                        │
                              │        │              above threshold? reuse ID
                              ▼        ▼
                        ┌─────────────────────┐
                        │  annotated video    │
                        │  tracks.csv         │
                        │  events.jsonl       │
                        └─────────────────────┘
```

**Detection.** YOLO, confidence-thresholded.

**Motion association.** DeepSORT: Kalman prediction plus IoU matching handles the frame-to-frame case cheaply. This is sufficient while a person remains continuously visible.

**Appearance re-association.** A PyTorch embedding model maintains a gallery of feature vectors per identity. When a track is lost and a new detection appears, cosine similarity against the gallery decides whether it is a returning person or a genuinely new one. This is the component that survives occlusion; motion prediction alone does not, because a Kalman filter extrapolating through a two-second gap has accumulated too much positional uncertainty to associate reliably on IoU.

**Audit trail.** Every association decision is logged with the similarity score that produced it, so an ID switch can be traced to a specific frame and threshold decision rather than guessed at.

## Configuration

The pipeline exposes three thresholds. The defaults below are the CLI's starting values — **replace with the values you actually settled on**:

| Parameter | Default | Controls |
|---|---|---|
| `--conf` | 0.5 | YOLO detection confidence floor |
| `--iou` | 0.45 | IoU gate for DeepSORT frame-to-frame association |
| `--reid-threshold` | 0.7 | Cosine similarity required to reuse an identity from the gallery |

### The ReID threshold is the real tuning knob

It trades the two failure modes directly against each other:

| Threshold | Failure mode | What you see |
|---|---|---|
| Too low | Identity **merges** | Two different people collapse into one ID |
| Too high | Identity **fragments** | One person picks up a new ID after every occlusion |

There is no universally correct value — it depends on how visually distinct the people in your footage are. Crowds in similar clothing push it up; sparse scenes with varied appearance let it come down.

`[FILL: the value you settled on and what you observed on either side of it. Even qualitative observation is worth writing — "at 0.5 two people in dark jackets merged; at 0.85 the same person picked up three IDs across one occlusion" demonstrates deliberate tuning, which is the point. This is the highest-value paragraph in this README.]`

## Evaluation

> **No tracking metrics are reported here yet.** MOTA, IDF1, and ID-switch counts are defined only against frame-by-frame annotated ground truth — they are properties of this tracker's output *compared to labelled footage*, not properties of the tracker. Numbers will be added once a labelled clip exists.

`scripts/evaluate_tracking.py` computes them. Getting to real numbers takes about one evening:

1. **Label 30–60 seconds, not the whole video.** One minute at 25fps with four or five people is enough for a defensible number.
2. **Use CVAT or Label Studio** (both free). CVAT interpolates between keyframes, so you annotate roughly every tenth frame and it fills the gaps.
3. **Export as MOT 1.1 format.** It drops straight into the script.
4. Run it:

```bash
pip install motmetrics pandas
python scripts/evaluate_tracking.py --pred results/tracks.csv --gt data/gt.txt
```

The script prints a paste-ready markdown table. **Report the numbers alongside the clip they came from** — duration, resolution, identity count. "MOTA 0.74 on a 60-second five-person indoor clip" is a defensible claim. "MOTA 0.74" alone is not, because the first question anyone asks is which sequence.

### Throughput

Frames per second needs no ground truth — time the pipeline over a known clip and divide. Report it with the hardware, since FPS without a GPU model is meaningless:

`[FILL: e.g. "18 FPS at 1080p on an RTX 3060" — measure this today, it takes five minutes]`

## Usage

```bash
pip install -r requirements.txt

python track.py \
    --source footage.mp4 \
    --conf 0.5 \
    --iou 0.45 \
    --reid-threshold 0.7 \
    --out results/
```

`[ADJUST to your actual CLI flags.]`

### Outputs

**`tracks.csv`** — one row per detection per frame:

```csv
frame,track_id,x1,y1,x2,y2,confidence,reid_similarity
1,1,320,140,410,480,0.94,
48,1,512,138,604,479,0.91,0.83
```

The `reid_similarity` column is populated only on re-association events, so a non-empty value marks every frame where the appearance gallery — rather than motion prediction — decided the identity.

**`events.jsonl`** — one record per identity lifecycle event:

```json
{"frame": 48, "event": "reid_match", "track_id": 1, "similarity": 0.83, "gap_frames": 19}
{"frame": 92, "event": "track_lost", "track_id": 3, "last_seen": 91}
```

## Repo layout

```
├── track.py                      # main CLI pipeline
├── src/
│   ├── detector.py               # YOLO wrapper
│   ├── tracker.py                # DeepSORT integration
│   └── reid.py                   # embedding model + gallery matching
├── scripts/
│   └── evaluate_tracking.py      # MOTA / IDF1 / ID switches vs ground truth
├── results/                      # sample outputs — commit a short clip's logs
└── requirements.txt
```

## Test footage

`[FILL: source, duration, resolution, approximate person count. If you used a public benchmark clip such as MOT17 or PETS2009, name it — public footage is better than self-recorded here, because it makes your numbers comparable to published baselines and sidesteps the consent question entirely.]`

## Limitations

- **Single camera only.** No cross-camera re-identification; the gallery is not shared across views, and appearance embeddings do not transfer across differing lighting and camera geometry without domain adaptation.
- Appearance matching degrades when people wear similar clothing — the embedding has little to separate two people in the same uniform, which is the common case in exactly the environments CCTV covers.
- Prototype throughput, not optimised for real-time deployment at scale.
- Unevaluated on crowded scenes with heavy mutual occlusion.

## Ethical and legal note

Person re-identification is surveillance technology. Deployment against real people is regulated — in the UAE by Federal Decree-Law No. 45 of 2021 on Personal Data Protection, and in the EU by GDPR Article 9, which treats biometric data used for unique identification as a special category requiring an explicit lawful basis.

This is a technical prototype built on `[FILL: public benchmark / self-recorded / synthetic]` footage. It is not intended for deployment against real individuals without a lawful basis, a data protection impact assessment, and defined retention limits.

## Licence

MIT — see [LICENSE](LICENSE).
