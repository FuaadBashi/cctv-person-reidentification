# CCTV Person Re-Identification — Single-Camera Tracking Prototype

Maintains stable person identities across frames and through occlusions by combining YOLO detection, DeepSORT motion tracking, and a PyTorch appearance-embedding gallery. Exports annotated video plus structured audit logs so that every identity decision is inspectable after the fact.

The hard part of single-camera re-ID is not detection. It is what happens when a person walks behind a pillar for two seconds: DeepSORT's Kalman filter loses the track, and a naive pipeline assigns a fresh ID on re-emergence. Appearance embeddings are what close that gap.

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

- **Detection** — YOLO, confidence-thresholded.
- **Motion association** — DeepSORT: Kalman prediction plus IoU matching handles the frame-to-frame case cheaply.
- **Appearance re-association** — a PyTorch embedding model maintains a gallery of feature vectors per identity. When a track is lost and a new detection appears, cosine similarity against the gallery decides whether it is a returning person or a genuinely new one.
- **Audit trail** — every association decision is logged, so an ID switch can be traced to the frame and the similarity score that caused it.

## Results

| Metric | Value |
|---|---|
| ID switches | `[FILL]` |
| MOTA | `[FILL]` |
| IDF1 | `[FILL]` |
| Tracking FPS | `[FILL]` |
| Test footage | `[FILL: source, duration, resolution, approx. person count]` |

`[If you have not computed MOTA/IDF1, either run py-motmetrics against a labelled clip or delete this table entirely and describe the qualitative behaviour instead. An empty metrics table is worse than no table — and an unlabelled number is worse than both.]`

### Threshold sensitivity

The ReID similarity threshold is the main tuning knob and it trades the two failure modes against each other:

| Threshold | Failure mode |
|---|---|
| Too low | Identity **merges** — two different people collapse into one ID |
| Too high | Identity **fragments** — one person picks up a new ID after every occlusion |

`[FILL: the value you settled on and what you observed on either side of it. This is the single most interview-relevant paragraph in the repo — it shows you tuned deliberately rather than accepting a default.]`

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

**`events.jsonl`** — one record per identity lifecycle event:

```json
{"frame": 48, "event": "reid_match", "track_id": 1, "similarity": 0.83, "gap_frames": 19}
{"frame": 92, "event": "track_lost", "track_id": 3, "last_seen": 91}
```

## Repo layout

```
├── track.py              # main CLI pipeline
├── src/
│   ├── detector.py       # YOLO wrapper
│   ├── tracker.py        # DeepSORT integration
│   └── reid.py           # embedding model + gallery matching
├── results/              # sample outputs — commit a short clip's logs
└── requirements.txt
```

## Limitations

- **Single camera only.** No cross-camera re-identification; the gallery is not shared across views and appearance embeddings do not transfer across differing lighting and camera geometry without domain adaptation.
- Appearance matching degrades when people wear similar clothing — the embedding has little to separate two people in the same uniform.
- Prototype throughput, not optimised for real-time deployment at scale.
- Evaluated on `[FILL]` footage only; behaviour on crowded scenes with heavy mutual occlusion is untested.

## Ethical and legal note

Person re-identification is surveillance technology. Deployment against real people is regulated — in the UAE by Federal Decree-Law No. 45 of 2021 on Personal Data Protection, in the EU by GDPR Article 9, which treats biometric data used for unique identification as a special category requiring an explicit lawful basis.

This is a technical prototype built on `[FILL: public benchmark / self-recorded / synthetic]` footage. It is not intended for deployment against real individuals without a lawful basis, a data protection impact assessment, and appropriate retention limits.

## Licence

MIT — see [LICENSE](LICENSE).
