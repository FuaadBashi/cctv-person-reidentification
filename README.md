# CCTV Person Re-Identification Prototype (Single Camera)

Prototype that tracks people in a single CCTV feed and re-identifies them across exits/re-entry.
Face embeddings are used when visible; the system falls back to body/appearance embeddings when the face is not visible.

## Outputs
Each run writes to outputs/<run_name>/:
- annotated.mp4
- events.jsonl
- tracks.csv

Re-ID proof: the same global_id appears across different track_id values after EXIT/ENTER events (see events.jsonl).
