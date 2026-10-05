# CCTV Object Localization and Natural Language Recall

The demo samples CCTV video, detects objects with YOLOv8, assigns lightweight track IDs between sampled frames, embeds each detected crop with CLIP, and stores vectors plus event metadata in Qdrant. Streamlit searches those object crops and shows the matching original frame with its YOLO box.

## Requirements

- Python 3.10+
- Docker (or another reachable Qdrant instance)
- Network access on first run to download `yolov8n.pt` and the Hugging Face `openai/clip-vit-base-patch32` model
- No API key is needed for the default local Qdrant setup. For hosted Qdrant, set `QDRANT_URL` and `QDRANT_API_KEY` or add them to `.streamlit/secrets.toml`.

## Run locally

Start Qdrant from the project directory:

```bash
docker run -d --name cctv-qdrant -p 6333:6333 -p 6334:6334 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

Install and launch:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload a clip, select **Process video**, then search for an object or visual description. Detection confidence is the YOLO model score; CLIP/Qdrant similarity is the text-to-crop vector score. They measure different things.

## Indexed metadata

Each object record includes the source video path/name, timestamp in seconds, frame number, object label, YOLO detection confidence, `[x1, y1, x2, y2]` frame coordinates, and a video-local track ID. Tracking is a simple class-aware IoU association between sampled frames, intended for a lightweight demo rather than identity-grade tracking.
