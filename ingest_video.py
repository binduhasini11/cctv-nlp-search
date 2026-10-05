"""Video ingestion: sampled CCTV frames -> YOLO detections -> CLIP object vectors."""
import os
import uuid
from pathlib import Path

import cv2
from qdrant_client.models import PointStruct
from ultralytics import YOLO

from embedder import get_image_embedding
from qdrant_store import COLLECTION_NAME, get_client, ensure_collection


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1)


def process_video_with_yolo(video_path, sample_rate_sec=1, confidence_threshold=0.4,
                            progress_callback=None):
    """Embed detected object crops; return an ingestion summary or raise a useful error."""
    video_path = str(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if sample_rate_sec <= 0:
        raise ValueError("sample_rate_sec must be greater than zero")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        raise ValueError("Video has no readable frame rate")

    interval = max(1, round(fps * sample_rate_sec))
    client = get_client()
    ensure_collection(client)
    model = YOLO("yolov8n.pt")
    temp_dir = Path("temp_crops")
    temp_dir.mkdir(exist_ok=True)
    # track association across sampled frames by class and IoU; IDs are local to this video run.
    tracks, next_track_id = [], 1
    points, frame_number, processed_frames = [], 0, 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_number % interval == 0:
                processed_frames += 1
                timestamp = frame_number / fps
                result = model(frame, verbose=False)[0]
                detections = []
                if result.boxes is not None:
                    for box in result.boxes:
                        coords = [int(round(v)) for v in box.xyxy[0].tolist()]
                        x1, y1, x2, y2 = coords
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        confidence = float(box.conf[0].item())
                        class_id = int(box.cls[0].item())
                        label = str(model.names[class_id])
                        if confidence < confidence_threshold or x2 <= x1 or y2 <= y1:
                            continue
                        detections.append((coords, label, confidence, (x1, y1, x2, y2)))

                assigned = []
                used = set()
                for coords, label, confidence, clipped in detections:
                    candidates = [(score, idx) for idx, (old_box, old_label, track_id) in enumerate(tracks)
                                  if idx not in used and old_label == label
                                  for score in [_iou(coords, old_box)] if score >= 0.25]
                    if candidates:
                        _, idx = max(candidates)
                        used.add(idx)
                        track_id = tracks[idx][2]
                    else:
                        track_id = next_track_id
                        next_track_id += 1
                    assigned.append((coords, label, confidence, clipped, track_id))
                tracks = [(d[0], d[1], d[4]) for d in assigned]

                for coords, label, confidence, (x1, y1, x2, y2), track_id in assigned:
                    point_id = str(uuid.uuid4())
                    crop_path = temp_dir / f"crop_{point_id}.jpg"
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0 or not cv2.imwrite(str(crop_path), crop):
                        continue
                    vector = get_image_embedding(str(crop_path))
                    points.append(PointStruct(id=point_id, vector=vector, payload={
                        "timestamp": timestamp, "frame_number": frame_number,
                        "video_source": Path(video_path).name, "video_path": os.path.abspath(video_path),
                        "object_label": label, "yolo_confidence": confidence,
                        "bbox": coords, "track_id": track_id,
                    }))
                if progress_callback:
                    progress_callback(processed_frames)
            frame_number += 1
    finally:
        cap.release()

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    return {"frames_processed": processed_frames, "objects_stored": len(points),
            "video_source": Path(video_path).name}


if __name__ == "__main__":
    print(process_video_with_yolo("Sample1.mp4"))
