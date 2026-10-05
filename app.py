import logging
import os
import tempfile

import cv2
import streamlit as st
from qdrant_client.models import FieldCondition, Filter, MatchValue

from embedder import get_text_embedding
from ingest_video import process_video_with_yolo
from qdrant_store import COLLECTION_NAME, get_client

logging.getLogger("transformers").setLevel(logging.ERROR)
st.set_page_config(page_title="CCTV Object Search", layout="wide")
st.title("📹 CCTV Object Search")
st.caption("Index detected object crops, then find them with natural language.")

client = get_client()
uploaded_file = st.file_uploader("Upload CCTV video", type=["mp4", "avi", "mov", "mkv"])
if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    if st.session_state.get("uploaded_video_name") != uploaded_file.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded_file.getbuffer())
            st.session_state.uploaded_video_path = f.name
        st.session_state.uploaded_video_name = uploaded_file.name
        st.session_state.video_processed = False
    st.success(f"Ready: {uploaded_file.name}")

    if st.button("Process video", type="primary"):
        try:
            with st.spinner("Detecting and indexing objects…"):
                summary = process_video_with_yolo(st.session_state.uploaded_video_path)
            st.session_state.video_processed = True
            st.success(f"Indexed {summary['objects_stored']} objects across {summary['frames_processed']} sampled frames.")
            if not summary["objects_stored"]:
                st.info("No objects passed the detection threshold. Try another video or lower the threshold in ingestion settings.")
        except Exception as exc:
            st.session_state.video_processed = False
            st.error(f"Video processing failed: {exc}")

query = st.text_input("Describe the object", placeholder="a person carrying a backpack")
top_k = st.slider("Results", 1, 10, 5)
if st.button("Search", type="primary"):
    if uploaded_file is None or not st.session_state.get("video_processed", False):
        st.warning("Upload and process a video before searching.")
    elif not query.strip():
        st.warning("Enter a description to search.")
    else:
        try:
            with st.spinner("Searching detected object crops…"):
                response = client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=get_text_embedding(query.strip()), limit=top_k,
                    query_filter=Filter(must=[FieldCondition(key="video_path", match=MatchValue(value=os.path.abspath(st.session_state.uploaded_video_path)))]),
                    with_payload=True,
                )
            hits = response.points
            if not hits:
                st.info("No matching detected objects found in this video. Try a broader description.")
            else:
                st.subheader(f"{len(hits)} best matches")
                cols = st.columns(min(3, len(hits)))
                for i, hit in enumerate(hits):
                    payload = hit.payload or {}
                    with cols[i % len(cols)]:
                        st.markdown(f"**{payload.get('object_label', 'Unknown object')}** · Track {payload.get('track_id', '—')}")
                        path = payload.get("video_path") or st.session_state.get("uploaded_video_path")
                        frame_no = payload.get("frame_number")
                        bbox = payload.get("bbox")
                        if not path or not os.path.isfile(path):
                            st.warning("Source video is unavailable; frame cannot be displayed.")
                        elif frame_no is None:
                            st.warning("Frame number is missing from this result.")
                        elif not bbox or len(bbox) != 4:
                            st.warning("Bounding-box metadata is missing or invalid.")
                        else:
                            cap = cv2.VideoCapture(path)
                            frame = None
                            if cap.isOpened():
                                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
                                ok, bgr = cap.read()
                                if ok:
                                    x1, y1, x2, y2 = [int(v) for v in bbox]
                                    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                    label = f"{payload.get('object_label', 'object')} · YOLO {float(payload.get('yolo_confidence', 0)):.2f}"
                                    cv2.putText(bgr, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                                    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                            cap.release()
                            if frame is None:
                                st.warning(f"Could not retrieve frame {frame_no} from the source video.")
                            else:
                                st.image(frame, use_container_width=True)
                        st.write(f"Timestamp: {float(payload.get('timestamp', 0)):.2f}s · Frame: {payload.get('frame_number', '—')}")
                        yolo_conf = payload.get("yolo_confidence")
                        st.write(f"YOLO detection confidence: {float(yolo_conf):.3f}" if yolo_conf is not None else "YOLO detection confidence: unavailable")
                        st.write(f"CLIP/Qdrant similarity: {float(hit.score):.3f}")
        except Exception as exc:
            st.error(f"Search failed. Check that Qdrant is running and the collection is available: {exc}")
