"""Command-line helper for searching all indexed CCTV object crops."""
from embedder import get_text_embedding
from qdrant_store import COLLECTION_NAME, get_client


def search_video(query_text, top_k=3, video_path=None):
    if not query_text or not query_text.strip():
        raise ValueError("query_text cannot be empty")
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        print("No indexed CCTV collection exists yet.")
        return []
    kwargs = {"collection_name": COLLECTION_NAME, "query": get_text_embedding(query_text.strip()),
              "limit": top_k, "with_payload": True}
    if video_path:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        import os
        kwargs["query_filter"] = Filter(must=[FieldCondition(key="video_path", match=MatchValue(value=os.path.abspath(video_path)))])
    hits = client.query_points(**kwargs).points
    if not hits:
        print("No matching objects found.")
        return []
    for i, hit in enumerate(hits, 1):
        payload = hit.payload or {}
        print(f"#{i} {payload.get('object_label', 'unknown')} | time {payload.get('timestamp', 0):.2f}s | "
              f"frame {payload.get('frame_number', '—')} | track {payload.get('track_id', '—')} | "
              f"YOLO confidence {payload.get('yolo_confidence', 'unavailable')} | CLIP/Qdrant similarity {hit.score:.3f}")
    return hits


if __name__ == "__main__":
    search_video("a person carrying a backpack")
