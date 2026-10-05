"""Shared Qdrant configuration for local development and Streamlit deployment."""
import os

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "cctv_frames"


def get_client():
    """Use Streamlit secrets or environment variables; default to local Qdrant."""
    try:
        url = st.secrets.get("QDRANT_URL")
        api_key = st.secrets.get("QDRANT_API_KEY")
    except Exception:
        url = api_key = None
    url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = api_key or os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(client):
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(COLLECTION_NAME, vectors_config=VectorParams(size=512, distance=Distance.COSINE))
