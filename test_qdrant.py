"""Manual connectivity check; run with `python test_qdrant.py`."""
from qdrant_store import get_client

if __name__ == "__main__":
    client = get_client()
    print("Connected to Qdrant. Collections:", client.get_collections())
