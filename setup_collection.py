from qdrant_store import COLLECTION_NAME, ensure_collection, get_client

if __name__ == "__main__":
    client = get_client()
    ensure_collection(client)
    print(f"Qdrant collection '{COLLECTION_NAME}' is ready.")
