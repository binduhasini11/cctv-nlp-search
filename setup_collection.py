from qdrant_client.models import PayloadSchemaType

from qdrant_store import COLLECTION_NAME, ensure_collection, get_client


if __name__ == "__main__":
    client = get_client()

    # Create the collection if it does not already exist
    ensure_collection(client)

    # Create an index for video_path so filtered searches work
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="video_path",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print("Payload index 'video_path' created.")
    except Exception as e:
        # The index may already exist
        if "already exists" in str(e).lower():
            print("Payload index 'video_path' already exists.")
        else:
            raise

    print(f"Qdrant collection '{COLLECTION_NAME}' is ready.")