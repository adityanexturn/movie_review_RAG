import weaviate
from weaviate import connect_to_local

# Connect to the local Weaviate instance
client = connect_to_local()

collection_name = "Chunk"

try:
    if not client.is_ready():
        raise RuntimeError("Weaviate is not ready. Please make sure Docker is running.")

    if client.collections.exists(collection_name):
        chunk_collection = client.collections.get(collection_name)

        confirm = input(f"This will delete ALL objects in the '{collection_name}' collection. Are you sure? (y/n): ").strip().lower()
        
        if confirm != "y":
            print("Cancelled. No data was deleted.")
        else:
            # Fetch all objects - UUID is automatically included
            objects = chunk_collection.query.fetch_objects(
                limit=10000
            )

            total = 0
            for obj in objects.objects:
                uuid = obj.uuid  # Access UUID directly from the object
                chunk_collection.data.delete_by_id(uuid)
                total += 1

            print(f"Deleted {total} objects from collection '{collection_name}'.")

    else:
        print(f"Collection '{collection_name}' does not exist. Nothing to delete.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Be sure to close the connection
    client.close()