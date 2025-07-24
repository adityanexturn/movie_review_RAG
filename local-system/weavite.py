import weaviate
from weaviate import connect_to_local
from weaviate.collections.classes.config import DataType
import json
import numpy as np

# Connect to local Weaviate container
client = connect_to_local()

try:
    if not client.is_ready():
        raise RuntimeError("Weaviate is not running! Start Docker.")

    # Create collection if not exists
    if not client.collections.exists("Chunk"):
        client.collections.create(
            name="Chunk",
            properties=[
                {"name": "text", "data_type": DataType.TEXT},
                {"name": "source", "data_type": DataType.TEXT},
                {"name": "chunk_id", "data_type": DataType.TEXT}
            ],
            vectorizer_config=None  # Manual vector insertion
        )
        print("Created 'Chunk' collection.")
    else:
        print("'Chunk' collection already exists.")

    chunk_collection = client.collections.get("Chunk")

    # Check for existing chunks
    existing_objects = chunk_collection.query.fetch_objects(limit=1)
    
    if len(existing_objects.objects) > 0:
        print(f"Found existing chunks in the collection.")
        confirm = input("Do you want to clear all existing chunks before uploading new ones? (y/n): ").strip().lower()
        
        if confirm == "y":
            print("Clearing existing chunks...")
            # Fetch all objects to get their UUIDs
            all_objects = chunk_collection.query.fetch_objects(limit=10000)
            
            deleted_count = 0
            for obj in all_objects.objects:
                chunk_collection.data.delete_by_id(obj.uuid)
                deleted_count += 1
            
            print(f"Deleted {deleted_count} existing chunks.")
        else:
            print("Keeping existing chunks. New chunks will be added alongside them.")
    else:
        print("No existing chunks found in the collection.")

    # Load chunk embeddings
    try:
        with open("embeddings_output.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print("Error: 'embeddings_output.json' file not found!")
        exit(1)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in 'embeddings_output.json'!")
        exit(1)

    if not chunks:
        print("No chunks found in the embeddings file.")
        exit(0)

    print(f"Uploading {len(chunks)} chunks to Weaviate...")

    # Upload to Weaviate
    successful_uploads = 0
    for i, item in enumerate(chunks):
        try:
            vector = np.array(item['embedding'], dtype=np.float32)
            properties = {
                "text": item["text"],
                "source": item["source"],
                "chunk_id": item["id"]
            }
            chunk_collection.data.insert(
                properties=properties,
                vector=vector
            )
            successful_uploads += 1
            
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"Uploaded {i + 1}/{len(chunks)}")
                
        except Exception as e:
            print(f"Error uploading chunk {i + 1}: {e}")

    print(f"Successfully uploaded {successful_uploads}/{len(chunks)} chunks.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    client.close()
    print("Connection closed.")