import weaviate
from weaviate.collections.classes.config import DataType
import json
import numpy as np
import os

# Your Weaviate Cloud credentials
WEAVIATE_CLUSTER_URL = "https://ombbyvzstfcnhbhxohyw.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "WURyQ0JlQWUvQ3UvcXdXdV9ON2VPL2R0UlJxWGpLaEowRE4wZ3RnQnhxSjFhSFNCNTB0SWJMcFQ3eUxRPV92MjAw"

# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_CLUSTER_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
)

if not client.is_ready():
    raise RuntimeError("Weaviate Cloud is not ready. Check your credentials.")

print("Successfully connected to Weaviate Cloud")

collection_name = "Chunk"

# Create collection if it doesn't exist
if not client.collections.exists(collection_name):
    client.collections.create(
        name=collection_name,
        properties=[
            {"name": "text", "data_type": DataType.TEXT},
            {"name": "source", "data_type": DataType.TEXT},
            {"name": "chunk_id", "data_type": DataType.TEXT}
        ],
        vectorizer_config=None  # We provide our own embeddings
    )
    print(f"Created '{collection_name}' collection.")
else:
    print(f"Collection '{collection_name}' already exists.")

chunk_collection = client.collections.get(collection_name)

# Load your embeddings from the JSON file
with open("embeddings_output.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Uploading {len(chunks)} chunks to Weaviate Cloud...")

# Upload each chunk
for i, item in enumerate(chunks):
    try:
        vector = np.array(item["embedding"], dtype=np.float32)
        properties = {
            "text": item["text"],
            "source": item["source"],
            "chunk_id": item["id"]
        }
        
        chunk_collection.data.insert(
            properties=properties,
            vector=vector
        )
        
        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"Uploaded {i + 1}/{len(chunks)} chunks")
            
    except Exception as e:
        print(f"Error uploading chunk {i + 1}: {e}")

client.close()
print("All chunks uploaded to Weaviate Cloud successfully!")
