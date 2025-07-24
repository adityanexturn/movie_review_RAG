from sentence_transformers import SentenceTransformer

# Load the pre-trained model only once
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(chunks):
    """
    Attach vector embeddings to each text chunk for retrieval.
    """
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(texts, show_progress_bar=True)  # Batched, efficient

    for i in range(len(chunks)):
        chunks[i]['embedding'] = vectors[i]  # Each embedding is a numpy array

    return chunks

# Standalone script: Test this file directly
if __name__ == "__main__":
    from loading import load_all_documents  # Adjust path if needed

    DATA_FOLDER = r"C:\Users\adity\Desktop\Gen-Ai Rag\review-data"
    chunks = load_all_documents(DATA_FOLDER)
    print(len(chunks))
    print(list(set([c["source"] for c in chunks])))
    chunks = generate_embeddings(chunks)
    print(f"Generated embeddings for {len(chunks)} chunks.")
    print("Shape of embedding vector (first chunk):", chunks[0]['embedding'].shape)
    print("First embedding vector (preview):", chunks[0]['embedding'][:5], "...")

import json
import numpy as np

# Suppose 'chunks' is your final list from the embedding function
# Each chunk: {'text': ..., 'source': ..., 'id': ..., 'embedding': np.array([...])}

to_save = []
for c in chunks:
    to_save.append({
        "text": c["text"],
        "source": c["source"],
        "id": c["id"],
        "embedding": c["embedding"].tolist()  # Convert numpy array to list
    })

with open("embeddings_output.json", "w", encoding="utf-8") as f:
    json.dump(to_save, f, ensure_ascii=False, indent=2)
    print("Saved embeddings to embeddings_output.json")

with open("embeddings_output.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    # Optionally convert embeddings back to numpy arrays when needed
    for item in data:
        item['embedding'] = np.array(item['embedding'])




