import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
# Use relative path for portability
DATA_FOLDER = r"C:\Users\adity\Desktop\Gen-Ai Rag\review-data"

# Load the pre-trained model only once
print("Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully.")

def generate_embeddings(chunks):
    """
    Attach vector embeddings to each text chunk for retrieval.
    Uses batched processing for efficiency.
    """
    if not chunks:
        print("Warning: No chunks provided for embedding generation.")
        return chunks
    
    print(f"Generating embeddings for {len(chunks)} text chunks...")
    
    # Extract text from chunks for batch processing
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings in batch (much more efficient than one-by-one)
    vectors = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    # Attach embeddings back to the original chunk dictionaries
    for i in range(len(chunks)):
        chunks[i]['embedding'] = vectors[i]  # Each embedding is a numpy array
    
    print(f"Successfully generated {len(chunks)} embeddings.")
    return chunks

def save_embeddings_to_json(chunks, output_file="embeddings_output.json"):
    """
    Save chunks with their embeddings to a JSON file.
    Converts numpy arrays to lists for JSON serialization.
    """
    print(f"Saving embeddings to '{output_file}'...")
    
    to_save = []
    for chunk in chunks:
        to_save.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "id": chunk["id"],
            "embedding": chunk["embedding"].tolist()  # Convert numpy array to list
        })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully saved {len(chunks)} embeddings to '{output_file}'.")

def load_embeddings_from_json(input_file="embeddings_output.json"):
    """
    Load chunks with embeddings from a JSON file.
    Converts embedding lists back to numpy arrays.
    """
    print(f"Loading embeddings from '{input_file}'...")
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert embeddings back to numpy arrays
        for item in data:
            item['embedding'] = np.array(item['embedding'])
        
        print(f"Successfully loaded {len(data)} embeddings from '{input_file}'.")
        return data
    
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{input_file}'.")
        return []

# Standalone script: Test this file directly
if __name__ == "__main__":
    # Import the loading function (should work with relative imports)
    try:
        from loading import load_all_documents
    except ImportError:
        print("Error: Could not import 'load_all_documents' from loading.py")
        print("Make sure loading.py is in the same directory as this script.")
        exit(1)
    
    print(f"=== EMBEDDING GENERATION PIPELINE ===")
    print(f"Data folder: '{DATA_FOLDER}'")
    
    # Step 1: Load documents and create chunks
    chunks = load_all_documents(DATA_FOLDER)
    
    if not chunks:
        print("No chunks were loaded. Please check your data folder.")
        exit(1)
    
    print(f"Loaded {len(chunks)} chunks from the following sources:")
    unique_sources = list(set([c["source"] for c in chunks]))
    for source in unique_sources:
        print(f"  - {source}")
    
    # Step 2: Generate embeddings
    chunks = generate_embeddings(chunks)
    
    # Step 3: Display sample information
    print(f"\n=== EMBEDDING DETAILS ===")
    print(f"Total chunks processed: {len(chunks)}")
    print(f"Embedding vector shape: {chunks[0]['embedding'].shape}")
    print(f"Embedding vector dimension: {len(chunks[0]['embedding'])}")
    print(f"First embedding preview: {chunks[0]['embedding'][:5]}...")
    
    # Step 4: Save embeddings to JSON
    save_embeddings_to_json(chunks)
    
    # Step 5: Test loading the saved embeddings (verification)
    print(f"\n=== VERIFICATION ===")
    loaded_chunks = load_embeddings_from_json()
    
    if loaded_chunks:
        print(f"Verification successful: Loaded {len(loaded_chunks)} chunks from JSON.")
        print(f"Embedding shape after reload: {loaded_chunks[0]['embedding'].shape}")
    else:
        print("Verification failed: Could not reload embeddings from JSON.")
    
    print(f"Your embeddings are ready for upload to Weaviate!")
