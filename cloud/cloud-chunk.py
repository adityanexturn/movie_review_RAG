

import weaviate

# Your Weaviate Cloud credentials
WEAVIATE_CLUSTER_URL = "https://ombbyvzstfcnhbhxohyw.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "WURyQ0JlQWUvQ3UvcXdXdV9ON2VPL2R0UlJxWGpLaEowRE4wZ3RnQnhxSjFhSFNCNTB0SWJMcFQ3eUxRPV92MjAw"

def connect_to_weaviate():
    """Connect to Weaviate Cloud"""
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_CLUSTER_URL,
        auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
    )
    
    if not client.is_ready():
        raise RuntimeError("Weaviate Cloud is not ready. Check your credentials.")
    
    print("Successfully connected to Weaviate Cloud")
    return client

def check_chunks(client, collection_name="Chunk"):
    """Check how many chunks exist in the collection"""
    if not client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' does not exist.")
        return 0
    
    chunk_collection = client.collections.get(collection_name)
    
    try:
        result = chunk_collection.aggregate.over_all(total_count=True)
        total_count = result.total_count
        print(f"Found {total_count} chunks in collection '{collection_name}'.")
        return total_count
    except Exception as e:
        print(f"Error checking chunk count: {e}")
        return 0

def delete_all_chunks(client, collection_name="Chunk"):
    """Delete all chunks from the collection"""
    if not client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' does not exist. Nothing to delete.")
        return
    
    chunk_collection = client.collections.get(collection_name)
    
    try:
        print("Fetching all chunk IDs for deletion...")
        objects = chunk_collection.query.fetch_objects(
            limit=10000,
            return_metadata=["uuid"]
        )
        
        if not objects.objects:
            print("No chunks found to delete.")
            return
        
        total_deleted = 0
        for obj in objects.objects:
            uuid = obj.metadata["uuid"]
            chunk_collection.data.delete_by_id(uuid)
            total_deleted += 1
            
            if total_deleted % 50 == 0:
                print(f"Deleted {total_deleted} chunks...")
        
        print(f"Successfully deleted {total_deleted} chunks from collection '{collection_name}'.")
        
    except Exception as e:
        print(f"Error deleting chunks: {e}")

def main():
    """Main function for chunk management"""
    collection_name = "Chunk"
    
    client = connect_to_weaviate()
    
    try:
        # Check existing chunks
        chunk_count = check_chunks(client, collection_name)
        
        if chunk_count == 0:
            print("No chunks found. Nothing to delete.")
        else:
            # Ask user if they want to delete
            while True:
                choice = input(f"\nDo you want to delete all {chunk_count} chunks? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    delete_all_chunks(client, collection_name)
                    
                    # Verify deletion
                    print("\nVerifying deletion...")
                    final_count = check_chunks(client, collection_name)
                    if final_count == 0:
                        print("All chunks successfully deleted.")
                    else:
                        print(f"Warning: {final_count} chunks still remain.")
                    break
                    
                elif choice in ['n', 'no']:
                    print("Chunks preserved. No deletion performed.")
                    break
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
    
    finally:
        client.close()
        print("Connection to Weaviate Cloud closed.")

if __name__ == "__main__":
    main()
