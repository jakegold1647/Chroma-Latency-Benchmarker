import time
import random

def mock_chroma_query(collection_name, query_vector, n_results=5):
    """
    Simulates a query to ChromaDB for performance testing.
    """
    print(f"Initializing connection to collection: {collection_name}")
    # Simulate processing time
    processing_time = random.uniform(0.045, 0.065)
    time.sleep(processing_time)
    
    print(f"Query completed in {processing_time*1000:.2f}ms")
    return {
        "ids": [f"id_{i}" for i in range(n_results)],
        "distances": [random.random() for _ in range(n_results)],
        "metadatas": [{"source": "historical_archive"} for _ in range(n_results)]
    }

if __name__ == "__main__":
    # Example usage
    results = mock_chroma_query("property_dataset_v4", [0.1] * 1536)
    print(f"Retrieved {len(results['ids'])} results.")
